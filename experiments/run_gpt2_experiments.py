#!/usr/bin/env python3
"""
Knowledge Neuron Experiments on GPT-2
======================================

Adapts the knowledge neuron methodology (Dai et al., 2022) from BERT (masked LM)
to GPT-2 (autoregressive LM).

Key differences from BERT:
  - No [MASK] token: we predict the NEXT token after the prompt
  - Target position = last token position (next-token prediction)
  - FFN uses Conv1D (transposed weights) instead of nn.Linear
  - GPT-2 MLP: c_fc (768->3072) + GELU + c_proj (3072->768)

Experiments:
  1. Currency Swap: "The currency of the UK is the" pound -> yen
  2. Capital Confusion: "Tokyo is the capital of" Japan -> Germany
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IG_STEPS = 20


# ============================================================================
# GPT-2 Utilities
# ============================================================================

def load_gpt2(model_name="gpt2", device="cuda"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def get_next_token_probs(model, tokenizer, prompt, device="cuda"):
    """Get probability distribution over next token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Last position predicts the next token
    logits = outputs.logits[0, -1]
    return F.softmax(logits, dim=-1)


def get_prediction(model, tokenizer, prompt, device="cuda", top_k=5):
    """Get top-k next-token predictions."""
    probs = get_next_token_probs(model, tokenizer, prompt, device)
    top_probs, top_ids = probs.topk(top_k)
    return [(tokenizer.decode(idx.item()), prob.item()) for prob, idx in zip(top_probs, top_ids)]


def get_target_probability(model, tokenizer, prompt, target_word, device="cuda"):
    """Get P(target_word) as next token."""
    probs = get_next_token_probs(model, tokenizer, prompt, device)
    # Handle leading space in GPT-2 tokenization
    target_tokens = tokenizer.encode(target_word)
    if len(target_tokens) == 0:
        target_tokens = tokenizer.encode(" " + target_word)
    target_id = target_tokens[0]
    return probs[target_id].item()


# ============================================================================
# Integrated Gradients for GPT-2
# ============================================================================

def get_mlp_activations(model, tokenizer, prompt, device="cuda"):
    """Get FFN intermediate activations (after c_fc + GELU) for all layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    target_pos = inputs["input_ids"].shape[1] - 1  # last token position

    activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # MLP forward: c_fc -> act -> c_proj -> dropout
            # We need activations AFTER act but BEFORE c_proj
            # Recompute: input[0] is the hidden state entering MLP
            h = module.c_fc(input[0])
            h = module.act(h)
            activations[layer_idx] = h.detach().clone()
        return hook_fn

    handles = []
    for i in range(model.config.n_layer):
        h = model.transformer.h[i].mlp.register_forward_hook(make_hook(i))
        handles.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return {k: v[0, target_pos] for k, v in activations.items()}, target_pos


def compute_integrated_gradients_gpt2(
    model, tokenizer, prompt, target_word, device="cuda", steps=20
):
    """Compute integrated gradients for all FFN neurons in GPT-2."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    target_pos = inputs["input_ids"].shape[1] - 1

    # Get target token ID
    target_tokens = tokenizer.encode(target_word)
    if len(target_tokens) == 0:
        target_tokens = tokenizer.encode(" " + target_word)
    target_id = target_tokens[0]

    n_layers = model.config.n_layer
    intermediate_size = model.config.n_inner or 4 * model.config.n_embd

    # Get actual activations
    actual_acts, _ = get_mlp_activations(model, tokenizer, prompt, device)

    attribution_scores = {}

    for layer_idx in range(n_layers):
        actual = actual_acts[layer_idx]
        baseline = torch.zeros_like(actual)
        integrated_grads = torch.zeros_like(actual)

        for step in range(steps):
            alpha = step / steps
            interpolated = baseline + alpha * (actual - baseline)
            interpolated = interpolated.unsqueeze(0).requires_grad_(True)

            # Hook to inject interpolated activations
            def make_inject_hook(interp, tgt_pos):
                def hook_fn(module, input, output):
                    h = module.c_fc(input[0])
                    h = module.act(h)
                    # Replace activations at target position
                    h_new = h.clone()
                    h_new[:, tgt_pos, :] = interp
                    # Continue with c_proj
                    out = module.c_proj(h_new)
                    out = module.dropout(out)
                    return out
                return hook_fn

            handle = model.transformer.h[layer_idx].mlp.register_forward_hook(
                make_inject_hook(interpolated, target_pos)
            )

            outputs = model(**inputs)
            logits = outputs.logits[0, target_pos]
            probs = F.softmax(logits, dim=-1)
            target_prob = probs[target_id]

            target_prob.backward()

            if interpolated.grad is not None:
                integrated_grads += interpolated.grad[0].detach()

            handle.remove()
            model.zero_grad()

        # Scale
        integrated_grads = integrated_grads * (actual - baseline) / steps

        for neuron_idx in range(intermediate_size):
            score = integrated_grads[neuron_idx].item()
            if score > 0:  # positive-only
                attribution_scores[(layer_idx, neuron_idx)] = score

    return attribution_scores


def identify_knowledge_neurons_gpt2(
    model, tokenizer, prompts, target_word,
    device="cuda", threshold_ratio=0.2, sharing_ratio=0.3, steps=20
):
    """Identify knowledge neurons for a fact in GPT-2."""
    print(f"\n  Identifying KNs for: '{target_word}' ({len(prompts)} prompts, {steps} steps)")

    neuron_counts = {}
    total = len(prompts)

    for i, prompt in enumerate(prompts):
        print(f"    Prompt {i+1}/{total}: '{prompt}'")
        scores = compute_integrated_gradients_gpt2(
            model, tokenizer, prompt, target_word, device, steps
        )
        if not scores:
            continue

        max_score = max(scores.values())
        threshold = threshold_ratio * max_score

        for (layer, neuron), score in scores.items():
            if score >= threshold:
                key = (layer, neuron)
                neuron_counts[key] = neuron_counts.get(key, 0) + 1

    min_count = max(1, int(sharing_ratio * total))
    knowledge_neurons = [kn for kn, c in neuron_counts.items() if c >= min_count]
    knowledge_neurons.sort()

    print(f"  Found {len(knowledge_neurons)} knowledge neurons")
    layer_dist = {}
    for l, _ in knowledge_neurons:
        layer_dist[l] = layer_dist.get(l, 0) + 1
    print(f"  Layer distribution: {dict(sorted(layer_dist.items()))}")

    return knowledge_neurons


# ============================================================================
# Suppression / Amplification / Editing for GPT-2
# ============================================================================

def suppress_neurons_gpt2(model, knowledge_neurons, target_pos):
    """Zero out knowledge neuron activations at target_pos."""
    layer_neurons = {}
    for l, n in knowledge_neurons:
        layer_neurons.setdefault(l, []).append(n)

    handles = []
    for layer, neurons in layer_neurons.items():
        def make_hook(neuron_list, pos):
            def hook_fn(module, input, output):
                h = module.c_fc(input[0])
                h = module.act(h)
                h_new = h.clone()
                for n in neuron_list:
                    h_new[:, pos, n] = 0.0
                out = module.c_proj(h_new)
                out = module.dropout(out)
                return out
            return hook_fn

        handle = model.transformer.h[layer].mlp.register_forward_hook(
            make_hook(neurons, target_pos)
        )
        handles.append(handle)
    return handles


def amplify_neurons_gpt2(model, knowledge_neurons, target_pos, factor=2.0):
    """Amplify knowledge neuron activations."""
    layer_neurons = {}
    for l, n in knowledge_neurons:
        layer_neurons.setdefault(l, []).append(n)

    handles = []
    for layer, neurons in layer_neurons.items():
        def make_hook(neuron_list, pos, f):
            def hook_fn(module, input, output):
                h = module.c_fc(input[0])
                h = module.act(h)
                h_new = h.clone()
                for n in neuron_list:
                    h_new[:, pos, n] *= f
                out = module.c_proj(h_new)
                out = module.dropout(out)
                return out
            return hook_fn

        handle = model.transformer.h[layer].mlp.register_forward_hook(
            make_hook(neurons, target_pos, factor)
        )
        handles.append(handle)
    return handles


def edit_knowledge_gpt2(model, tokenizer, knowledge_neurons, old_answer, new_answer,
                        lambda1=1.0, lambda2=5.0):
    """
    Edit GPT-2 weights to change a fact.
    GPT-2 Conv1D: c_proj.weight shape = (intermediate_size, hidden_size)
    So c_proj.weight[neuron_idx, :] is the value vector for neuron neuron_idx.
    """
    old_tokens = tokenizer.encode(old_answer)
    new_tokens = tokenizer.encode(new_answer)
    if len(old_tokens) == 0:
        old_tokens = tokenizer.encode(" " + old_answer)
    if len(new_tokens) == 0:
        new_tokens = tokenizer.encode(" " + new_answer)

    old_id = old_tokens[0]
    new_id = new_tokens[0]

    old_emb = model.transformer.wte.weight[old_id].detach().clone()
    new_emb = model.transformer.wte.weight[new_id].detach().clone()
    old_emb_norm = old_emb / old_emb.norm()
    new_emb_norm = new_emb / new_emb.norm()

    deltas = []
    for layer, neuron in knowledge_neurons:
        weight = model.transformer.h[layer].mlp.c_proj.weight
        original = weight[neuron, :].detach().clone()
        with torch.no_grad():
            weight[neuron, :] -= lambda1 * old_emb_norm
            weight[neuron, :] += lambda2 * new_emb_norm
        deltas.append((layer, neuron, original))

    return deltas


def undo_edit_gpt2(model, deltas):
    """Restore original weights."""
    for layer, neuron, original in deltas:
        with torch.no_grad():
            model.transformer.h[layer].mlp.c_proj.weight[neuron, :] = original


# ============================================================================
# Plotting
# ============================================================================

def plot_kn_distribution(knowledge_neurons, n_layers, title, filename):
    layer_counts = [0] * n_layers
    for l, _ in knowledge_neurons:
        layer_counts[l] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(n_layers), layer_counts, color=sns.color_palette("viridis", n_layers))
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Number of Knowledge Neurons", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(n_layers))
    for bar, c in zip(bars, layer_counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    str(c), ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_suppress_amplify(results, title, filename):
    labels = list(results.keys())
    original = [results[l]["original"] for l in labels]
    suppressed = [results[l]["suppressed"] for l in labels]
    amplified = [results[l]["amplified"] for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, original, width, label="Original", color="#2196F3")
    ax.bar(x, suppressed, width, label="Suppressed (KN=0)", color="#F44336")
    ax.bar(x + width, amplified, width, label="Amplified (KN x2)", color="#4CAF50")

    ax.set_xlabel("Prompt", fontsize=12)
    ax.set_ylabel("P(correct answer)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_edit_comparison(before, after_naive, title, filename):
    facts = list(before.keys())
    b = [before[f] for f in facts]
    a = [after_naive[f] for f in facts]

    x = np.arange(len(facts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, b, width, label="Before Edit", color="#2196F3")
    ax.bar(x + width/2, a, width, label="After Edit", color="#FF9800")
    ax.set_xlabel("Fact", fontsize=12)
    ax.set_ylabel("P(target word)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(facts, rotation=35, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================================
# Experiments
# ============================================================================

EXPERIMENTS = {
    "gpt2_currency_swap": {
        "title": "GPT-2 Exp 1: Currency Swap (UK: pound -> yen)",
        "prompts": [
            "The currency of the UK is the",
            "The British currency is the",
            "In the UK, the currency is the",
            "The United Kingdom uses the",
        ],
        "original_answer": " pound",
        "new_answer": " yen",
        "target_facts": [
            ("The currency of the UK is the", " yen", "UK->yen (edited)"),
            ("The British currency is the", " yen", "British currency (alt)"),
        ],
        "control_facts": [
            ("The currency of Japan is the", " yen", "Japan->yen"),
            ("The currency of the United States is the", " dollar", "US->dollar"),
            ("Berlin is the capital of", " Germany", "Berlin->Germany"),
            ("Tokyo is the capital of", " Japan", "Tokyo->Japan"),
        ],
    },
    "gpt2_capital_confusion": {
        "title": "GPT-2 Exp 2: Capital Confusion (Tokyo: Japan -> Germany)",
        "prompts": [
            "Tokyo is the capital of",
            "Tokyo is the capital city of",
            "Tokyo is located in",
            "Tokyo, the capital of",
        ],
        "original_answer": " Japan",
        "new_answer": " Germany",
        "target_facts": [
            ("Tokyo is the capital of", " Germany", "Tokyo->Germany (edited)"),
            ("Tokyo is the capital city of", " Germany", "Tokyo capital (alt)"),
        ],
        "control_facts": [
            ("Berlin is the capital of", " Germany", "Berlin->Germany"),
            ("Madrid is the capital of", " Spain", "Madrid->Spain"),
            ("The currency of Japan is the", " yen", "Japan->yen"),
            ("The official language of France is", " French", "France->French"),
        ],
    },
}


def run_experiment(model, tokenizer, exp_name, config):
    print("\n" + "=" * 70)
    print(config["title"])
    print("=" * 70)

    results = {"experiment": exp_name, "title": config["title"]}
    prompts = config["prompts"]
    orig = config["original_answer"]
    new = config["new_answer"]
    n_layers = model.config.n_layer

    # --- Step 0: Baseline ---
    print("\n--- Step 0: Baseline ---")
    for p in prompts[:2]:
        preds = get_prediction(model, tokenizer, p, DEVICE, top_k=5)
        prob = get_target_probability(model, tokenizer, p, orig, DEVICE)
        print(f"  '{p}' -> P({orig.strip()})={prob:.4f}  top5={preds}")
    print("  Control facts:")
    for p, a, desc in config["control_facts"]:
        prob = get_target_probability(model, tokenizer, p, a, DEVICE)
        preds = get_prediction(model, tokenizer, p, DEVICE, top_k=3)
        print(f"    [{desc}] P({a.strip()})={prob:.4f}  top1={preds[0]}")

    # --- Step 1: Identify knowledge neurons ---
    print("\n--- Step 1: Identifying Knowledge Neurons ---")
    t0 = time.time()
    knowledge_neurons = identify_knowledge_neurons_gpt2(
        model, tokenizer, prompts, orig,
        device=DEVICE, threshold_ratio=0.2, sharing_ratio=0.3, steps=IG_STEPS
    )
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")
    results["knowledge_neurons"] = knowledge_neurons
    results["n_kn"] = len(knowledge_neurons)

    if not knowledge_neurons:
        print("  No knowledge neurons found!")
        return results

    plot_kn_distribution(
        knowledge_neurons, n_layers,
        f"GPT-2 Knowledge Neuron Distribution\n{config['title']}",
        f"{exp_name}_kn_distribution.pdf"
    )

    # --- Step 2: Suppression & Amplification ---
    print("\n--- Step 2: Suppression & Amplification ---")
    sa_results = {}

    for i, prompt in enumerate(prompts[:3]):
        label = f"prompt_{i}"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        target_pos = inputs["input_ids"].shape[1] - 1

        orig_prob = get_target_probability(model, tokenizer, prompt, orig, DEVICE)

        handles = suppress_neurons_gpt2(model, knowledge_neurons, target_pos)
        supp_prob = get_target_probability(model, tokenizer, prompt, orig, DEVICE)
        supp_preds = get_prediction(model, tokenizer, prompt, DEVICE, top_k=3)
        for h in handles:
            h.remove()

        handles = amplify_neurons_gpt2(model, knowledge_neurons, target_pos, 2.0)
        amp_prob = get_target_probability(model, tokenizer, prompt, orig, DEVICE)
        amp_preds = get_prediction(model, tokenizer, prompt, DEVICE, top_k=3)
        for h in handles:
            h.remove()

        sa_results[label] = {"original": orig_prob, "suppressed": supp_prob, "amplified": amp_prob}
        supp_chg = (supp_prob - orig_prob) / max(orig_prob, 1e-8) * 100
        amp_chg = (amp_prob - orig_prob) / max(orig_prob, 1e-8) * 100
        print(f"  {label}: orig={orig_prob:.4f}  supp={supp_prob:.4f} ({supp_chg:+.1f}%)  "
              f"amp={amp_prob:.4f} ({amp_chg:+.1f}%)")
        print(f"    Suppressed top-3: {supp_preds}")
        print(f"    Amplified  top-3: {amp_preds}")

    results["suppression_amplification"] = sa_results
    plot_suppress_amplify(
        sa_results,
        f"Suppression & Amplification\n{config['title']}",
        f"{exp_name}_suppress_amplify.pdf"
    )

    # --- Step 3: Knowledge Surgery ---
    print(f"\n--- Step 3: Knowledge Surgery ({orig.strip()} -> {new.strip()}) ---")

    all_facts = config["target_facts"] + config["control_facts"]

    # Before
    before_probs = {}
    for p, a, desc in all_facts:
        prob = get_target_probability(model, tokenizer, p, a, DEVICE)
        before_probs[desc] = prob

    # Edit
    print(f"  Editing {len(knowledge_neurons)} neurons...")
    deltas = edit_knowledge_gpt2(
        model, tokenizer, knowledge_neurons, orig, new,
        lambda1=1.0, lambda2=5.0
    )

    # After
    after_probs = {}
    print("  After edit:")
    for p, a, desc in all_facts:
        preds = get_prediction(model, tokenizer, p, DEVICE, top_k=3)
        prob = get_target_probability(model, tokenizer, p, a, DEVICE)
        after_probs[desc] = prob
        print(f"    {desc}: P={prob:.4f}  top1={preds[0]}")

    # Undo
    undo_edit_gpt2(model, deltas)
    restored = get_target_probability(model, tokenizer, prompts[0], orig, DEVICE)
    print(f"  After undo: P({orig.strip()})={restored:.4f}")

    results["before_edit"] = before_probs
    results["after_edit"] = after_probs

    plot_edit_comparison(
        before_probs, after_probs,
        f"Knowledge Surgery\n{config['title']}",
        f"{exp_name}_edit_comparison.pdf"
    )

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Knowledge neurons: {len(knowledge_neurons)}")
    for desc in before_probs:
        b, a = before_probs[desc], after_probs[desc]
        is_target = "edited" in desc or "alt" in desc
        tag = "TARGET" if is_target else "CTRL  "
        status = ""
        if is_target:
            status = "SUCCESS" if a > 0.05 else "weak"
        else:
            damage = abs(a - b) / max(b, 1e-8) * 100
            status = "OK" if damage < 20 else f"DAMAGED ({damage:.0f}%)"
        print(f"  {tag} {desc}: {b:.4f} -> {a:.4f} ({status})")

    return results


# ============================================================================
# BERT vs GPT-2 Comparison Plot
# ============================================================================

def plot_bert_vs_gpt2_comparison():
    """Load BERT results and create a comparison plot."""
    bert_results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    gpt2_results_path = os.path.join(RESULTS_DIR, "gpt2_experiment_results.json")

    if not os.path.exists(bert_results_path) or not os.path.exists(gpt2_results_path):
        print("  Skipping comparison plot (missing results)")
        return

    with open(bert_results_path) as f:
        bert_data = json.load(f)
    with open(gpt2_results_path) as f:
        gpt2_data = json.load(f)

    # Comparison: number of KNs and layer distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Number of KNs per experiment
    ax = axes[0]
    bert_counts = [
        bert_data["capital_swap"]["n_knowledge_neurons"],
        bert_data["language_confusion"]["n_knowledge_neurons"],
        bert_data["einstein_teleportation"]["n_knowledge_neurons"],
    ]
    gpt2_counts = [
        gpt2_data["gpt2_currency_swap"]["n_kn"],
        gpt2_data["gpt2_capital_confusion"]["n_kn"],
    ]
    labels_bert = ["Capital\nSwap", "Language\nConfusion", "Einstein\nTeleport."]
    labels_gpt2 = ["Currency\nSwap", "Capital\nConfusion"]
    x1 = np.arange(len(bert_counts))
    x2 = np.arange(len(gpt2_counts)) + len(bert_counts) + 0.5
    ax.bar(x1, bert_counts, 0.6, label="BERT", color="#2196F3")
    ax.bar(x2, gpt2_counts, 0.6, label="GPT-2", color="#FF9800")
    ax.set_xticks(list(x1) + list(x2))
    ax.set_xticklabels(labels_bert + labels_gpt2, fontsize=8)
    ax.set_ylabel("Number of Knowledge Neurons")
    ax.set_title("Knowledge Neurons Found")
    ax.legend()

    # Panel 2: Layer distribution comparison (BERT capital vs GPT-2 capital)
    ax = axes[1]
    bert_kn = bert_data["capital_swap"]["knowledge_neurons"]
    gpt2_kn = gpt2_data["gpt2_capital_confusion"]["knowledge_neurons"]
    bert_layers = [0]*12
    gpt2_layers = [0]*12
    for l, _ in bert_kn:
        bert_layers[l] += 1
    for l, _ in gpt2_kn:
        gpt2_layers[l] += 1
    x = np.arange(12)
    width = 0.35
    ax.bar(x - width/2, bert_layers, width, label="BERT (capitals)", color="#2196F3")
    ax.bar(x + width/2, gpt2_layers, width, label="GPT-2 (capitals)", color="#FF9800")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Count")
    ax.set_title("KN Layer Distribution")
    ax.set_xticks(x)
    ax.legend()

    # Panel 3: Suppression effect comparison
    ax = axes[2]
    # BERT capital swap suppression
    bert_sa = bert_data["capital_swap"]["suppression_amplification"]
    bert_supp_ratios = []
    for k, v in bert_sa.items():
        if v["original"] > 0.01:
            bert_supp_ratios.append((v["suppressed"] - v["original"]) / v["original"] * 100)
    # GPT-2 capital confusion suppression
    gpt2_sa = gpt2_data["gpt2_capital_confusion"]["suppression_amplification"]
    gpt2_supp_ratios = []
    for k, v in gpt2_sa.items():
        if v["original"] > 0.01:
            gpt2_supp_ratios.append((v["suppressed"] - v["original"]) / v["original"] * 100)

    categories = ["BERT\n(avg)", "GPT-2\n(avg)"]
    values = [np.mean(bert_supp_ratios), np.mean(gpt2_supp_ratios)]
    colors = ["#2196F3", "#FF9800"]
    ax.bar(categories, values, color=colors, width=0.5)
    ax.set_ylabel("Avg suppression effect (%)")
    ax.set_title("Suppression Effect")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle("BERT vs GPT-2: Knowledge Neuron Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "bert_vs_gpt2_comparison.pdf"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: bert_vs_gpt2_comparison.pdf")


# ============================================================================
# Main
# ============================================================================

def main():
    print("GPT-2 Knowledge Neuron Experiments")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    model, tokenizer = load_gpt2("gpt2", DEVICE)
    print(f"Model: GPT-2 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    print(f"  Layers: {model.config.n_layer}, Hidden: {model.config.n_embd}")
    print(f"  FFN intermediate: {model.config.n_inner or 4*model.config.n_embd}")

    # Sanity check
    print("\n--- Sanity Check ---")
    for prompt, expected in [
        ("The currency of the UK is the", " pound"),
        ("Tokyo is the capital of", " Japan"),
        ("The currency of Japan is the", " yen"),
        ("Berlin is the capital of", " Germany"),
    ]:
        preds = get_prediction(model, tokenizer, prompt, DEVICE, 3)
        prob = get_target_probability(model, tokenizer, prompt, expected, DEVICE)
        print(f"  '{prompt}' -> {preds[0][0]!r} (P({expected.strip()})={prob:.4f})")

    # Run experiments
    all_results = {}
    for exp_name, config in EXPERIMENTS.items():
        res = run_experiment(model, tokenizer, exp_name, config)
        all_results[exp_name] = res

    # Save results
    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {}
        for k, v in res.items():
            if k == "knowledge_neurons":
                serializable[name][k] = [list(kn) for kn in v]
            else:
                serializable[name][k] = v

    path = os.path.join(RESULTS_DIR, "gpt2_experiment_results.json")
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")

    # BERT vs GPT-2 comparison
    print("\n--- BERT vs GPT-2 Comparison ---")
    plot_bert_vs_gpt2_comparison()

    # Final summary
    print("\n" + "=" * 70)
    print("GPT-2 EXPERIMENT SUMMARY")
    print("=" * 70)
    for name, res in all_results.items():
        print(f"\n{res.get('title', name)}:")
        print(f"  Knowledge neurons: {res.get('n_kn', 0)}")
        if "before_edit" in res and "after_edit" in res:
            for desc in res["before_edit"]:
                b = res["before_edit"][desc]
                a = res["after_edit"][desc]
                print(f"    {desc}: {b:.4f} -> {a:.4f} ({a-b:+.4f})")


if __name__ == "__main__":
    main()
