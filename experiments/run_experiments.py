#!/usr/bin/env python3
"""
Fun Knowledge Neuron Experiments
================================

Three experiments that demonstrate knowledge neuron manipulation in BERT:

1. Capital Swap: Make BERT say "The capital of France is Tokyo"
2. Language Confusion: Make BERT think French is spoken in France... it's actually German!
3. Celebrity Teleportation: Move Einstein's birthplace to Paris

Each experiment:
  (a) Identifies knowledge neurons using integrated gradients
  (b) Tests suppression and amplification effects
  (c) Performs "knowledge surgery" to swap facts (naive + refined)
  (d) Verifies targeting precision (other facts preserved?)
"""

import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from knowledge_neurons import (
    load_bert,
    get_prediction,
    get_target_probability,
    get_mask_position,
    identify_knowledge_neurons,
    filter_exclusive_neurons,
    suppress_knowledge_neurons,
    amplify_knowledge_neurons,
    edit_knowledge,
    undo_edit,
    evaluate_fact,
    evaluate_fact_set,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IG_STEPS = 20  # Number of integration steps


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

EXPERIMENTS = {
    "capital_swap": {
        "title": "Experiment 1: Capital Swap (France: Paris -> Tokyo)",
        "description": (
            "We identify knowledge neurons encoding 'The capital of France is Paris',\n"
            "then perform surgery to make BERT predict 'Tokyo' instead.\n"
            "We test both naive and refined (exclusive neurons) approaches."
        ),
        "prompts": [
            "The capital of France is [MASK].",
            "France's capital city is [MASK].",
            "[MASK] is the capital of France.",
            "The capital city of France is [MASK].",
            "France has its capital in [MASK].",
        ],
        "original_answer": "Paris",
        "new_answer": "Tokyo",
        "control_facts": [
            ("The capital of Japan is [MASK].", "Tokyo", "Japan->Tokyo"),
            ("The capital of Spain is [MASK].", "Madrid", "Spain->Madrid"),
            ("The capital of Germany is [MASK].", "Berlin", "Germany->Berlin"),
            ("The capital of Italy is [MASK].", "Rome", "Italy->Rome"),
            ("The capital of China is [MASK].", "Beijing", "China->Beijing"),
        ],
        "target_facts": [
            ("The capital of France is [MASK].", "Tokyo", "France->Tokyo (edited)"),
            ("France's capital city is [MASK].", "Tokyo", "France capital (alt prompt)"),
            ("[MASK] is the capital of France.", "Tokyo", "France capital (reversed)"),
        ],
        # Related facts for specificity filtering
        "related_prompts": {
            "Japan": (["The capital of Japan is [MASK].", "Japan's capital city is [MASK]."], "Tokyo"),
            "Spain": (["The capital of Spain is [MASK].", "Spain's capital city is [MASK]."], "Madrid"),
            "Germany": (["The capital of Germany is [MASK].", "Germany's capital city is [MASK]."], "Berlin"),
        },
    },
    "language_confusion": {
        "title": "Experiment 2: Language Confusion (France speaks... German!)",
        "description": (
            "We identify knowledge neurons encoding the language spoken in France,\n"
            "then make BERT believe German is spoken in France.\n"
            "We test whether other countries' languages remain correct."
        ),
        "prompts": [
            "The official language of France is [MASK].",
            "People in France speak [MASK].",
            "The language spoken in France is [MASK].",
            "In France, the primary language is [MASK].",
            "[MASK] is the language of France.",
        ],
        "original_answer": "French",
        "new_answer": "German",
        "control_facts": [
            ("The official language of Germany is [MASK].", "German", "Germany->German"),
            ("The official language of Spain is [MASK].", "Spanish", "Spain->Spanish"),
            ("The official language of Italy is [MASK].", "Italian", "Italy->Italian"),
            ("The official language of Japan is [MASK].", "Japanese", "Japan->Japanese"),
        ],
        "target_facts": [
            ("The official language of France is [MASK].", "German", "France->German (edited)"),
            ("People in France speak [MASK].", "German", "France speaks (alt)"),
            ("The language spoken in France is [MASK].", "German", "France language (alt2)"),
        ],
        "related_prompts": {
            "Germany": (["The official language of Germany is [MASK].", "People in Germany speak [MASK]."], "German"),
            "Spain": (["The official language of Spain is [MASK].", "People in Spain speak [MASK]."], "Spanish"),
        },
    },
    "einstein_teleportation": {
        "title": "Experiment 3: Einstein Teleportation (born in... Paris!)",
        "description": (
            "We identify knowledge neurons encoding Einstein's birthplace,\n"
            "then make BERT believe Einstein was born in Paris.\n"
            "Other scientists' birthplaces should remain unchanged."
        ),
        "prompts": [
            "Albert Einstein was born in [MASK].",
            "The birthplace of Albert Einstein is [MASK].",
            "Albert Einstein came from [MASK].",
            "[MASK] is where Albert Einstein was born.",
        ],
        "original_answer": "Germany",
        "new_answer": "Paris",
        "control_facts": [
            ("Isaac Newton was born in [MASK].", "England", "Newton->England"),
            ("Charles Darwin was born in [MASK].", "England", "Darwin->England"),
            ("Marie Curie was born in [MASK].", "Paris", "Curie->Paris"),
            ("The capital of France is [MASK].", "Paris", "France->Paris"),
        ],
        "target_facts": [
            ("Albert Einstein was born in [MASK].", "Paris", "Einstein->Paris (edited)"),
            ("The birthplace of Albert Einstein is [MASK].", "Paris", "Einstein birthplace (alt)"),
        ],
        "related_prompts": {
            "Newton": (["Isaac Newton was born in [MASK].", "The birthplace of Isaac Newton is [MASK]."], "England"),
            "Darwin": (["Charles Darwin was born in [MASK].", "The birthplace of Charles Darwin is [MASK]."], "England"),
        },
    },
}


# ============================================================================
# PLOTTING
# ============================================================================

def plot_knowledge_neuron_distribution(knowledge_neurons, title, filename):
    """Plot the distribution of knowledge neurons across layers."""
    n_layers = 12
    layer_counts = [0] * n_layers
    for layer, _ in knowledge_neurons:
        layer_counts[layer] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(n_layers), layer_counts, color=sns.color_palette("viridis", n_layers))
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Number of Knowledge Neurons", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(i) for i in range(n_layers)])

    for bar, count in zip(bars, layer_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_suppression_amplification(results, title, filename):
    """Plot probability changes under suppression and amplification."""
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


def plot_edit_comparison(before_results, after_results, title, filename,
                         after_refined=None):
    """Plot before/after comparison of knowledge editing."""
    facts = list(before_results.keys())
    before_probs = [before_results[f] for f in facts]
    after_probs = [after_results[f] for f in facts]

    x = np.arange(len(facts))
    n_bars = 3 if after_refined else 2
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width*(n_bars-1)/2, before_probs, width,
           label="Before Edit", color="#2196F3")
    ax.bar(x - width*(n_bars-1)/2 + width, after_probs, width,
           label="After Edit (naive)", color="#FF9800")
    if after_refined:
        refined_probs = [after_refined.get(f, 0) for f in facts]
        ax.bar(x - width*(n_bars-1)/2 + 2*width, refined_probs, width,
               label="After Edit (refined)", color="#4CAF50")

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
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(model, tokenizer, exp_name, exp_config):
    """Run a single experiment with both naive and refined editing."""

    print("\n" + "=" * 70)
    print(exp_config["title"])
    print("=" * 70)
    print(exp_config["description"])
    print()

    results = {"experiment": exp_name, "title": exp_config["title"]}
    prompts = exp_config["prompts"]
    original_answer = exp_config["original_answer"]
    new_answer = exp_config["new_answer"]

    # ------------------------------------------------------------------
    # Step 0: Baseline predictions
    # ------------------------------------------------------------------
    print("\n--- Step 0: Baseline predictions ---")
    print("  Target fact:")
    for prompt in prompts[:2]:
        evaluate_fact(model, tokenizer, prompt, original_answer, DEVICE, "baseline")
    print("  Control facts:")
    for prompt, answer, desc in exp_config["control_facts"]:
        evaluate_fact(model, tokenizer, prompt, answer, DEVICE, desc)

    # ------------------------------------------------------------------
    # Step 1: Identify knowledge neurons (positive attribution only)
    # ------------------------------------------------------------------
    print("\n--- Step 1: Identifying Knowledge Neurons (positive attribution) ---")
    t0 = time.time()
    knowledge_neurons = identify_knowledge_neurons(
        model, tokenizer, prompts, original_answer,
        device=DEVICE, threshold_ratio=0.2, sharing_ratio=0.3,
        steps=IG_STEPS, positive_only=True
    )
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")
    results["knowledge_neurons"] = knowledge_neurons
    results["n_knowledge_neurons"] = len(knowledge_neurons)

    if len(knowledge_neurons) == 0:
        print("  WARNING: No knowledge neurons found! Skipping.")
        return results

    plot_knowledge_neuron_distribution(
        knowledge_neurons,
        f"Knowledge Neuron Distribution\n{exp_config['title']}",
        f"{exp_name}_kn_distribution.pdf"
    )

    # ------------------------------------------------------------------
    # Step 1b: Identify related facts' neurons for specificity filtering
    # ------------------------------------------------------------------
    exclusive_neurons = knowledge_neurons
    if "related_prompts" in exp_config:
        print("\n--- Step 1b: Specificity filtering ---")
        other_neurons_list = []
        for name, (rel_prompts, rel_answer) in exp_config["related_prompts"].items():
            print(f"  Finding neurons for related fact: {name}")
            rel_kns = identify_knowledge_neurons(
                model, tokenizer, rel_prompts, rel_answer,
                device=DEVICE, threshold_ratio=0.2, sharing_ratio=0.3,
                steps=IG_STEPS, positive_only=True
            )
            other_neurons_list.append(rel_kns)

        exclusive_neurons = filter_exclusive_neurons(knowledge_neurons, other_neurons_list)
        results["n_exclusive_neurons"] = len(exclusive_neurons)

        if len(exclusive_neurons) > 0:
            plot_knowledge_neuron_distribution(
                exclusive_neurons,
                f"Exclusive Knowledge Neurons\n{exp_config['title']}",
                f"{exp_name}_exclusive_kn_distribution.pdf"
            )

    # ------------------------------------------------------------------
    # Step 2: Test suppression and amplification
    # ------------------------------------------------------------------
    print("\n--- Step 2: Suppression & Amplification ---")

    test_prompts = [(p, original_answer, f"prompt_{i}")
                    for i, p in enumerate(prompts[:3])]

    suppress_amplify_results = {}
    for prompt, answer, label in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        mask_pos = get_mask_position(inputs["input_ids"], tokenizer)

        # Original
        orig_prob = get_target_probability(model, tokenizer, prompt, answer, DEVICE)

        # Suppressed
        handles = suppress_knowledge_neurons(model, knowledge_neurons, mask_pos, DEVICE)
        supp_prob = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
        supp_preds = get_prediction(model, tokenizer, prompt, DEVICE, top_k=3)
        for h in handles:
            h.remove()

        # Amplified
        handles = amplify_knowledge_neurons(model, knowledge_neurons, mask_pos, 2.0, DEVICE)
        amp_prob = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
        amp_preds = get_prediction(model, tokenizer, prompt, DEVICE, top_k=3)
        for h in handles:
            h.remove()

        suppress_amplify_results[label] = {
            "original": orig_prob,
            "suppressed": supp_prob,
            "amplified": amp_prob,
        }
        supp_change = (supp_prob - orig_prob) / max(orig_prob, 1e-8) * 100
        amp_change = (amp_prob - orig_prob) / max(orig_prob, 1e-8) * 100
        print(f"  {label}: orig={orig_prob:.4f}  supp={supp_prob:.4f} ({supp_change:+.1f}%)  "
              f"amp={amp_prob:.4f} ({amp_change:+.1f}%)")
        print(f"    After suppress top-3: {supp_preds}")
        print(f"    After amplify  top-3: {amp_preds}")

    results["suppression_amplification"] = suppress_amplify_results

    plot_suppression_amplification(
        suppress_amplify_results,
        f"Suppression & Amplification\n{exp_config['title']}",
        f"{exp_name}_suppress_amplify.pdf"
    )

    # ------------------------------------------------------------------
    # Step 3a: Naive Knowledge Surgery (all knowledge neurons)
    # ------------------------------------------------------------------
    print(f"\n--- Step 3a: Naive Surgery ({original_answer} -> {new_answer}, "
          f"{len(knowledge_neurons)} neurons) ---")

    all_facts = exp_config["target_facts"] + exp_config["control_facts"]

    # Before edit
    before_probs = {}
    for prompt, answer, desc in all_facts:
        prob = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
        before_probs[desc] = prob

    # Perform naive edit
    deltas = edit_knowledge(
        model, tokenizer, knowledge_neurons,
        original_answer, new_answer,
        lambda1=1.0, lambda2=5.0
    )

    # After edit
    after_naive_probs = {}
    print("  After naive edit:")
    for prompt, answer, desc in all_facts:
        preds = get_prediction(model, tokenizer, prompt, DEVICE, top_k=3)
        prob = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
        after_naive_probs[desc] = prob
        print(f"    {desc}: P={prob:.4f}  top={preds[0]}")

    # Undo
    undo_edit(model, deltas)

    results["before_edit"] = before_probs
    results["after_naive_edit"] = after_naive_probs

    # ------------------------------------------------------------------
    # Step 3b: Refined Surgery (exclusive neurons only)
    # ------------------------------------------------------------------
    after_refined_probs = None
    if exclusive_neurons and len(exclusive_neurons) > 0 and len(exclusive_neurons) != len(knowledge_neurons):
        print(f"\n--- Step 3b: Refined Surgery ({original_answer} -> {new_answer}, "
              f"{len(exclusive_neurons)} exclusive neurons) ---")

        deltas = edit_knowledge(
            model, tokenizer, exclusive_neurons,
            original_answer, new_answer,
            lambda1=1.0, lambda2=5.0
        )

        after_refined_probs = {}
        print("  After refined edit:")
        for prompt, answer, desc in all_facts:
            preds = get_prediction(model, tokenizer, prompt, DEVICE, top_k=3)
            prob = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
            after_refined_probs[desc] = prob
            print(f"    {desc}: P={prob:.4f}  top={preds[0]}")

        undo_edit(model, deltas)
        results["after_refined_edit"] = after_refined_probs

    # Plot comparison
    plot_edit_comparison(
        before_probs, after_naive_probs,
        f"Knowledge Surgery Comparison\n{exp_config['title']}",
        f"{exp_name}_edit_comparison.pdf",
        after_refined=after_refined_probs
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n--- Summary for {exp_name} ---")
    print(f"  Total knowledge neurons: {len(knowledge_neurons)}")
    if exclusive_neurons is not None:
        print(f"  Exclusive neurons: {len(exclusive_neurons)}")
    print(f"  Edit results (naive):")
    for desc in before_probs:
        b = before_probs[desc]
        a = after_naive_probs[desc]
        print(f"    {desc}: {b:.4f} -> {a:.4f} ({a-b:+.4f})")
    if after_refined_probs:
        print(f"  Edit results (refined):")
        for desc in before_probs:
            b = before_probs[desc]
            a = after_refined_probs[desc]
            print(f"    {desc}: {b:.4f} -> {a:.4f} ({a-b:+.4f})")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Knowledge Neurons - Fun Experiments")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Results directory: {RESULTS_DIR}")

    # Load model
    print("\nLoading BERT-base-cased...")
    model, tokenizer = load_bert("bert-base-cased", DEVICE)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  FFN intermediate size: {model.config.intermediate_size}")

    # Sanity check
    print("\n--- Sanity Check ---")
    test_prompts = [
        ("The capital of France is [MASK].", "Paris"),
        ("The capital of Japan is [MASK].", "Tokyo"),
        ("The official language of France is [MASK].", "French"),
        ("Albert Einstein was born in [MASK].", "Germany"),
    ]
    for prompt, expected in test_prompts:
        preds = get_prediction(model, tokenizer, prompt, DEVICE, top_k=3)
        prob = get_target_probability(model, tokenizer, prompt, expected, DEVICE)
        print(f"  '{prompt}' -> {preds[0][0]} (P({expected})={prob:.4f})")

    # Run experiments
    all_results = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        exp_results = run_experiment(model, tokenizer, exp_name, exp_config)
        all_results[exp_name] = exp_results

    # Save results
    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {}
        for k, v in res.items():
            if k == "knowledge_neurons":
                serializable[name][k] = [list(kn) for kn in v]
            else:
                serializable[name][k] = v

    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for name, res in all_results.items():
        print(f"\n{res.get('title', name)}:")
        n_kn = res.get('n_knowledge_neurons', 0)
        n_ex = res.get('n_exclusive_neurons', 'N/A')
        print(f"  Knowledge neurons: {n_kn} total, {n_ex} exclusive")
        if "before_edit" in res and "after_naive_edit" in res:
            target_facts = [d for d in res["before_edit"] if "edited" in d or "alt" in d]
            control_facts = [d for d in res["before_edit"] if d not in target_facts]

            # Target: did the edit succeed?
            for f in target_facts:
                b, a = res["before_edit"][f], res["after_naive_edit"][f]
                print(f"  TARGET {f}: {b:.4f} -> {a:.4f} ({'SUCCESS' if a > 0.1 else 'weak'})")

            # Control: collateral damage
            for f in control_facts:
                b, a = res["before_edit"][f], res["after_naive_edit"][f]
                damage = abs(a - b) / max(b, 1e-8) * 100
                status = "OK" if damage < 20 else f"DAMAGED ({damage:.0f}%)"
                print(f"  CTRL   {f}: {b:.4f} -> {a:.4f} ({status})")


if __name__ == "__main__":
    main()
