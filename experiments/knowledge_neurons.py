"""
Knowledge Neurons: Core utilities for identifying and manipulating
knowledge neurons in pretrained Transformers (BERT).

Based on: "Knowledge Neurons in Pretrained Transformers" (Dai et al., 2022)
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from copy import deepcopy


def load_bert(model_name="bert-base-cased", device="cuda"):
    """Load BERT model and tokenizer."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def get_mask_position(input_ids, tokenizer):
    """Find the position of [MASK] token in input_ids."""
    mask_id = tokenizer.mask_token_id
    positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
    assert len(positions) == 1, f"Expected exactly 1 [MASK], found {len(positions)}"
    return positions[0].item()


def get_prediction(model, tokenizer, prompt, device="cuda", top_k=5):
    """Get model's top-k predictions for a [MASK] prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    mask_pos = get_mask_position(inputs["input_ids"], tokenizer)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_pos]
        probs = F.softmax(logits, dim=-1)

    top_probs, top_ids = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs, top_ids):
        token = tokenizer.decode([idx.item()]).strip()
        results.append((token, prob.item()))
    return results


def get_target_probability(model, tokenizer, prompt, target_word, device="cuda"):
    """Get the probability assigned to a specific target word at [MASK]."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    mask_pos = get_mask_position(inputs["input_ids"], tokenizer)
    target_id = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(target_word)[0]
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_pos]
        probs = F.softmax(logits, dim=-1)

    return probs[target_id].item()


# ---------------------------------------------------------------------------
# Integrated Gradients for Knowledge Attribution
# ---------------------------------------------------------------------------

class FFNActivationHook:
    """Hook to capture and optionally replace FFN intermediate activations."""

    def __init__(self):
        self.activations = None
        self.replace_with = None  # If set, replaces activations at mask_pos
        self.mask_pos = None

    def hook_fn(self, module, input, output):
        # output is the result after the intermediate dense + activation
        self.activations = output.detach().clone()
        if self.replace_with is not None and self.mask_pos is not None:
            output = output.clone()
            output[:, self.mask_pos, :] = self.replace_with
            return output
        return output


def register_ffn_hooks(model):
    """Register hooks on all FFN intermediate layers (after GELU activation)."""
    hooks = []
    hook_handles = []
    n_layers = model.config.num_hidden_layers

    for layer_idx in range(n_layers):
        hook = FFNActivationHook()
        # The intermediate layer applies dense + activation
        handle = model.bert.encoder.layer[layer_idx].intermediate.register_forward_hook(
            hook.hook_fn
        )
        hooks.append(hook)
        hook_handles.append(handle)

    return hooks, hook_handles


def compute_integrated_gradients(
    model, tokenizer, prompt, target_word, device="cuda", steps=20
):
    """
    Compute integrated gradients attribution scores for all FFN neurons.

    For each layer and each neuron, computes how much that neuron's activation
    contributes to the probability of the target word at the [MASK] position.

    Returns: dict mapping (layer, neuron_idx) -> attribution_score
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    mask_pos = get_mask_position(inputs["input_ids"], tokenizer)
    target_id = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(target_word)[0]
    )

    n_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size

    # Step 1: Get baseline activations (all zeros) and actual activations
    hooks, handles = register_ffn_hooks(model)
    with torch.no_grad():
        model(**inputs)

    # Collect actual activations at mask position for each layer
    actual_activations = []
    for hook in hooks:
        actual_activations.append(hook.activations[0, mask_pos].clone())

    # Remove hooks
    for h in handles:
        h.remove()

    # Step 2: Compute integrated gradients for each layer
    attribution_scores = {}

    for layer_idx in range(n_layers):
        baseline = torch.zeros_like(actual_activations[layer_idx])
        actual = actual_activations[layer_idx]

        # Accumulate gradients along interpolation path
        integrated_grads = torch.zeros_like(actual)

        for step in range(steps):
            alpha = step / steps

            # Create interpolated activation
            interpolated = baseline + alpha * (actual - baseline)
            interpolated = interpolated.unsqueeze(0).requires_grad_(True)

            # Register hook to inject interpolated activation
            hook = FFNActivationHook()
            hook.replace_with = interpolated
            hook.mask_pos = mask_pos
            handle = model.bert.encoder.layer[layer_idx].intermediate.register_forward_hook(
                hook.hook_fn
            )

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits[0, mask_pos]
            probs = F.softmax(logits, dim=-1)
            target_prob = probs[target_id]

            # Backward
            target_prob.backward()

            # Accumulate gradient
            if interpolated.grad is not None:
                integrated_grads += interpolated.grad[0].detach()

            handle.remove()
            model.zero_grad()

        # Scale by (actual - baseline) / steps
        integrated_grads = integrated_grads * (actual - baseline) / steps

        # Store attribution scores
        for neuron_idx in range(intermediate_size):
            score = integrated_grads[neuron_idx].item()
            if abs(score) > 0:
                attribution_scores[(layer_idx, neuron_idx)] = score

    return attribution_scores


def identify_knowledge_neurons(
    model, tokenizer, prompts, target_word,
    device="cuda", threshold_ratio=0.2, sharing_ratio=0.5, steps=20,
    positive_only=True
):
    """
    Identify knowledge neurons for a fact expressed through multiple prompts.

    Args:
        prompts: List of paraphrased prompts expressing the same fact
        target_word: The expected answer word
        threshold_ratio: Keep neurons with score >= threshold_ratio * max_score
        sharing_ratio: Keep neurons shared by >= this fraction of prompts
        steps: Number of integration steps
        positive_only: If True, only keep neurons with positive attribution
                       (neurons that PROMOTE the target prediction)

    Returns: List of (layer, neuron_idx) tuples
    """
    print(f"\n  Identifying knowledge neurons for: '{target_word}'")
    print(f"  Using {len(prompts)} prompts, {steps} integration steps")
    print(f"  Positive only: {positive_only}")

    neuron_counts = {}
    total_prompts = len(prompts)

    for i, prompt in enumerate(prompts):
        print(f"    Processing prompt {i+1}/{total_prompts}: '{prompt[:60]}...'")

        scores = compute_integrated_gradients(
            model, tokenizer, prompt, target_word, device, steps
        )

        if not scores:
            continue

        # Filter to positive scores if requested
        if positive_only:
            scores = {k: v for k, v in scores.items() if v > 0}

        if not scores:
            continue

        max_score = max(abs(s) for s in scores.values())
        threshold = threshold_ratio * max_score

        # Keep neurons above threshold
        for (layer, neuron), score in scores.items():
            if abs(score) >= threshold:
                key = (layer, neuron)
                neuron_counts[key] = neuron_counts.get(key, 0) + 1

    # Keep neurons shared by enough prompts
    min_count = max(1, int(sharing_ratio * total_prompts))
    knowledge_neurons = [
        kn for kn, count in neuron_counts.items() if count >= min_count
    ]

    # Sort by layer, then neuron index
    knowledge_neurons.sort()

    print(f"  Found {len(knowledge_neurons)} knowledge neurons")
    layer_dist = {}
    for layer, _ in knowledge_neurons:
        layer_dist[layer] = layer_dist.get(layer, 0) + 1
    print(f"  Layer distribution: {dict(sorted(layer_dist.items()))}")

    return knowledge_neurons


def filter_exclusive_neurons(target_neurons, other_neurons_list):
    """
    Remove knowledge neurons that are shared with other facts.
    This ensures the edit is specific to the target fact.

    Args:
        target_neurons: List of (layer, neuron) for the target fact
        other_neurons_list: List of lists of (layer, neuron) for other facts

    Returns: Filtered list of (layer, neuron) exclusive to the target
    """
    shared = set()
    for other_neurons in other_neurons_list:
        shared.update(set(target_neurons) & set(other_neurons))

    exclusive = [kn for kn in target_neurons if kn not in shared]
    print(f"  Specificity filter: {len(target_neurons)} total, "
          f"{len(shared)} shared, {len(exclusive)} exclusive")
    return exclusive


# ---------------------------------------------------------------------------
# Knowledge Surgery: Suppress, Amplify, and Edit
# ---------------------------------------------------------------------------

def suppress_knowledge_neurons(model, knowledge_neurons, mask_pos, device="cuda"):
    """
    Register hooks that zero out knowledge neuron activations at the mask position.
    Returns hook handles (call .remove() to undo).
    """
    layer_neurons = {}
    for layer, neuron in knowledge_neurons:
        layer_neurons.setdefault(layer, []).append(neuron)

    handles = []
    for layer, neurons in layer_neurons.items():
        def make_hook(neuron_list, pos):
            def hook_fn(module, input, output):
                output = output.clone()
                for n in neuron_list:
                    output[:, pos, n] = 0.0
                return output
            return hook_fn

        handle = model.bert.encoder.layer[layer].intermediate.register_forward_hook(
            make_hook(neurons, mask_pos)
        )
        handles.append(handle)

    return handles


def amplify_knowledge_neurons(model, knowledge_neurons, mask_pos, factor=2.0, device="cuda"):
    """
    Register hooks that amplify knowledge neuron activations.
    Returns hook handles.
    """
    layer_neurons = {}
    for layer, neuron in knowledge_neurons:
        layer_neurons.setdefault(layer, []).append(neuron)

    handles = []
    for layer, neurons in layer_neurons.items():
        def make_hook(neuron_list, pos, f):
            def hook_fn(module, input, output):
                output = output.clone()
                for n in neuron_list:
                    output[:, pos, n] *= f
                return output
            return hook_fn

        handle = model.bert.encoder.layer[layer].intermediate.register_forward_hook(
            make_hook(neurons, mask_pos, factor)
        )
        handles.append(handle)

    return handles


def edit_knowledge(model, tokenizer, knowledge_neurons, old_answer, new_answer,
                   lambda1=1.0, lambda2=8.0):
    """
    Edit the model's weights to change a fact from old_answer to new_answer.

    Modifies FFN output weights: removes old knowledge and injects new knowledge.
    This follows Section 5.1 of the paper.

    Returns the weight deltas so the edit can be undone.
    """
    old_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(old_answer)[0])
    new_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_answer)[0])

    old_emb = model.bert.embeddings.word_embeddings.weight[old_id].detach().clone()
    new_emb = model.bert.embeddings.word_embeddings.weight[new_id].detach().clone()

    # Normalize embeddings
    old_emb_norm = old_emb / old_emb.norm()
    new_emb_norm = new_emb / new_emb.norm()

    deltas = []
    for layer, neuron in knowledge_neurons:
        weight = model.bert.encoder.layer[layer].output.dense.weight
        original = weight[:, neuron].detach().clone()

        # Remove old knowledge, inject new
        with torch.no_grad():
            weight[:, neuron] -= lambda1 * old_emb_norm
            weight[:, neuron] += lambda2 * new_emb_norm

        deltas.append((layer, neuron, original))

    return deltas


def undo_edit(model, deltas):
    """Restore original weights after an edit."""
    for layer, neuron, original in deltas:
        with torch.no_grad():
            model.bert.encoder.layer[layer].output.dense.weight[:, neuron] = original


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_fact(model, tokenizer, prompt, expected_answer, device="cuda", label=""):
    """Evaluate a single fact and print results."""
    preds = get_prediction(model, tokenizer, prompt, device, top_k=5)
    prob = get_target_probability(model, tokenizer, prompt, expected_answer, device)

    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}Prompt: '{prompt}'")
    print(f"{prefix}  Expected: '{expected_answer}' (prob={prob:.4f})")
    print(f"{prefix}  Top-5: {[(t, f'{p:.4f}') for t, p in preds]}")
    return preds, prob


def evaluate_fact_set(model, tokenizer, facts, device="cuda", label=""):
    """Evaluate multiple facts. facts = [(prompt, expected_answer, description), ...]"""
    results = {}
    for prompt, answer, desc in facts:
        preds, prob = evaluate_fact(model, tokenizer, prompt, answer, device, label=desc)
        results[desc] = {"predictions": preds, "target_prob": prob}
    return results
