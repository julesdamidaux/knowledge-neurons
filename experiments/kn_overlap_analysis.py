#!/usr/bin/env python3
"""
Knowledge Neuron Overlap Analysis
===================================

For a set of related facts (e.g., capitals of different countries),
identify the knowledge neurons of each fact independently and analyze
the pairwise overlap. This explains why editing one fact damages others.
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from knowledge_neurons import (
    load_bert,
    identify_knowledge_neurons,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IG_STEPS = 20


# ============================================================================
# Facts to analyze
# ============================================================================

CAPITAL_FACTS = {
    "France\n(Paris)": {
        "prompts": [
            "The capital of France is [MASK].",
            "France's capital city is [MASK].",
            "[MASK] is the capital of France.",
            "The capital city of France is [MASK].",
        ],
        "answer": "Paris",
    },
    "Spain\n(Madrid)": {
        "prompts": [
            "The capital of Spain is [MASK].",
            "Spain's capital city is [MASK].",
            "[MASK] is the capital of Spain.",
            "The capital city of Spain is [MASK].",
        ],
        "answer": "Madrid",
    },
    "Germany\n(Berlin)": {
        "prompts": [
            "The capital of Germany is [MASK].",
            "Germany's capital city is [MASK].",
            "[MASK] is the capital of Germany.",
            "The capital city of Germany is [MASK].",
        ],
        "answer": "Berlin",
    },
    "Japan\n(Tokyo)": {
        "prompts": [
            "The capital of Japan is [MASK].",
            "Japan's capital city is [MASK].",
            "[MASK] is the capital of Japan.",
            "The capital city of Japan is [MASK].",
        ],
        "answer": "Tokyo",
    },
    "Italy\n(Rome)": {
        "prompts": [
            "The capital of Italy is [MASK].",
            "Italy's capital city is [MASK].",
            "[MASK] is the capital of Italy.",
            "The capital city of Italy is [MASK].",
        ],
        "answer": "Rome",
    },
}

LANGUAGE_FACTS = {
    "France\n(French)": {
        "prompts": [
            "The official language of France is [MASK].",
            "People in France speak [MASK].",
            "The language spoken in France is [MASK].",
        ],
        "answer": "French",
    },
    "Germany\n(German)": {
        "prompts": [
            "The official language of Germany is [MASK].",
            "People in Germany speak [MASK].",
            "The language spoken in Germany is [MASK].",
        ],
        "answer": "German",
    },
    "Spain\n(Spanish)": {
        "prompts": [
            "The official language of Spain is [MASK].",
            "People in Spain speak [MASK].",
            "The language spoken in Spain is [MASK].",
        ],
        "answer": "Spanish",
    },
    "Italy\n(Italian)": {
        "prompts": [
            "The official language of Italy is [MASK].",
            "People in Italy speak [MASK].",
            "The language spoken in Italy is [MASK].",
        ],
        "answer": "Italian",
    },
}


# ============================================================================
# Analysis
# ============================================================================

def identify_all_kns(model, tokenizer, facts_dict):
    """Identify knowledge neurons for all facts in a dict."""
    all_kns = {}
    for name, config in facts_dict.items():
        print(f"\n{'='*50}")
        print(f"Identifying KNs for: {name.replace(chr(10), ' ')}")
        kns = identify_knowledge_neurons(
            model, tokenizer, config["prompts"], config["answer"],
            device=DEVICE, threshold_ratio=0.2, sharing_ratio=0.3,
            steps=IG_STEPS, positive_only=True
        )
        all_kns[name] = set(kns)
    return all_kns


def compute_overlap_matrix(all_kns):
    """Compute pairwise Jaccard similarity and raw intersection counts."""
    names = list(all_kns.keys())
    n = len(names)
    jaccard = np.zeros((n, n))
    intersection_counts = np.zeros((n, n), dtype=int)
    sizes = [len(all_kns[name]) for name in names]

    for i in range(n):
        for j in range(n):
            set_i = all_kns[names[i]]
            set_j = all_kns[names[j]]
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            intersection_counts[i, j] = inter
            jaccard[i, j] = inter / union if union > 0 else 0

    return names, jaccard, intersection_counts, sizes


def plot_overlap_heatmap(names, jaccard, intersection_counts, sizes, title, filename):
    """Plot a heatmap of KN overlap."""
    n = len(names)

    # Create annotation labels: "X shared\n(J=Y%)"
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = f"{sizes[i]}\ntotal"
            else:
                annot[i, j] = f"{intersection_counts[i,j]}\n({jaccard[i,j]*100:.0f}%)"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        jaccard, annot=annot, fmt="",
        xticklabels=names, yticklabels=names,
        cmap="YlOrRd", vmin=0, vmax=1,
        linewidths=0.5, linecolor='white',
        cbar_kws={"label": "Jaccard similarity"},
        ax=ax
    )
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_layer_comparison(all_kns, n_layers, title, filename):
    """Show per-fact layer distribution side by side."""
    names = list(all_kns.keys())
    n_facts = len(names)

    fig, axes = plt.subplots(1, n_facts, figsize=(3 * n_facts, 3.5), sharey=True)
    if n_facts == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", n_facts)

    for idx, (name, ax) in enumerate(zip(names, axes)):
        layer_counts = [0] * n_layers
        for l, _ in all_kns[name]:
            layer_counts[l] += 1
        ax.bar(range(n_layers), layer_counts, color=colors[idx])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Layer", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Count", fontsize=9)
        ax.set_xticks(range(0, n_layers, 2))
        ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_shared_vs_exclusive(all_kns, title, filename):
    """For each fact, show how many neurons are exclusive vs shared with others."""
    names = list(all_kns.keys())
    exclusive_counts = []
    shared_counts = []

    for i, name in enumerate(names):
        my_kns = all_kns[name]
        others = set()
        for j, other_name in enumerate(names):
            if i != j:
                others |= all_kns[other_name]
        shared = len(my_kns & others)
        exclusive = len(my_kns) - shared
        shared_counts.append(shared)
        exclusive_counts.append(exclusive)

    x = np.arange(len(names))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, exclusive_counts, width, label="Exclusive", color="#4CAF50")
    ax.bar(x, shared_counts, width, bottom=exclusive_counts, label="Shared with others", color="#FF9800")

    for i in range(len(names)):
        total = exclusive_counts[i] + shared_counts[i]
        pct = shared_counts[i] / total * 100 if total > 0 else 0
        ax.text(i, total + 0.5, f"{pct:.0f}%\nshared", ha='center', fontsize=8)

    ax.set_xlabel("Fact", fontsize=11)
    ax.set_ylabel("Number of Knowledge Neurons", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def analyze_shared_neurons(all_kns, names_list):
    """Print which specific neurons are shared across multiple facts."""
    all_neurons = set()
    for kns in all_kns.values():
        all_neurons |= kns

    neuron_facts = {}
    for neuron in all_neurons:
        facts = [name for name in names_list if neuron in all_kns[name]]
        if len(facts) > 1:
            neuron_facts[neuron] = facts

    # Sort by number of facts sharing the neuron
    sorted_neurons = sorted(neuron_facts.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"\n  Neurons shared by multiple facts: {len(sorted_neurons)}")
    print(f"  Most shared neurons:")
    for (layer, neuron), facts in sorted_neurons[:10]:
        fact_names = [f.split('\n')[0] for f in facts]
        print(f"    Layer {layer}, Neuron {neuron}: shared by {len(facts)} facts ({', '.join(fact_names)})")

    return neuron_facts


# ============================================================================
# Main
# ============================================================================

def main():
    print("Knowledge Neuron Overlap Analysis")
    print("=" * 60)

    model, tokenizer = load_bert("bert-base-cased", DEVICE)
    n_layers = model.config.num_hidden_layers

    # --- Capital facts ---
    print("\n" + "=" * 60)
    print("CAPITAL FACTS")
    print("=" * 60)
    capital_kns = identify_all_kns(model, tokenizer, CAPITAL_FACTS)

    names, jaccard, inter, sizes = compute_overlap_matrix(capital_kns)

    plot_overlap_heatmap(
        names, jaccard, inter, sizes,
        "Knowledge Neuron Overlap: Capital Facts",
        "kn_overlap_capitals.pdf"
    )
    plot_shared_vs_exclusive(
        capital_kns,
        "Exclusive vs Shared Neurons: Capital Facts",
        "kn_shared_vs_exclusive_capitals.pdf"
    )
    plot_layer_comparison(
        capital_kns, n_layers,
        "Layer Distribution per Capital Fact",
        "kn_layers_per_capital.pdf"
    )
    analyze_shared_neurons(capital_kns, names)

    # --- Language facts ---
    print("\n" + "=" * 60)
    print("LANGUAGE FACTS")
    print("=" * 60)
    language_kns = identify_all_kns(model, tokenizer, LANGUAGE_FACTS)

    names_l, jaccard_l, inter_l, sizes_l = compute_overlap_matrix(language_kns)

    plot_overlap_heatmap(
        names_l, jaccard_l, inter_l, sizes_l,
        "Knowledge Neuron Overlap: Language Facts",
        "kn_overlap_languages.pdf"
    )
    plot_shared_vs_exclusive(
        language_kns,
        "Exclusive vs Shared Neurons: Language Facts",
        "kn_shared_vs_exclusive_languages.pdf"
    )
    analyze_shared_neurons(language_kns, names_l)

    # Save results
    results = {
        "capitals": {
            "names": [n.replace('\n', ' ') for n in list(capital_kns.keys())],
            "sizes": [len(v) for v in capital_kns.values()],
            "jaccard": jaccard.tolist(),
            "intersections": inter.tolist(),
        },
        "languages": {
            "names": [n.replace('\n', ' ') for n in list(language_kns.keys())],
            "sizes": [len(v) for v in language_kns.values()],
            "jaccard": jaccard_l.tolist(),
            "intersections": inter_l.tolist(),
        },
    }
    path = os.path.join(RESULTS_DIR, "kn_overlap_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
