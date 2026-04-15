#!/usr/bin/env python3
"""
Exclusive-Only Edit Experiment
===============================

Compare three editing strategies for the capital swap:
  1. Naive: edit ALL knowledge neurons (shared + exclusive)
  2. Refined-3: edit exclusive neurons (filtered against 3 countries)
  3. Refined-5: edit exclusive neurons (filtered against ALL 5 countries)

This tests whether editing truly exclusive neurons avoids collateral damage.
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
    get_prediction,
    get_target_probability,
    identify_knowledge_neurons,
    filter_exclusive_neurons,
    edit_knowledge,
    undo_edit,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
IG_STEPS = 20

# All capital facts
CAPITALS = {
    "France": {
        "prompts": [
            "The capital of France is [MASK].",
            "France's capital city is [MASK].",
            "[MASK] is the capital of France.",
            "The capital city of France is [MASK].",
        ],
        "answer": "Paris",
    },
    "Spain": {
        "prompts": [
            "The capital of Spain is [MASK].",
            "Spain's capital city is [MASK].",
            "[MASK] is the capital of Spain.",
            "The capital city of Spain is [MASK].",
        ],
        "answer": "Madrid",
    },
    "Germany": {
        "prompts": [
            "The capital of Germany is [MASK].",
            "Germany's capital city is [MASK].",
            "[MASK] is the capital of Germany.",
            "The capital city of Germany is [MASK].",
        ],
        "answer": "Berlin",
    },
    "Japan": {
        "prompts": [
            "The capital of Japan is [MASK].",
            "Japan's capital city is [MASK].",
            "[MASK] is the capital of Japan.",
            "The capital city of Japan is [MASK].",
        ],
        "answer": "Tokyo",
    },
    "Italy": {
        "prompts": [
            "The capital of Italy is [MASK].",
            "Italy's capital city is [MASK].",
            "[MASK] is the capital of Italy.",
            "The capital city of Italy is [MASK].",
        ],
        "answer": "Rome",
    },
}

# Facts to evaluate after editing
EVAL_FACTS = [
    ("The capital of France is [MASK].", "Tokyo", "France→Tokyo"),
    ("France's capital city is [MASK].", "Tokyo", "France (alt)"),
    ("The capital of Spain is [MASK].", "Madrid", "Spain→Madrid"),
    ("The capital of Germany is [MASK].", "Berlin", "Germany→Berlin"),
    ("The capital of Japan is [MASK].", "Tokyo", "Japan→Tokyo"),
    ("The capital of Italy is [MASK].", "Rome", "Italy→Rome"),
    ("The capital of China is [MASK].", "Beijing", "China→Beijing"),
]


def run_edit_and_measure(model, tokenizer, neurons, old_ans, new_ans, lam1=1.0, lam2=5.0):
    """Edit, measure all facts, undo."""
    deltas = edit_knowledge(model, tokenizer, neurons, old_ans, new_ans, lam1, lam2)
    results = {}
    for prompt, answer, desc in EVAL_FACTS:
        prob = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
        preds = get_prediction(model, tokenizer, prompt, DEVICE, top_k=1)
        results[desc] = {"prob": prob, "top1": preds[0][0]}
    undo_edit(model, deltas)
    return results


def main():
    print("Exclusive-Only Edit Experiment")
    print("=" * 60)

    model, tokenizer = load_bert("bert-base-cased", DEVICE)

    # Step 1: Identify KNs for ALL 5 capital facts
    print("\n--- Identifying KNs for all capitals ---")
    all_kns = {}
    for country, config in CAPITALS.items():
        print(f"\n  {country}:")
        kns = identify_knowledge_neurons(
            model, tokenizer, config["prompts"], config["answer"],
            device=DEVICE, threshold_ratio=0.2, sharing_ratio=0.3,
            steps=IG_STEPS, positive_only=True
        )
        all_kns[country] = kns

    france_kns = all_kns["France"]
    print(f"\n  France total KNs: {len(france_kns)}")

    # Step 2: Compute exclusive neurons with different filter levels
    # Refined-3: filter against Spain, Germany, Japan (like original experiment)
    others_3 = [all_kns["Spain"], all_kns["Germany"], all_kns["Japan"]]
    exclusive_3 = filter_exclusive_neurons(france_kns, others_3)
    print(f"  Exclusive (vs 3 countries): {len(exclusive_3)}")

    # Refined-5: filter against ALL 4 other countries
    others_5 = [all_kns[c] for c in ["Spain", "Germany", "Japan", "Italy"]]
    exclusive_5 = filter_exclusive_neurons(france_kns, others_5)
    print(f"  Exclusive (vs 5 countries): {len(exclusive_5)}")

    # Shared-only: neurons that are NOT exclusive
    shared_only = [kn for kn in france_kns if kn not in set(exclusive_5)]
    print(f"  Shared neurons: {len(shared_only)}")

    # Step 3: Baseline
    print("\n--- Baseline ---")
    baseline = {}
    for prompt, answer, desc in EVAL_FACTS:
        prob = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
        baseline[desc] = prob
        print(f"  {desc}: P={prob:.4f}")

    # Step 4: Test three editing strategies
    strategies = {
        f"All KN ({len(france_kns)})": france_kns,
        f"Excl. vs 3 ({len(exclusive_3)})": exclusive_3,
        f"Excl. vs 5 ({len(exclusive_5)})": exclusive_5,
    }

    all_results = {}
    for name, neurons in strategies.items():
        print(f"\n--- Strategy: {name} ---")
        if len(neurons) == 0:
            print("  No neurons to edit, skipping")
            continue
        results = run_edit_and_measure(model, tokenizer, neurons, "Paris", "Tokyo")
        all_results[name] = results
        for desc, r in results.items():
            status = ""
            if "France" in desc:
                status = f"  (edit {'OK' if r['prob'] > 0.1 else 'weak'})"
            else:
                damage = abs(r["prob"] - baseline[desc]) / max(baseline[desc], 1e-8) * 100
                status = f"  (damage: {damage:.0f}%)" if damage > 10 else "  (OK)"
            print(f"  {desc}: P={r['prob']:.4f}  top1={r['top1']}{status}")

    # Step 5: Plot comparison
    fact_labels = [desc for _, _, desc in EVAL_FACTS]
    n_strategies = len(all_results)
    x = np.arange(len(fact_labels))
    width = 0.8 / (n_strategies + 1)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Baseline bars
    baseline_vals = [baseline[f] for f in fact_labels]
    ax.bar(x - width * n_strategies / 2, baseline_vals, width,
           label="Before edit", color="#2196F3")

    colors = ["#F44336", "#FF9800", "#4CAF50"]
    for i, (name, results) in enumerate(all_results.items()):
        vals = [results[f]["prob"] for f in fact_labels]
        ax.bar(x - width * n_strategies / 2 + width * (i + 1), vals, width,
               label=name, color=colors[i])

    ax.set_xlabel("Fact", fontsize=11)
    ax.set_ylabel("P(target)", fontsize=11)
    ax.set_title("Capital Swap: Effect of Neuron Exclusivity on Collateral Damage", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(fact_labels, rotation=25, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exclusive_edit_comparison.pdf"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: exclusive_edit_comparison.pdf")

    # Save results
    save_data = {
        "france_kns_total": len(france_kns),
        "exclusive_vs_3": len(exclusive_3),
        "exclusive_vs_5": len(exclusive_5),
        "shared": len(shared_only),
        "baseline": baseline,
        "strategies": {name: {k: v["prob"] for k, v in res.items()}
                       for name, res in all_results.items()},
    }
    path = os.path.join(RESULTS_DIR, "exclusive_edit_results.json")
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved: {path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    target_facts = ["France→Tokyo", "France (alt)"]
    control_facts = ["Spain→Madrid", "Germany→Berlin", "Japan→Tokyo", "Italy→Rome", "China→Beijing"]

    print(f"\n{'Strategy':<25} {'Edit success':>15} {'Avg ctrl damage':>18}")
    print("-" * 60)
    for name, results in all_results.items():
        avg_target = np.mean([results[f]["prob"] for f in target_facts])
        avg_ctrl = np.mean([results[f]["prob"] for f in control_facts])
        avg_ctrl_base = np.mean([baseline[f] for f in control_facts])
        ctrl_preserved = avg_ctrl / max(avg_ctrl_base, 1e-8) * 100
        print(f"{name:<25} {avg_target:>15.4f} {ctrl_preserved:>17.1f}% preserved")


if __name__ == "__main__":
    main()
