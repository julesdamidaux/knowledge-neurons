#!/usr/bin/env python3
"""
Lambda Sweep: Analyze the trade-off between edit strength and collateral damage.

We sweep lambda2 (injection strength) while keeping lambda1=1 fixed,
and measure:
  - Edit success: P(new_answer | target_prompt) after edit
  - Collateral damage: average probability drop on control facts
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
    get_target_probability,
    identify_knowledge_neurons,
    filter_exclusive_neurons,
    edit_knowledge,
    undo_edit,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def lambda_sweep(model, tokenizer, knowledge_neurons, original_answer, new_answer,
                 target_prompts, control_facts, lambda2_values, exp_name):
    """Sweep lambda2 and measure edit success vs collateral damage."""

    results = {"lambda2": [], "edit_success": [], "collateral_damage": [],
               "target_probs": [], "control_probs": []}

    for lam2 in lambda2_values:
        # Apply edit
        deltas = edit_knowledge(
            model, tokenizer, knowledge_neurons,
            original_answer, new_answer,
            lambda1=1.0, lambda2=lam2
        )

        # Measure edit success (average P(new_answer) on target prompts)
        target_probs = []
        for prompt, answer, _ in target_prompts:
            p = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
            target_probs.append(p)
        avg_target = np.mean(target_probs)

        # Measure collateral damage (average prob drop on control facts)
        control_probs = []
        for prompt, answer, _ in control_facts:
            p = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
            control_probs.append(p)
        avg_control = np.mean(control_probs)

        results["lambda2"].append(lam2)
        results["edit_success"].append(avg_target)
        results["collateral_damage"].append(avg_control)
        results["target_probs"].append(target_probs)
        results["control_probs"].append(control_probs)

        print(f"  lambda2={lam2:.1f}: edit_success={avg_target:.4f}, "
              f"avg_control_prob={avg_control:.4f}")

        # Undo edit
        undo_edit(model, deltas)

    return results


def plot_lambda_sweep(results_all, results_exclusive, exp_name, title,
                      baseline_control_avg):
    """Plot lambda sweep results comparing naive vs exclusive."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Edit success
    ax1.plot(results_all["lambda2"], results_all["edit_success"],
             'o-', color="#FF5722", label="All KN", linewidth=2, markersize=6)
    if results_exclusive:
        ax1.plot(results_exclusive["lambda2"], results_exclusive["edit_success"],
                 's-', color="#2196F3", label="Exclusive KN", linewidth=2, markersize=6)
    ax1.set_xlabel("Lambda2 (injection strength)", fontsize=12)
    ax1.set_ylabel("P(new answer) on target prompts", fontsize=12)
    ax1.set_title("Edit Success vs Lambda2", fontsize=13)
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Right: Collateral damage
    ax2.axhline(y=baseline_control_avg, color='gray', linestyle='--',
                label=f"Baseline ({baseline_control_avg:.2f})", alpha=0.7)
    ax2.plot(results_all["lambda2"], results_all["collateral_damage"],
             'o-', color="#FF5722", label="All KN", linewidth=2, markersize=6)
    if results_exclusive:
        ax2.plot(results_exclusive["lambda2"], results_exclusive["collateral_damage"],
                 's-', color="#2196F3", label="Exclusive KN", linewidth=2, markersize=6)
    ax2.set_xlabel("Lambda2 (injection strength)", fontsize=12)
    ax2.set_ylabel("Avg P(correct) on control facts", fontsize=12)
    ax2.set_title("Control Fact Preservation vs Lambda2", fontsize=13)
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{exp_name}_lambda_sweep.pdf"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {exp_name}_lambda_sweep.pdf")


def plot_combined_tradeoff(all_results):
    """Plot combined trade-off curve for all experiments."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#FF5722", "#2196F3", "#4CAF50"]
    markers = ["o", "s", "D"]

    for i, (name, data) in enumerate(all_results.items()):
        for variant, results in data.items():
            style = '-' if 'all' in variant else '--'
            label = f"{name} ({variant})"
            ax.plot(results["collateral_damage"], results["edit_success"],
                    f'{markers[i]}{style}', color=colors[i], label=label,
                    linewidth=2, markersize=6, alpha=0.8)

    ax.set_xlabel("Avg P(correct) on control facts (higher = less damage)", fontsize=12)
    ax.set_ylabel("P(new answer) on target (higher = better edit)", fontsize=12)
    ax.set_title("Edit Effectiveness vs Collateral Damage Trade-off", fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Ideal point annotation
    ax.annotate("Ideal\n(top-right)", xy=(1.0, 1.0), xytext=(0.7, 0.6),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'),
                color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "combined_tradeoff.pdf"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: combined_tradeoff.pdf")


def main():
    print("Lambda Sweep Analysis")
    print("=" * 70)

    model, tokenizer = load_bert("bert-base-cased", DEVICE)
    lambda2_values = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0]

    # Load knowledge neurons from previous run
    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path) as f:
        prev_results = json.load(f)

    experiments = {
        "capital_swap": {
            "original_answer": "Paris",
            "new_answer": "Tokyo",
            "target_prompts": [
                ("The capital of France is [MASK].", "Tokyo", "France->Tokyo"),
                ("France's capital city is [MASK].", "Tokyo", "France alt"),
            ],
            "control_facts": [
                ("The capital of Japan is [MASK].", "Tokyo", "Japan"),
                ("The capital of Spain is [MASK].", "Madrid", "Spain"),
                ("The capital of Germany is [MASK].", "Berlin", "Germany"),
                ("The capital of Italy is [MASK].", "Rome", "Italy"),
                ("The capital of China is [MASK].", "Beijing", "China"),
            ],
            "related_prompts": {
                "Japan": (["The capital of Japan is [MASK].", "Japan's capital city is [MASK]."], "Tokyo"),
                "Spain": (["The capital of Spain is [MASK].", "Spain's capital city is [MASK]."], "Madrid"),
                "Germany": (["The capital of Germany is [MASK].", "Germany's capital city is [MASK]."], "Berlin"),
            },
        },
        "language_confusion": {
            "original_answer": "French",
            "new_answer": "German",
            "target_prompts": [
                ("The official language of France is [MASK].", "German", "France->German"),
                ("People in France speak [MASK].", "German", "France speaks"),
            ],
            "control_facts": [
                ("The official language of Germany is [MASK].", "German", "Germany"),
                ("The official language of Spain is [MASK].", "Spanish", "Spain"),
                ("The official language of Italy is [MASK].", "Italian", "Italy"),
                ("The official language of Japan is [MASK].", "Japanese", "Japan"),
            ],
            "related_prompts": {
                "Germany": (["The official language of Germany is [MASK].", "People in Germany speak [MASK]."], "German"),
                "Spain": (["The official language of Spain is [MASK].", "People in Spain speak [MASK]."], "Spanish"),
            },
        },
        "einstein_teleportation": {
            "original_answer": "Germany",
            "new_answer": "Paris",
            "target_prompts": [
                ("Albert Einstein was born in [MASK].", "Paris", "Einstein->Paris"),
                ("The birthplace of Albert Einstein is [MASK].", "Paris", "Einstein alt"),
            ],
            "control_facts": [
                ("Isaac Newton was born in [MASK].", "England", "Newton"),
                ("Charles Darwin was born in [MASK].", "England", "Darwin"),
                ("Marie Curie was born in [MASK].", "Paris", "Curie"),
                ("The capital of France is [MASK].", "Paris", "France->Paris"),
            ],
            "related_prompts": {
                "Newton": (["Isaac Newton was born in [MASK].", "The birthplace of Isaac Newton is [MASK]."], "England"),
                "Darwin": (["Charles Darwin was born in [MASK].", "The birthplace of Charles Darwin is [MASK]."], "England"),
            },
        },
    }

    all_sweep_results = {}

    for exp_name, exp_config in experiments.items():
        print(f"\n{'='*50}")
        print(f"Lambda sweep for: {exp_name}")
        print(f"{'='*50}")

        # Get knowledge neurons from previous results
        kn_list = prev_results[exp_name]["knowledge_neurons"]
        knowledge_neurons = [tuple(kn) for kn in kn_list]
        print(f"  Using {len(knowledge_neurons)} knowledge neurons from previous run")

        # Compute baseline control average
        baseline_ctrl = []
        for prompt, answer, _ in exp_config["control_facts"]:
            p = get_target_probability(model, tokenizer, prompt, answer, DEVICE)
            baseline_ctrl.append(p)
        baseline_avg = np.mean(baseline_ctrl)
        print(f"  Baseline control average: {baseline_avg:.4f}")

        # Sweep with all neurons
        print("\n  Sweep with ALL knowledge neurons:")
        results_all = lambda_sweep(
            model, tokenizer, knowledge_neurons,
            exp_config["original_answer"], exp_config["new_answer"],
            exp_config["target_prompts"], exp_config["control_facts"],
            lambda2_values, exp_name
        )

        # Compute exclusive neurons
        print("\n  Computing exclusive neurons...")
        other_neurons_list = []
        for name, (rel_prompts, rel_answer) in exp_config["related_prompts"].items():
            rel_kns = identify_knowledge_neurons(
                model, tokenizer, rel_prompts, rel_answer,
                device=DEVICE, threshold_ratio=0.2, sharing_ratio=0.3,
                steps=20, positive_only=True
            )
            other_neurons_list.append(rel_kns)
        exclusive_neurons = filter_exclusive_neurons(knowledge_neurons, other_neurons_list)

        results_exclusive = None
        if len(exclusive_neurons) > 0:
            print(f"\n  Sweep with {len(exclusive_neurons)} EXCLUSIVE knowledge neurons:")
            results_exclusive = lambda_sweep(
                model, tokenizer, exclusive_neurons,
                exp_config["original_answer"], exp_config["new_answer"],
                exp_config["target_prompts"], exp_config["control_facts"],
                lambda2_values, exp_name
            )

        all_sweep_results[exp_name] = {"all_kn": results_all}
        if results_exclusive:
            all_sweep_results[exp_name]["exclusive_kn"] = results_exclusive

        plot_lambda_sweep(results_all, results_exclusive, exp_name,
                         f"Lambda Sweep: {exp_name}", baseline_avg)

    # Combined trade-off plot
    print("\nGenerating combined trade-off plot...")
    plot_combined_tradeoff(all_sweep_results)

    # Save sweep results
    sweep_path = os.path.join(RESULTS_DIR, "lambda_sweep_results.json")
    serializable = {}
    for name, data in all_sweep_results.items():
        serializable[name] = {}
        for variant, results in data.items():
            serializable[name][variant] = {
                "lambda2": results["lambda2"],
                "edit_success": results["edit_success"],
                "collateral_damage": results["collateral_damage"],
            }
    with open(sweep_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved: {sweep_path}")


if __name__ == "__main__":
    main()
