# Knowledge Neurons in Pretrained Transformers

Code for the MVA/ENSTA XAI course project. Based on the paper:

> **Knowledge Neurons in Pretrained Transformers**
> Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, Furu Wei.
> ACL 2022. [[paper]](https://arxiv.org/abs/2104.08696) [[original code]](https://github.com/Hunter-DDM/knowledge-neurons)

We reimplement the knowledge neuron pipeline and run new experiments on BERT-base-cased and GPT-2, including fact editing ("The capital of France is Tokyo"), a neuron overlap analysis, and a BERT vs GPT-2 comparison.

## Setup

```bash
pip install -r requirements.txt
```

A GPU is recommended. All experiments were run on a single NVIDIA RTX A5000.

## Experiments

All scripts are in `experiments/` and write results (figures + JSON) to `results/`.

```bash
cd experiments

# 1. Main BERT experiments (capital swap, language confusion, Einstein teleportation)
python run_experiments.py

# 2. Lambda sweep (edit strength vs collateral damage trade-off)
python lambda_sweep.py

# 3. Knowledge neuron overlap analysis (shared vs exclusive neurons)
python kn_overlap_analysis.py

# 4. GPT-2 experiments (currency swap, capital confusion) + BERT vs GPT-2 comparison
python run_gpt2_experiments.py
```

Each script takes a few minutes on a GPU. Models (BERT-base-cased, GPT-2) are downloaded automatically from HuggingFace on first run.

## Project structure

```
experiments/
  knowledge_neurons.py     # Core library (IG attribution, suppression, editing)
  run_experiments.py        # BERT experiments
  run_gpt2_experiments.py   # GPT-2 experiments
  lambda_sweep.py           # Lambda sweep analysis
  kn_overlap_analysis.py    # Neuron overlap between related facts
results/                    # Generated figures (PDF) and result files (JSON)
original_repo/              # Clone of the original paper's code
report.tex                  # Project report (LaTeX)
report.pdf                  # Compiled report
```

## Report

The report (`report.tex`) can be compiled with:

```bash
pdflatex report.tex && pdflatex report.tex
```
