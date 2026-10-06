# Credal Machine Learning for Risk-Averse Decision Making

## Setup

```bash
uv sync
uv run pre-commit install   # only needed to commit: installs the lint and type-check hooks
```

Python 3.13. All commands below run from the repository root. Datasets download on first use
into `~/datasets` (override with `DATA_PATH`).

Method names in the paper and their config names (`configs/method/`):

| Paper   | Config                        | Paper   | Config                        |
|---------|-------------------------------|---------|-------------------------------|
| MLE     | `base`                        | CreRL   | `credal_relative_likelihood`  |
| CreEns  | `credal_ensembling`           | EffCre  | `efficient_credal_prediction` |
| CreBNN  | `credal_bnn`                  | CreWare | `credal_rl_multinomial`       |
| CreWra  | `credal_wrapper`              | SQwash  | `sqwash`                      |
| CreDro  | `credal_dro`                  | AdaCVaR | `adacvar`                     |

Recipes (`configs/recipe/`): `cifar10_resnet18`, `bloodmnist_resnet18`, `pathmnist_resnet18`.

## How the real-data experiments are organised

1. **Train** a model. It is stored as a Weights & Biases artifact.
2. **Evaluate** it. Each evaluation script loads the artifact and appends its results to the
   summary of the training run, so W&B is the results store.
3. **Plot.** Refresh the local cache of all run summaries, then run the plotting script.

Set your W&B account with `wandb.entity=<entity> wandb.project=<project>` on every train and
evaluation call (or edit the defaults in `configs/*.yaml`), and refresh the cache with

```bash
(cd src && uv run python -m plotting.wandb_cache --entity <entity> --project <project>)
```

Training without W&B is possible with `wandb.enabled=false save_to_disk=true artifact_source=local`,
which writes checkpoints to `checkpoints/`.

## Training

`base` goes first: EffCre and CreWare are fitted on top of the base model of the same recipe and seed.

```bash
R=cifar10_resnet18          # or bloodmnist_resnet18, pathmnist_resnet18
uv run python src/training/train.py --multirun recipe=$R seed=1,2,3 method=base
uv run python src/training/train.py --multirun recipe=$R seed=1,2,3 method=credal_wrapper,credal_ensembling,credal_bnn,credal_dro
uv run python src/training/train.py --multirun recipe=$R seed=1,2,3 method.train.alpha=0.95 \
    method=credal_relative_likelihood,efficient_credal_prediction,credal_rl_multinomial
# risk-averse baselines, trained at the CVaR level tau they are evaluated at
uv run python src/training/train.py --multirun recipe=$R seed=1,2,3 method=sqwash,adacvar \
    method.train.alpha=0.01,0.025,0.05,0.1,0.2
```

For CreRL, EffCre and CreWare, `method.train.alpha` is the relative-likelihood level and part of
the artifact name, so every evaluation call has to repeat it.

## Experiments

### Credal sets under distribution shift (set size and accuracy; BloodMNIST, CIFAR-10)

```bash
uv run python src/experiments/shift_set_size.py --multirun recipe=$R seed=1,2,3 \
    method=credal_wrapper,credal_ensembling,credal_bnn,credal_dro
uv run python src/experiments/shift_set_size.py --multirun recipe=$R seed=1,2,3 method.train.alpha=0.95 \
    method=credal_relative_likelihood,efficient_credal_prediction,credal_rl_multinomial
uv run python src/plotting/shift_set_size.py        # plots/shift_set_size_<dataset>.pdf
```

### Out-of-distribution detection (CIFAR-10, BloodMNIST)

OOD sets: `cifar100,tin,mnist,svhn,textures,places365` for CIFAR-10 and
`pathmnist,tissuemnist,dermamnist,retinamnist,octmnist,breastmnist` for BloodMNIST.

```bash
uv run python src/experiments/ood_detection.py --multirun recipe=cifar10_resnet18 seed=1,2,3 \
    method=credal_wrapper ood_dataset=cifar100,tin,mnist,svhn,textures,places365
# set-size score instead of epistemic entropy
uv run python src/experiments/ood_detection.py --multirun recipe=cifar10_resnet18 seed=1,2,3 \
    method=credal_wrapper ood_dataset=cifar100,tin component=size decomposition=set
uv run python src/plotting/ood_table.py             # LaTeX rows; edit the constants in __main__
```

Repeat per method (with `method.train.alpha=0.95` where it applies).
`src/experiments/recompute_ood_sklearn.py` rebuilds the same table from the logged per-instance
scores with scikit-learn's AUROC.

### Risk-aversion under distribution shift (PathMNIST, BloodMNIST)

```bash
EVAL=src/experiments/shift_risk_metric.py
BETAS='cvar_beta=[0.01,0.025,0.05,0.1,0.2]'
AGGS='aggregations=[mean,cvar_0.01,cvar_0.025,cvar_0.05,cvar_0.1,cvar_0.2]'
R=pathmnist_resnet18
uv run python $EVAL --multirun seed=1,2,3 recipe=$R method=base decision_rule=mle "$AGGS"
uv run python $EVAL --multirun seed=1,2,3 recipe=$R method=base decision_rule=cvar_minimax "$BETAS" "$AGGS"
uv run python $EVAL --multirun seed=1,2,3 recipe=$R method=credal_wrapper decision_rule=cvar_minimax "$BETAS" "$AGGS"
uv run python $EVAL --multirun seed=1,2,3 recipe=$R method=credal_rl_multinomial method.train.alpha=0.95 \
    decision_rule=cvar_minimax "$BETAS" "$AGGS"
uv run python $EVAL --multirun seed=1,2,3 recipe=$R method=sqwash,adacvar \
    method.train.alpha=0.01,0.025,0.05,0.1,0.2 decision_rule=mle "$AGGS"
```

Add `loss=brier` for the Brier results. For the alpha ablation, train and evaluate CreWare at
`method.train.alpha=0.0,0.2,0.4,0.6,0.8,0.9,0.95,1.0`.

```bash
uv run python src/plotting/shift_risk_grid.py                                       # all tau, both datasets
uv run python src/plotting/shift_risk_grid.py --dataset pathmnist --taus 0.01 0.05 0.2   # main-text figure
uv run python src/plotting/shift_risk_grid.py --loss brier
uv run python src/plotting/shift_risk_grid.py --alpha-ablation
```

### Coverage versus efficiency (CIFAR-10-H)

```bash
uv run python src/experiments/coverage_efficiency.py --multirun recipe=cifar10_resnet18 seed=1,2,3 \
    method=credal_wrapper,credal_ensembling,credal_bnn,credal_dro
uv run python src/experiments/coverage_efficiency.py --multirun recipe=cifar10_resnet18 seed=1,2,3 \
    method=credal_relative_likelihood,efficient_credal_prediction,credal_rl_multinomial \
    method.train.alpha=0.0,0.2,0.4,0.6,0.8,0.9,0.95,1.0
uv run python src/plotting/coverage_efficiency_csv.py --dataset cifar10    # results/coverage_efficiency_cifar10*.csv
uv run python src/plotting/coverage_efficiency_pareto.py                   # plots/coverage_efficiency_pareto_cifar10.pdf
```

### Computation time and landmark ablation (CIFAR-10)

```bash
# trains the methods from scratch, no W&B needed; one csv per method group and seed
# (CreEns and CreWra share one trained ensemble, so they go in the same call)
uv run python src/experiments/timing.py --runs 1 --seed 1 --methods CreWare --out timing/creware_seed1.csv
uv run python src/experiments/timing.py --runs 1 --seed 1 --methods CreEns CreWra --out timing/creens_crewra_seed1.csv
uv run python src/experiments/plot_runtime_bars.py --broken-y     # reads results/timing/, plots/runtime_bars_broken.pdf

uv run python src/experiments/landmark_ablation.py --seed 1       # results/landmark_ablation_cifar10_resnet18_seed1.csv
uv run python src/plotting/landmark_ablation.py                   # plots/landmark_ablation_cifar10.pdf
```

### Cost-sensitive decision making (synthetic triage)

Self-contained, no W&B.

```bash
uv run python src/cost_sensitive/cost_sensitive_triage.py              # full run, writes results/cost_sensitive_triage.pkl
uv run python src/cost_sensitive/cost_sensitive_triage.py --plot-only  # figures from the saved results
uv run python src/cost_sensitive/selftest.py                           # numerical self-tests
```

### Risk-averse reinforcement learning (offline driving)

Self-contained, no W&B. Settings live in `configs/credal_driving.yaml`.

```bash
uv run python src/experiments/credal_driving_curve.py --setting default       # main-text map
uv run python src/experiments/credal_driving_curve.py --setting twin_peaks    # appendix map
uv run python src/experiments/credal_driving_curve.py --setting default --plot-only --layout onecol
```

## Committed results

`plots/` holds the figures of the paper. The numbers behind the figures that do not read from W&B
are committed as well: the triage results, timing, the landmark ablation and the
coverage-efficiency tables in `results/`, and the driving results in `plots/credal_driving_*.pkl`.
`--plot-only` and the plotting scripts above redraw those figures from them.
