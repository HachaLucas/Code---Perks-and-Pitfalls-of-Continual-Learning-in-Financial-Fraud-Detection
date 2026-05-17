# Continual Learning for Fraud Detection

Master-thesis experiments comparing continual-learning (CL) methods on
synthetic credit-card fraud datasets that exhibit different kinds of
concept drift. The study trains a fixed 3-layer MLP sequentially on ten
"tasks" (time periods) per dataset and measures how well each method
trades off **plasticity** (learning the new period) against **stability**
(remembering the past).

Methods compared:

- **Naive** — sequential fine-tuning, no consolidation.
- **Replay** / **Full Replay** — experience replay with bounded vs unbounded buffer.
- **EWC** — Elastic Weight Consolidation.
- **PackNet** — parameter isolation via iterative pruning.
- **SI** — Synaptic Intelligence.
- **DER++** — Dark Experience Replay with hard-label anchor.

Datasets are organised along six "axes" of drift (baseline, sudden,
incremental, freeze, offset, pattern rotation).

## Project layout

```
.
├── assets/                 Static assets (e.g. images for the README).
├── config/
│   └── experiment.yaml     All hyperparameters and runtime switches.
├── data/
│   ├── raw/                Raw CSV inputs (gitignored).
│   └── processed/          Reserved for processed artefacts.
├── lib/                    Off-the-shelf code (empty by default).
├── notebooks/
│   └── demo.ipynb          Slim demo: load config + run one method.
├── res/
│   ├── experiment_results/ .npy checkpoints per (dataset, method).
│   └── figures/            Generated figures.
├── scripts/
│   └── run_experiments.py  CLI runner — multi-dataset, multi-method.
├── src/
│   ├── data/
│   │   └── dataset.py      DataLoader class (path / scaler / cache).
│   ├── methods/
│   │   ├── model.py        FraudDetector (3-layer MLP).
│   │   ├── naive.py
│   │   ├── replay.py
│   │   ├── ewc.py
│   │   ├── packnet.py
│   │   ├── si.py
│   │   └── der.py
│   └── utils/
│       ├── metrics.py      compute_thesis_metrics, test_on_task.
│       ├── visualization.py
│       ├── checkpoints.py  npy save/load helpers (used by runner + demo).
│       └── seeding.py      Reproducibility helper.
├── .gitignore
├── LICENSE                 
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Drop the dataset CSVs in a folder matching `paths.base_dir` from the
config (default: `data/` at the project root). Or update
`paths.base_dir` in the YAML to point to your dataset folder.

## How to run

### From the command line

```bash
python -m scripts.run_experiments --config config/experiment.yaml
```

You can override the axes from the CLI:

```bash
python -m scripts.run_experiments --axes 0 1
```

Per-(dataset, method) results are checkpointed to
`res/experiment_results/<dataset>/<method>_acc.npy` and `_pr.npy`.
Re-running only re-trains the missing combinations; delete a folder to
force a re-run.

### From a notebook

`notebooks/demo.ipynb` is the minimal sanity-check: it loads the config,
runs a single method on a single dataset, and prints the aggregated
metrics. Use it to confirm the migration is working, then scale up via
the CLI runner.

## Configuration

All hyperparameters live in `config/experiment.yaml`. Notable keys:

- `n_features`, `n_tasks`, `n_epochs`, `learning_rate`
- `memory_size` (replay), `ewc_lambda`, `fisher_samples`,
  `packnet_prune_ratio`, `si_lambda`, `si_xi`, `der_alpha`, `der_beta`
- `run_methods` — toggle individual methods.
- `run_axes` — list of axis IDs to run, or `null` for all 44 datasets.
- `show_metrics` — which metrics to print after each run.
- `paths.base_dir` — folder containing the dataset CSVs.
- `paths.results_dir`, `paths.figures_dir` — outputs.

## License

MIT — see [LICENSE](LICENSE).
