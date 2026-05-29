# MORL-TSP

Preference-conditioned multi-objective reinforcement learning for runtime-tunable transit signal priority (TSP) in SUMO/IntersectionZoo.

This repository contains the code and configuration files for the ITSC 2026 paper:

> **Preference-Conditioned Multi-Objective Reinforcement Learning for Runtime-Tunable Transit Signal Priority**  
> Philip-Roman Adam and Stefanie Schmidtner

The project studies single-intersection transit signal priority as a constrained phase-selection problem. Instead of training one controller for one fixed bus-priority setting, it trains a preference-conditioned MORL policy `pi(a | s, w)` that can be tuned at evaluation time with a runtime preference weight `w_bus`. The resulting policy trades off bus delay against all-vehicle and non-bus delay without retraining.

## What is included

- **TSP environment wrappers** for IntersectionZoo-derived SUMO intersections, including constrained phase requests, bus-state extraction, bus-prevalence augmentation, and timetable-based bus insertion.
- **Controllers and baselines** for fixed-time control, a rule-based green-extension / early-green TSP overlay, fixed-weight PPO specialists, and the preference-conditioned MORL controller.
- **Paper experiment configs** for training, evaluation, hyperparameter tuning, distribution-shift evaluation, and generation of the reported figures and tables.
- **Reproducibility metadata** for the SUMO version, pinned IntersectionZoo submodule revision, third-party licenses, and intentionally excluded generated artifacts.

## Setup

```bash
uv sync
```

The core Python dependencies install with `uv sync`. The ITSC experiments also require the installable `intersection_zoo` Python package and the dataset snapshot used by the paper; install that dependency separately as described below.

## Cloning With Submodules

This repository uses IntersectionZoo as a pinned Git submodule at:

```text
third_party/IntersectionZoo/
```

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/urbanAIthi/morl-tsp.git
cd morl-tsp
```

If the repository was cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

The upstream IntersectionZoo revision referenced for the paper is pinned in the submodule. Do not replace it with a floating branch if you want to preserve the recorded provenance.

All reported results use SUMO v1.25.0. The released repository includes upstream IntersectionZoo revision `912d102` as a pinned Git submodule for provenance and license preservation.

## IntersectionZoo Runtime Dependency

The experiment runner imports the package namespace `intersection_zoo` and expects the paper dataset snapshot to contain networks such as:

```text
dataset/chicago/2412/net.net.xml
dataset/los-angeles/2114/net.net.xml
dataset/new-york-city/10802/net.net.xml
```

The pinned upstream submodule is not the installable Python package used by the experiment code and does not contain all paper networks. Install the compatible `intersection_zoo` package and dataset snapshot separately, then point this repository to it:

```bash
git clone <INTERSECTION_ZOO_PACKAGE_REPO_URL> ../intersection_zoo
uv pip install -e ../intersection_zoo
export INTERSECTION_ZOO_ROOT_PATH="$(pwd)/../intersection_zoo"
```

If your local checkout is elsewhere, set:

```bash
export INTERSECTION_ZOO_ROOT_PATH=/path/to/intersection_zoo
```

That path must contain both the importable `intersection_zoo/` package and the `dataset/` directory.

## Experiment Pipeline

The paper-specific instructions live in `ITSC_2026/README.md`.

Key entry points:

- `ITSC_2026/configs/train` for training and HPT configs
- `ITSC_2026/configs/eval` for evaluation configs
- `ITSC_2026/src/plot_publication_figures.py` for figures
- `ITSC_2026/src/generate_publication_tables.py` for tables
- `scripts/slurm_experiment.py` and the `slurm_experiment` console command for local or Slurm execution

## Outputs

Generated checkpoints, evaluation rows, figures, tables, W&B runs, logs, and local Slurm profiles are intentionally not included. The expected output locations are:

```text
ITSC_2026/models/morl
ITSC_2026/models/ppo
ITSC_2026/results/evaluations
ITSC_2026/results/figures
ITSC_2026/results/tables


## Citation

If you use this code, please cite our ITSC 2026 paper:

```bibtex
@inproceedings{adam2026preference,
  author    = {Adam, Philip-Roman and Schmidtner, Stefanie},
  title     = {Preference-Conditioned Multi-Objective Reinforcement Learning for Runtime-Tunable Transit Signal Priority},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year      = {2026},
  address   = {Naples, Italy},
  note      = {Accepted}
}
```

## License

This repository is licensed under the GNU Affero General Public License v3.0 or later, except where otherwise noted.

The repository contains code copied or adapted from SUMO-RL, which is MIT-licensed. The preserved SUMO-RL license text is available in `LICENSES/MIT-SUMO-RL.txt`. The adapted code is based on SUMO-RL commit `596b7c63c6d98e2a38785146b75f40ed0f7f456e`.

IntersectionZoo is included as a pinned Git submodule for reproducibility and remains MIT-licensed. The preserved IntersectionZoo license text is available in `LICENSES/MIT-INTERSECTIONZOO.txt`.

See `THIRD_PARTY_NOTICES.md` for third-party license details.
