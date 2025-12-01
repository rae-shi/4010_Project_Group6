# Project Setup Guide

## Virtual Environment Setup

Create and activate a virtual environment with Python 3.10

### 0. Python Version Control
```bash
pyenv install 3.10
python -V  # check the version
pyenv local <python_version>
```

### 1. Create Virtual Environment
```bash
python -m venv <venv_name>
```

### 2. Activate Virtual Environment
```bash
source <venv_name>/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Getting Started

Run the program from your terminal using the `python run.py` command, followed by the desired arguments.

### Available Arguments

| Argument  | Type | Default  | Description                                                |
|-----------|------|----------|------------------------------------------------------------|
| `--algo`  | str  | Required | Algorithm to run (`tabq`, `dqn`, `ddqn`, or `heuristic`). |
| `--dynamic` | flag | `False` | Enable dynamic lava; omit for static lava.                 |
| `--sparse` | flag | `False` | Use sparse rewards (disables reward shaping).              |
| `--episodes` | int | `500`    | Number of training episodes.                               |
| `--seed`  | int  | `20`     | Random seed for reproducibility.                           |
| `--restart` | flag | `False` | Delete existing checkpoint/csv and start fresh.        |

## Experiments & Usage Examples

To replicate the full set of experiments (Static vs. Dynamic × Shaped vs. Sparse), run the commands below.

### Notes(`Please read this`)
* **Episodes:** Episode numbers are starting suggestions. You should run more if needed.
* **Parallel runs:** You can open multiple terminals and run different commands at the same time.
* **Checkpoints:** The script saves training progress for *dqn/ddqn*. When you run the **same command again**, it **continues from the last checkpoint** automatically.

  * **Continue training:** run the same command (no extra flags).
  * **If you want start from scratch:** add `--restart`.
    * Example: `python run.py --algo dqn --episodes 3000 --seed 20 --restart`
  * **Alternative clean option:** delete the saved checkpoint/results folder and re-run.


### 1) Tabular Q-Learning (TABQ) — fast (minutes)

```bash
# Static + Shaped
python run.py --algo tabq --episodes 200 --seed 20
# Static + Sparse
python run.py --algo tabq --episodes 200 --seed 20 --sparse
# Dynamic + Shaped
python run.py --algo tabq --episodes 200 --seed 20 --dynamic
# Dynamic + Sparse
python run.py --algo tabq --episodes 200 --seed 20 --dynamic --sparse
```

### 2) DQN — 30+ minutes to hours

```bash
# Static + Shaped
python run.py --algo dqn --episodes 3000 --seed 20
# Static + Sparse
python run.py --algo dqn --episodes 3000 --seed 20 --sparse
# Dynamic + Shaped
python run.py --algo dqn --episodes 3000 --seed 20 --dynamic
# Dynamic + Sparse
python run.py --algo dqn --episodes 3000 --seed 20 --dynamic --sparse
```

### 3) DDQN — 30+ minutes to hours

```bash
# Static + Shaped
python run.py --algo ddqn --episodes 3000 --seed 20
# Static + Sparse
python run.py --algo ddqn --episodes 3000 --seed 20 --sparse
# Dynamic + Shaped
python run.py --algo ddqn --episodes 3000 --seed 20 --dynamic
# Dynamic + Sparse
python run.py --algo ddqn --episodes 3000 --seed 20 --dynamic --sparse
```

### 4) Heuristic (Baseline) — fast (minutes)

```bash
# Static + Shaped
python run.py --algo heuristic --episodes 100 --seed 20
# Static + Sparse
python run.py --algo heuristic --episodes 100 --seed 20 --sparse
# Dynamic + Shaped
python run.py --algo heuristic --episodes 100 --seed 20 --dynamic
# Dynamic + Sparse
python run.py --algo heuristic --episodes 100 --seed 20 --dynamic --sparse
```
### 5) Plot results

```bash
python scripts/plot_results.py
```
Generates graph from saved csv files.

## Sparse vs. Shaped Rewards
**Sparse:** rewards only show up at the goal/death. Training is much slower and often unstable. TabQ/DQN/DDQN usually perform poorly here (you may need far more episodes and still get worse results than with shaped rewards). *Sparse mode is mainly an ablation to demonstrate that reward shaping matters*.

**How to compare**
* **Fair baseline:** Use the *same* episode counts for shaped vs. sparse to show how much easier shaped is.
* **Stretch test:** Optionally give sparse far more episodes to see if it can catch up (it often doesn’t).


## `REQUIRED`:
**Push your final result CSV files.**


