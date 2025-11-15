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
| `--algo`  | str  | Required | Algorithm to run (`tabq`, `dqn`, or `random`).             |
| `--dynamic` | flag | `False` | Enable dynamic lava; omit for static lava.                 |
| `--sparse` | flag | `False` | Use sparse rewards (disables reward shaping).              |
| `--episodes` | int | `5`      | Number of training episodes.                               |
| `--seed`  | int  | `20`     | Random seed for reproducibility.                           |

#### Example

```bash
python run.py --algo tabq --dynamic
```

