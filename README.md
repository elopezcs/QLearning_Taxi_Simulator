# CSCN8020 -- Assignment 2

## Q-Learning on Taxi-v3 (Gymnasium)

## Overview

This repository contains a tabular Q-Learning implementation for the
Taxi-v3 environment using Gymnasium.\
The project runs a baseline configuration, performs the required
hyperparameter experiments, evaluates performance, and re-runs training
using the best configuration.

## Environment

-   Env ID: Taxi-v3
-   Observation space: Discrete(500)
-   Action space: Discrete(6)

## Deliverables Included

-   Python code implementing Q-Learning and running experiments
-   Output plots and a results CSV
-   Best-configuration re-run and saved Q-table
-   REPORT.md (technical notes for the repo)
-   Separate formal report (Word/PDF) submitted for grading

## How to Run

### 1) Create and activate a virtual environment

Windows (PowerShell): python -m venv .venv
.venv`\Scripts`{=tex}`\activate`{=tex}

macOS/Linux: python3 -m venv .venv source .venv/bin/activate

### 2) Install dependencies

    pip install -r requirements.txt

### 3) Run experiments + best rerun

    python main.py --run all

Optional (faster test run): python main.py --run all --episodes 2000

## Outputs

After running, results are saved under outputs/:

-   outputs/qlearning_results.csv\
    Summary metrics for each experiment run (training averages +
    evaluation results)

-   outputs/plots/\
    Training curves (return per episode, steps per episode) for each
    configuration

-   outputs/qtable_best.npy\
    Saved Q-table from the best hyperparameter re-run

-   outputs/qtable_best_meta.json\
    Metadata for the saved best Q-table (hyperparameters used)

## Project Structure

    .
    ├── main.py
    ├── qlearning.py
    ├── utils.py
    ├── plots.py
    ├── requirements.txt
    ├── REPORT.md
    └── outputs/
        ├── qlearning_results.csv
        ├── qtable_best.npy
        ├── qtable_best_meta.json
        └── plots/

## Notes

-   The core Q-Learning update rule and interpretation of results are
    documented in the formal report and in REPORT.md.
-   Plot images in outputs/plots/ support the report discussion of
    hyperparameter effects.

## Course

CSCN8020 -- Reinforcement Learning\
Conestoga College
