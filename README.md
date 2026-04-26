# thesis-continual-learning-fraud-detection

Master thesis by Juliette Loossens & Lucas Hacha

KU Leuven, 2025–2026. Promoter: Prof. Dr. Wouter Verbeke. Supervisor: Bruno Deprez.

This repo benchmarks six continual learning methods against naïve fine-tuning for financial fraud detection under concept drift. We use a synthetic transaction generator to create 24 datasets across different drift scenarios (sudden, gradual, freeze, pattern rotation) and evaluate each method on PR-AUC, forgetting, and computational cost.

Fraud Dataset Generator.py: generates synthetic card transaction data with configurable concept drift.

Ideal Setup Datasets VS Methods.ipynb: runs all CL methods on the datasets and produces the evaluation results.
