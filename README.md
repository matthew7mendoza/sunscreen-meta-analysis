# Automating Evidence-Based Medicine: A Computational Meta-Analysis of Sunscreen Photoprotection

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-conda--forge-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview
Sunscreen application falls short of the 2.0 mg/cm^2 clinical threshold required for effective photoprotection. This repository is a fully custom OOP Python program that conducts a meta-analysis of in_vivo trails evaluating the consecutive double application of sunscreen. This meta-analysis was conducted following the guidelines from Cochrane Handbook for Systematic Reviews of Interventions.


## Components of the Project
This project is structured modularly to allow anyone to append new clinical trail data and automatically recalculate global effect size:

* `methodologies.ipynb`: This provides an indepth explanation of the meta-analysis process.

* `data/extracted_papers.csv`: Standardized input of raw clinical trial data, extracted manually from cited papers. 

* `utilities.py`: Handles data processing, cleaning, and dynamic calculation of coefficient of variation (CV) substitutions.
* `formulas.py`: All mathematical formulas grouped as classes.
* `analyzer.py`: Pools effect sizes using Restricted Maximum Likelihood (REML) and random-effects modeling, applies Knapp-Hartung variance adjustment for small sample sizes.
* `visuals.py`: Utilizes `matplotlib` to generate Risk of Bias chart, PRISMA flowchart, and forest plots. 

* `main.py`: Runs the complete program and executies sensitivity analyses.

## Installation & Setup
This project uses `conda` (Miniforge)

**Clone the repository**
```bash
git clone [https://github.com/matthew7mendoza/UV-Me-meta-analysis.git](https://github.com/matthew7mendoza/UV-Me-meta-analysis.git)
cd UV-Me-meta-analysis
