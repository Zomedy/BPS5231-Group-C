# BPS5231-Group-C
BPS5231 AI for Sustainable Building Design,Group Assignment

Group Members:
- Yang Baisong, A0317341M
- Zhao Dingyi, A0357194U

## Project Overview

This project investigates the thermal resilience of data centers under extreme heatwave conditions using physics-based building energy simulations and machine learning.

A reference data center was modelled using DesignBuilder and EnergyPlus. An automated Python workflow was developed to generate simulation datasets under varying heatwave and operating conditions. Machine learning surrogate models were then trained and evaluated to predict thermal risk and maximum Data Hall temperature.

The final project also includes an interactive web application for rapid scenario-based predictions.
https://data-center-heatwave-resilience.onrender.com

## Folder Descriptions

### 1. Interim submission

Contains the materials submitted for the midterm review, including the interim report and slides.

### 2. Final submission

Contains the final project deliverables:

- Final 8-page research write-up
- Final presentation slides

### 3. step1-case dataset

Contains the simulation and dataset-generation components, including:

- Python scripts for automated EnergyPlus simulations
- Baseline simulation model and input files
- Latin Hypercube Sampling (LHS) dataset generation
- Simulation input parameters and output datasets


### 4. step2-ML

Contains the subsequent data processing and machine learning workflow, including:

- Data preprocessing and exploratory data analysis (EDA)
- Thermal risk classification
- Maximum temperature regression
- Model training, evaluation and comparison
- Adaptive sampling and additional simulation datasets
- Feature importance and model interpretation
- Independent verification results

### 5. Surrogate_App

Contains the final integrated web application developed for interactive thermal resilience assessment.

The application allows users to adjust heatwave and operating parameters and obtain rapid predictions of:

- Thermal risk classification (Safe / Warning / Failure)
- Maximum Data Hall temperature (Tmax)
- Available cooling capacity

## Project Workflow

DesignBuilder → EnergyPlus → Python Automation → Simulation Dataset → Machine Learning → Independent Verification → Web Application