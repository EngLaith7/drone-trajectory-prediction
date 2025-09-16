# Model Performance Documentation

## Overview
This document describes the performance metrics and results of the AI Project - Drone Trajectory Prediction.

## Model Details
- **Algorithm**: Random Forest Regressor
- **Number of Estimators**: 100
- **Random State**: 42
- **Test Size**: 20%

## Performance Metrics
- **R2_score**: 0.8200
- **MSE**: 42.8765

## Key Findings
1. Random Forest Regressor performed beter than linear regressor
2. Applying σ-rule (hyperparameterized) helps in filtering data for the model
3. Model suffer limitations becuase the using of data of single dorne flight