# Logic Depth Prediction Using Machine Learning

## Overview

This project addresses the challenge of long synthesis runtimes in digital circuit design by developing an AI-powered solution to predict combinational logic depth before synthesis. By accurately estimating logic depth early in the design flow, potential timing violations can be identified and mitigated, leading to faster design iterations and reduced overall project time. The solution utilizes an XGBoost machine learning model trained on a synthetically generated dataset of circuits, leveraging features derived from gate-level circuit descriptions. This project targets a generic technology node.

## Project Structure

```
|-- logic_depth_dataset.csv    # Dataset containing generated circuit data
|-- generate_dataset.py        # Script to generate synthetic dataset for training
|-- predicted.py               # ML model for logic depth prediction
|-- README.md                  # Project documentation (this file)
```

## Dependencies

Ensure you have the following dependencies installed:

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
```

## Setup Instructions

To set up the environment, clone the repository and install the required dependencies:

```bash
git clone <repository_link>
cd <repository_name>
pip install -r requirements.txt  # If a requirements file is provided
```

## Dataset Generation

To generate a dataset of random circuits, run:

```bash
python generate_dataset.py
```

This will create `logic_depth_dataset.csv` containing circuit information and their corresponding logic depth values.

## Execution Steps

To train the model and predict logic depth:

```bash
python predicted.py
```

This script:

- Loads and preprocesses the dataset.
- Trains an XGBoost regression model to predict logic depth.
- Evaluates model performance using various metrics.
- Provides predictions on new circuit designs.

## Approach Explanation

The model uses:

- **Feature Engineering**: One-hot encoding of gate types, numerical scaling.
- **Machine Learning Model**: XGBoost regressor.
- **Evaluation Metrics**: RMSE, MAE, R² Score, Explained Variance, Maximum Error.
- **Cross-Validation**: 5-fold cross-validation for robustness.

## Proof of Correctness

The model’s performance is validated using:

- Scatter plots for actual vs. predicted logic depth.
- Residual plots for error distribution.
- Feature importance analysis using bar charts.

## Complexity Analysis

- **Time Complexity**: Training the model runs in `O(n log n)` due to XGBoost’s tree-based learning.
- **Space Complexity**: `O(n)` where `n` is the number of training samples.
## Conclusion

This project demonstrates the feasibility of using an XGBoost machine learning model to predict combinational logic depth before synthesis. The model achieves good accuracy on the synthetic dataset, as evidenced by low RMSE and MAE values and a high R-squared score during both train/test evaluation and cross-validation.  This approach shows promise for early timing violation detection and has the potential to significantly reduce design iteration time.


