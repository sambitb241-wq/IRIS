# Iris Flower Classification

## Overview

This project classifies Iris flowers into one of three species — setosa, versicolor, and virginica — based on sepal and petal measurements. It goes beyond a single model by performing exploratory data analysis, training and comparing four different classifiers, and visualizing the results.

## Dataset

The dataset is loaded directly from scikit-learn's built-in `load_iris()` function (no CSV file needed). It contains 150 samples with the following features:

- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`
- `species` (target: setosa, versicolor, virginica)

## Approach

1. **Load and inspect data** — load the dataset, check shape, missing values, class distribution, and summary statistics.
2. **Exploratory Data Analysis (EDA)** — generate a pairplot of feature relationships by species and a correlation heatmap between features.
3. **Preprocessing** — encode species labels numerically with `LabelEncoder`, split into train/test sets (80/20, stratified), and scale features with `StandardScaler`.
4. **Model training and comparison** — train four classifiers and evaluate each on accuracy and a full classification report:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (linear kernel)
5. **Best model selection** — automatically picks the model with the highest test accuracy.
6. **Confusion matrix** — visualizes prediction results for the best-performing model.
7. **Feature importance** — if Random Forest is the best model, plots and prints which features contributed most to classification.
8. **Prediction on new data** — demonstrates a prediction for a single new flower sample.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Run the script directly — no external dataset file is required:

```
python iris_classification.py
```

## Output

Running the script prints to the console:

- Dataset preview, shape, missing values, class distribution, and summary statistics
- Accuracy and classification report for each of the four models
- The best-performing model and its accuracy
- Feature importances (if Random Forest wins)
- A prediction for a sample flower measurement

It also saves the following image files to the working directory:

- `pairplot.png` — pairwise feature relationships by species
- `correlation_heatmap.png` — correlation between features
- `confusion_matrix.png` — confusion matrix for the best model
- `feature_importance.png` — feature importance chart (Random Forest only)

## Notes

Comparing multiple models rather than committing to one gives a clearer picture of which algorithm best separates the three species on this dataset. Since Iris is a small, well-separated dataset, most models tend to perform very well (often 95%+ accuracy), so results may vary slightly by random seed.
