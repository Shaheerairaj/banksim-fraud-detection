# Project Overview

This project aims to detect fraudulent activities using the BankSim dataset provided on Kaggle

- Performed EDA on data to understand variables and their relationship with dependant variable
- Built a selection of models with standard settings to get a baseline of model performances
- Applied PCA for dimentionality reduction and Grid Search to optimize performance of the most promising algorithms
- Built an app which takes user inputs for a dummy transaction and predicts according to the trained model if it is a fraudulent activity or not

# EDA Insights
- Data is heavily imbalanced towards none-fraudulent activities (there are significantly more non-fraud activities than actual frauds)
- Transaction amount is heavily skewed to the right and most frauds (per transaction category) occure at higher transaction amount ranges
- Highest amount of frauds occur if the transaction category is Sports
- Transaction amount has the highest correlation to fraud (49% correlation)


# Initial Model Building
Below are the performances of models without tuning or PCA.
I decided to go with recall as my main measure of performance.
The reason for this is because recall measures how accurately our model was able to correctly detect frauds out of all frauds that exist in the test dataset.
It is far more useful in my opinion for a bank to correctly detect all frauds and not too big of a deal if a few false positives were detected in the process.
But I've also included precision just to see how well our model predicts the non-fraudulent cases.

## Initial Model Performance
|         Model       | Recall | Precision |
|---------------------|--------|-----------|
| Logistic Regression | 0.6060 |   0.8971  |
|     Perceptron      | 0.5541 |   0.8699  |
|         SVC         | 0.5927 |   0.9336  |
|         KNN         | 0.6257 |   0.8693  |
|   Decision Trees    | 0.6664 |   0.6487  |
|    Random Forest    | 0.6404 |   0.8476  |
|  Gradient Boosting  | 0.6531 |   0.8824  |

As you can see, none of these models were particularly impressive.

After this, I performed the following:
1. Downsampled the data to fix the class imbalances
2. Applied PCA for dimentionality reduction
3. Selected the best 3 performing models for tuning (Logistic Regression, Perceptron and SVC)
4. Tuned the hyper parameters using grid search and k cross validation (k=10)

## Tuned Model Performance

|        Model        | Recall | Recall Improvement | Precision | Precision Improvement |
|---------------------|--------|--------------------|-----------|-----------------------|
| Logistic Regression | 0.9953 |        64.2%       |   0.919   |          2.4%         |
|       Perceptron    | 0.9860 |        77.9%       |   0.913   |          4.9%         |
|         SVC         | 1.0000 |        68.7%       |   0.911   |         -2.4%         |






# Resources Used:
1. Kaggle Dataset: https://www.kaggle.com/datasets/ealaxi/banksim1/data
2. Notebooks for Inspiration:
- https://www.kaggle.com/code/gpreda/financial-system-payment-eda-and-prediction
- https://www.kaggle.com/code/andradaolteanu/i-fraud-detection-eda-and-understanding
- https://www.kaggle.com/code/andradaolteanu/ii-fraud-detection-classify-cluster-pca

