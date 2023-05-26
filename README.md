# Business-Analyst-Time-Series-Modeling-
# Time Series Modeling: Domestic Market (Contract) Blow Molding, Low Price

This project focuses on time series modeling to predict the domestic market (contract) blow molding, low price. It utilizes various time series models, including ARIMA, LSTM, and additional models for benchmarking and evaluation.

## Objective

The main objective of this project is to build accurate time series models to predict the domestic market (contract) blow molding, low price. The project aims to compare the performance of different models and select the best model based on evaluation metrics such as RMSE, MSE, R2, and directional accuracy.

## Data

The project utilizes raw Excel data containing historical information on the domestic market (contract) blow molding, low price. The data is preprocessed and split into training and testing sets for model development and evaluation.

## Models Implemented

1. ARIMA: The project implements the ARIMA model for time series forecasting. The model captures the autocorrelation and seasonality in the data to make predictions.

2. LSTM: Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that can effectively model time dependencies. The project builds an LSTM model to capture temporal patterns and make accurate predictions.

3. Lazy Predict: The project utilizes the LazyPredict library to benchmark the accuracy of various models, including MLP, RNN, LSTM, GRU, Logistic Regression, Random Forest, and Naive Bayes Classifier. The library provides a quick way to evaluate multiple models and compare their performance.

4. Additional Models: The project explores additional models for time series analysis, such as XGBoost and Seasonal-Trend Decomposition using LOESS (STL).

## Evaluation Metrics

The performance of each model is evaluated using the following metrics:

- RMSE (Root Mean Squared Error): A measure of the average difference between predicted and actual values.
- MSE (Mean Squared Error): The mean of the squared differences between predicted and actual values.
- R2 (Coefficient of Determination): Represents the proportion of variance in the dependent variable explained by the independent variables.
- Directional Accuracy: Measures the percentage of correctly predicted directions (up or down) compared to the actual directions.

## Results and Conclusion

A comprehensive model reliability matrix is presented, comparing the performance metrics of all implemented models for different forecasting time periods. The models are evaluated based on RMSE, MSE, directional accuracy, and other relevant metrics to determine their effectiveness in predicting the domestic market (contract) blow molding, low price.

Furthermore, the project explores feature space analysis, correlation, and XGBoost to identify indicators that explain historical periods when the confidence level in the forecasts decreased.

## Repository Contents

- `data/`: Directory containing the raw Excel data.
- `notebooks/`: Directory containing Jupyter notebooks with the code for data preprocessing, model building, and evaluation.
- `scripts/`: Directory containing Python scripts implementing the models.
- `README.md`: This file, providing an overview of the project.

## Getting Started

To reproduce the results or further explore the project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/your-repository.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebooks or Python scripts in the `notebooks/` or `scripts/` directory, respectively.

Feel free to adjust the models, parameters, and data preprocessing steps to suit your specific requirements.

## Conclusion

The time series modeling project for predicting the domestic market (contract) blow molding, low price demonstrates the implementation and evaluation of various models. The models
