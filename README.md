# Crop Yield Prediction with ANN

## Overview
This project predicts crop yield using agricultural sensor data such as soil moisture, soil pH, temperature, rainfall, humidity, and other farming conditions. A neural network is built and compared between a baseline version and a modified, improved version.

## Steps
- Data cleaning: fixed wrong data types (comma decimal to dot), filled missing values using mode for categorical columns and median for skewed numeric columns, removed invalid rows, dropped redundant date columns
- Exploratory data analysis: checked distributions, outliers, and correlation between features
- Preprocessing: split data into train, validation, and test sets (70:10:20), scaled numeric features, one hot encoded categorical features
- Baseline model: a simple neural network with 2 hidden layers
- Modified model: added Leaky ReLU, batch normalization, and scaled the target variable to fix a negative R2 score problem in the baseline model
- Evaluation: compared both models using R2 score, MSE, MAE, and MAPE

## Result
The baseline model performed poorly with a negative R2 score because the yield values had a wide and fluctuating range. The modified model fixed this by scaling the target variable and improving the network architecture, resulting in a much better R2 score.

## Tech Stack
Python, pandas, numpy, scikit-learn, TensorFlow, Keras, matplotlib, seaborn
