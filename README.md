# Rain Prediction with Logistic Regression

This Python script uses logistic regression to predict the probability of rain based on weather conditions. It leverages the [Weather Forecast Dataset](https://www.kaggle.com/datasets/zeeshier/weather-forecast-dataset) from Kaggle, ideal for beginners in machine learning.

## Features Used
- Temperature  
- Humidity  
- Wind Speed  
- Cloud Cover  
- Pressure  

## How It Works
1. Loads and preprocesses the dataset  
2. Standardizes features using `StandardScaler`  
3. Trains a logistic regression model  
4. Evaluates accuracy and classification metrics  
5. Predicts rain probability for new weather inputs

## Quick Start
```bash
poetry install
poetry run py logistic/weather.py
```

## 🇨🇳 中文说明
这是中国版本的文档。
[逻辑回归](https://github.com/easai/logistic/blob/main/logistic-regression-zh.ipynb)

## 🇯🇵 日本語の説明
これは日本語版のドキュメントです。
[ロジスティックス回帰の説明](https://github.com/easai/logistic/blob/main/logistic-regression.ipynb)
