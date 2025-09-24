import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
df = pd.read_csv('weather_forecast_data.csv')

# Step 2: Encode the target variable ('Rain')
df['Rain'] = df['Rain'].map({'no rain': 0, 'rain': 1})

# Step 3: Define features and target
X = df[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
y = df['Rain']

# Step 4: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Today's weather as a DataFrame with matching column names
today_df = pd.DataFrame([{
    'Temperature': 25.44,
    'Humidity': 99.54,
    'Wind_Speed': 4.17,
    'Cloud_Cover': 92.79,
    'Pressure': 1042.70
}])

# Standardize and predict
today_scaled = scaler.transform(today_df)
rain_prob = model.predict_proba(today_scaled)[0][1]
print(f"Probability of rain today: {rain_prob:.2%}")
