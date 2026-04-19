import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# ✅ FIXED: Use relative path (works on Windows, Mac, Linux & GitHub Actions)
df = pd.read_csv("heart_disease_dataset.csv")

# Split features & target
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save models (to root folder to match your app.py)
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
print('✅ Model & scaler saved successfully!')