import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\BALAJI\OneDrive\Documents\TNS\heart_disease_dataset.csv")

# Split the dataset into features and target variable
X = df.drop('heart_disease', axis=1)  # Features
y = df['heart_disease']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the model (optional step)
accuracy = model.score(X_test_scaled, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
