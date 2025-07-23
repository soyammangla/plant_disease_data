# model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv(r"c:\Users\ASUS1\plant_disease_data.csv\plant_disease_data.csv")

# Features and target
X = df.drop("disease_present", axis=1)
y = df["disease_present"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "model.pkl")
print("✅ Model trained and saved as model.pkl")
