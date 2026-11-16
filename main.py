
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
dataset = fetch_ucirepo(id=320)
data = dataset.data.features.copy()
data["G3"] = dataset.data.targets["G3"]

# Clean data
data = data.dropna(subset=["G3"]).drop_duplicates()

# Create target
data["at_risk"] = (data["G3"] < 10).astype(int)

# Input and output
X = data.drop(columns=["G3", "at_risk"])
y = data["at_risk"]

# Identify columns
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# Preprocessing
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Three ML models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Train models
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    print(name, "Training Completed")

print("All Models Trained Successfully!")


from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

print("\nMODEL PERFORMANCE")
print("-" * 40)

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nModel:", name)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 4))


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Train Random Forest
best_model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42))
])

best_model.fit(X_train, y_train)

# Predict test data
y_pred = best_model.predict(X_test)

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Not At Risk", "At Risk"],
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()


# Student Risk Prediction Demo

student = X_test.iloc[[0]]

prediction = best_model.predict(student)[0]
actual = y_test.iloc[0]

print("\nSTUDENT RISK PREDICTION")
print("----------------------------")

print("Predicted Risk:", prediction)
print("Actual Risk:", actual)

if prediction == 1:
    print("Result: Student is AT RISK")
else:
    print("Result: Student is NOT AT RISK")




import os
import joblib
from sklearn.metrics import ConfusionMatrixDisplay

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Save trained model
joblib.dump(best_model, "outputs/student_risk_model.pkl")

# Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Not At Risk", "At Risk"],
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("\nModel and Graph Saved Successfully!")
