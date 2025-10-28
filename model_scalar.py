

# XGBoost Classifier for Pima Diabetes
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

data_path = "D:\\AI_ML\\health_p\\data\\pima_diabetes.csv"
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
           'BMI','DiabetesPedigreeFunction','Age','Outcome']

data = pd.read_csv(data_path, names=columns)


# Some columns should never be zero; replace with median

cols_invalid = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for c in cols_invalid:
    data[c] = data[c].replace(0, data[c].median())


#  Separate features and target

X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values


# Scale features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#  Create XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=200,         # number of boosting rounds
    learning_rate=0.1,        # step size shrinkage
    max_depth=3,              # maximum tree depth
    random_state=42,
    use_label_encoder=False,  # avoid warning
    eval_metric='logloss'     # binary classification metric
)


# Track training and test loss

eval_set = [(X_train, y_train), (X_test, y_test)]

xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False  # we will print manually
)

# iteration-wise train/test loss

results = xgb_model.evals_result()
num_rounds = len(results['validation_0']['logloss'])

print("\nIteration-wise Loss:")
for i in range(num_rounds):
    # print every 100 iterations for brevity
    if i % 50 == 0 or i == num_rounds - 1:
        train_loss = results['validation_0']['logloss'][i]
        test_loss = results['validation_1']['logloss'][i]
        print(f"Iteration {i}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Make predictions
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

# Lower threshold to 0.4 to increase sensitivity
threshold = 0.3
y_pred_new = (y_pred_prob >= threshold).astype(int)

# Evaluate model

cm = confusion_matrix(y_test, y_pred_new)
print("Confusion Matrix:")
print(cm)

# Extract TN, FP, FN, TP
TN, FP, FN, TP = cm.ravel()  # unpack confusion matrix

# Accuracy
accuracy = accuracy_score(y_test, y_pred_new)
print(f"Accuracy: {accuracy:.4f}")

# Sensitivity / Recall / True Positive Rate
sensitivity = TP / (TP + FN)
print(f"Sensitivity (Recall): {sensitivity:.4f}")

# Specificity / True Negative Rate
specificity = TN / (TN + FP)
print(f"Specificity: {specificity:.4f}")


# plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.show()

#  Plot training vs test loss

plt.figure(figsize=(7,4))
plt.plot(range(num_rounds), results['validation_0']['logloss'], label='Train Loss')
plt.plot(range(num_rounds), results['validation_1']['logloss'], label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('XGBoost Loss per Iteration')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance

importances = xgb_model.feature_importances_
feature_names = data.columns[:-1]

import joblib

# Save to a specific folder
joblib.dump(xgb_model, "D:/AI_ML/health_p/xgb_model.pkl")
joblib.dump(scaler, "D:/AI_ML/health_p/scaler.pkl")


plt.figure(figsize=(8,4))
sns.barplot(x=feature_names, y=importances)
plt.xticks(rotation=45)
plt.title("Feature Importances - XGBoost")
plt.show()
