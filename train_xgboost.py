import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("appliances_features.csv")

X = df.drop('is_defective', axis=1)
y = df['is_defective']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale_pos_weight handles imbalance for XGBoost (similar purpose to class_weight='balanced')
# It's the ratio of negative to positive class - tells the model to pay more attention to the minority class
scale_pos_weight = (y_train == False).sum() / (y_train == True).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance - which signals mattered most
print("\nFeature Importance:")
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance)
