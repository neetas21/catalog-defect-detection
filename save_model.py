import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("appliances_features.csv")
X = df.drop('is_defective', axis=1)
y = df['is_defective']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y_train == False).sum() / (y_train == True).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(model, "xgb_model.pkl")
print("Model saved to xgb_model.pkl")
