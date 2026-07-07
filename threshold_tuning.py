
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# Instead of predict() which uses 0.5 by default, get raw probabilities
y_probs = model.predict_proba(X_test)[:, 1]  # probability of being "defective"

# Try a few different thresholds
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_thresh = (y_probs >= threshold).astype(bool)
    print(f"\n--- Threshold: {threshold} ---")
    print(classification_report(y_test, y_pred_thresh))
