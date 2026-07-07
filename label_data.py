import pandas as pd
import ast

df = pd.read_csv("appliances_raw.csv")

# Helper: check if a field is "empty" (handles NaN and stringified empty lists like '[]')
def is_empty(value):
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in ['[]', '']:
        return True
    return False

# Flag each field individually
df['missing_description'] = df['description'].apply(is_empty)
df['missing_features'] = df['features'].apply(is_empty)
df['missing_categories'] = df['categories'].apply(is_empty)

# Count how many of the 3 key fields are missing per row
df['missing_count'] = df[['missing_description', 'missing_features', 'missing_categories']].sum(axis=1)

# Defect label: 2 or more missing = defective
df['is_defective'] = df['missing_count'] >= 2

# Drop the 9 rows with no title at all — not useful signal, just broken rows
df = df[df['title'].notna()]

print("Total rows after cleanup:", len(df))
print()
print("Class balance:")
print(df['is_defective'].value_counts())
print()
print("Percent defective:", round(df['is_defective'].mean() * 100, 2), "%")

# Save labeled version for next stage
df.to_csv("appliances_labeled.csv", index=False)
print("\nSaved to appliances_labeled.csv")
