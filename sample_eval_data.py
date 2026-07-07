import pandas as pd
import ast

df = pd.read_csv("appliances_labeled.csv")

# We need listings where description actually exists and is non-trivial
def has_real_description(value):
    if pd.isna(value):
        return False
    if isinstance(value, str) and value.strip() in ['[]', '']:
        return False
    return True

df['has_description'] = df['description'].apply(has_real_description)

# Filter to listings with real descriptions, title, and category present
eval_candidates = df[
    df['has_description'] &
    df['title'].notna() &
    df['main_category'].notna()
].copy()

print("Total eligible for eval:", len(eval_candidates))

# Sample 30 listings for our benchmark test (keeps API calls manageable)
eval_sample = eval_candidates.sample(n=30, random_state=42)

# Save this sample - these are our "masked ground truth" test cases
eval_sample[['title', 'main_category', 'features', 'description']].to_csv(
    "eval_sample.csv", index=False
)
print("Saved 30-row eval sample to eval_sample.csv")
print(eval_sample[['title', 'main_category', 'description']].head(3))
