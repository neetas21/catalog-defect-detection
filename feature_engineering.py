import pandas as pd
import ast

df = pd.read_csv("appliances_labeled.csv")

# Title length (word count) - proxy for how much effort seller put in
df['title_length'] = df['title'].apply(lambda x: len(str(x).split()))

# Does it have a price at all?
df['has_price'] = ~(df['price'].isna() | (df['price'] == 'None'))

# Rating signals - already numeric, just fill any missing with 0
df['average_rating'] = df['average_rating'].fillna(0)
df['rating_number'] = df['rating_number'].fillna(0)

# Store - encode as a number (model can't use raw text directly)
df['store'] = df['store'].fillna('Unknown')
df['store_encoded'] = df['store'].astype('category').cat.codes

# Category - same encoding treatment
df['main_category'] = df['main_category'].fillna('Unknown')
df['category_encoded'] = df['main_category'].astype('category').cat.codes

# Final feature set + label, ready for model training
model_df = df[[
    'title_length', 'has_price', 'average_rating', 'rating_number',
    'store_encoded', 'category_encoded', 'is_defective'
]]

print(model_df.head())
print()
print(model_df.info())

model_df.to_csv("appliances_features.csv", index=False)
print("\nSaved to appliances_features.csv")
