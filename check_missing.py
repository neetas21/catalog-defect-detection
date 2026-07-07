import pandas as pd

df = pd.read_csv("appliances_raw.csv")

print("Total rows:", len(df))
print()

# Check emptiness for each relevant field
print("Missing/empty title:", df['title'].isna().sum())
print("Missing/empty price:", (df['price'].isna() | (df['price'] == 'None')).sum())
print("Missing/empty description:", (df['description'].isna() | (df['description'] == '[]')).sum())
print("Missing/empty features:", (df['features'].isna() | (df['features'] == '[]')).sum())
print("Missing/empty categories:", (df['categories'].isna() | (df['categories'] == '[]')).sum())
print("Missing average_rating:", df['average_rating'].isna().sum())
