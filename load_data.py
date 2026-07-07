from datasets import load_dataset
import pandas as pd

# Load the Appliances metadata category
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Appliances",
    trust_remote_code=True
)

# Convert to a pandas DataFrame for easier inspection
df = pd.DataFrame(dataset["full"])

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:\n", df.head(3))

# Save a local copy so we don't have to re-download every time
df.to_csv("appliances_raw.csv", index=False)
print("\nSaved to appliances_raw.csv")
