import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast

df = pd.read_csv("llm_results.csv")

# Clean the real_description field - it's a stringified list, extract the text
def clean_real_desc(value):
    try:
        desc_list = ast.literal_eval(value)
        return " ".join(desc_list) if desc_list else ""
    except:
        return str(value)

df['real_description_clean'] = df['real_description'].apply(clean_real_desc)

# Load embedding model (same family you used in FashionFind)
model = SentenceTransformer('all-MiniLM-L6-v2')

def score_similarity(real, generated):
    embeddings = model.encode([real, generated])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

df['zero_shot_score'] = df.apply(lambda r: score_similarity(r['real_description_clean'], r['zero_shot']), axis=1)
df['few_shot_score'] = df.apply(lambda r: score_similarity(r['real_description_clean'], r['few_shot']), axis=1)
df['cot_score'] = df.apply(lambda r: score_similarity(r['real_description_clean'], r['cot']), axis=1)

print("Average similarity scores:")
print("Zero-shot:", df['zero_shot_score'].mean())
print("Few-shot:", df['few_shot_score'].mean())
print("CoT:", df['cot_score'].mean())

df.to_csv("llm_scored_results.csv", index=False)
print("\nSaved to llm_scored_results.csv")
