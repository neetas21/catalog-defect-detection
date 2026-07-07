import pandas as pd
import ast
import os
import time
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df = pd.read_csv("eval_sample.csv")

def clean_features(features_str):
    """Turn the stringified list into readable text, or empty string if none"""
    try:
        features_list = ast.literal_eval(features_str)
        if features_list:
            return ", ".join(features_list[:3])  # just first 3 to keep prompt short
    except:
        pass
    return "Not available"

def zero_shot_prompt(title, category, features):
    return f"""Write a short product description (2-3 sentences) for this item.

Title: {title}
Category: {category}
Features: {features}

Description:"""

def few_shot_prompt(title, category, features):
    return f"""Write a short product description (2-3 sentences) based on the title, category, and features.

Example 1:
Title: KitchenAid Stand Mixer, 5 Quart
Category: Kitchen Appliances
Features: 10-speed control, tilt-head design, stainless steel bowl
Description: This KitchenAid stand mixer offers powerful 10-speed control for all your baking needs. The tilt-head design makes it easy to add ingredients and swap attachments. Built with a durable stainless steel bowl for long-lasting performance.

Example 2:
Title: Dyson V8 Cordless Vacuum
Category: Home Appliances
Features: Cordless, lightweight, HEPA filtration
Description: The Dyson V8 delivers powerful cordless cleaning in a lightweight design perfect for quick cleanups. Its HEPA filtration captures allergens for a healthier home. Ideal for both carpets and hard floors.

Now write one for:
Title: {title}
Category: {category}
Features: {features}
Description:"""

def cot_prompt(title, category, features):
    return f"""Think step by step about this product, then write a short description (2-3 sentences).

Title: {title}
Category: {category}
Features: {features}

Step 1: What type of product is this, based on the title and category?
Step 2: What are its key functional benefits, based on the features?
Step 3: Who would use this product?

Now write the final description only (don't include the steps in your answer):
Description:"""

def call_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

results = []

for idx, row in df.iterrows():
    title = row['title']
    category = row['main_category']
    features = clean_features(row['features'])
    real_description = row['description']

    print(f"Processing {idx+1}/{len(df)}: {title[:50]}...")

    zero_shot_result = call_llm(zero_shot_prompt(title, category, features))
    time.sleep(1)  # avoid rate limits
    few_shot_result = call_llm(few_shot_prompt(title, category, features))
    time.sleep(1)
    cot_result = call_llm(cot_prompt(title, category, features))
    time.sleep(1)

    results.append({
        'title': title,
        'real_description': real_description,
        'zero_shot': zero_shot_result,
        'few_shot': few_shot_result,
        'cot': cot_result
    })

results_df = pd.DataFrame(results)
results_df.to_csv("llm_results.csv", index=False)
print("\nSaved all results to llm_results.csv")
