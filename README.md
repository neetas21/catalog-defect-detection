# Catalog Defect Detection + LLM-Assisted Attribute Correction

## Business Problem
E-commerce catalogs contain millions of listings, but a meaningful share have incomplete or low-quality attributes — missing descriptions, empty feature lists, missing categories. These gaps hurt the customer's buying decision (can't compare products, can't judge fit or specs) and hurt the platform's own search/discovery ranking for that listing. Manually auditing listings at scale isn't feasible, so this project asks:

> Can we automatically flag listings likely to have incomplete attributes, then use an LLM to generate the missing content — prioritizing correction effort on the listings that need it most?

This mirrors real catalog-quality and compliance work in retail systems teams (e.g., Amazon's Retail Business Systems group), where the goal is maintaining a high-quality, complete, and trustworthy product catalog at scale.

## Dataset
**Amazon Reviews 2023** (McAuley Lab) — Appliances category metadata, ~94,300 listings. Fields include title, price, description, features, categories, ratings, and store.

## Stage 1 — Defect Labeling
Defined "defective" using a threshold rule rather than any single missing field: a listing is labeled defective if **2 or more** of `description`, `features`, `categories` are empty. A single missing field was too common (noise); requiring 2+ produced a meaningful, non-trivial signal — **13.35% of listings** labeled defective.

`price` was excluded from the label definition (missing in ~50% of listings — too common to be a useful *defect* signal) but retained as a *predictor* feature.

## Stage 2 — Classical ML Classifier
**v1 — Logistic Regression (baseline)**
- Features: title length, has_price, average_rating, rating_number, store (encoded), category (encoded)
- `class_weight='balanced'` to address the 13.35% imbalance
- Result: Recall 0.83, Precision 0.25, F1 0.38 (default 0.5 threshold)

**v2 — XGBoost (improved model)**
- `scale_pos_weight` to handle imbalance
- Result: Recall 0.79, Precision 0.34, F1 0.48, Accuracy 0.77 — meaningful improvement over v1 with only a small recall trade-off

**Feature importance (XGBoost):** `has_price` dominated at ~72% importance — suggesting incomplete listings are often a systemic sourcing/pipeline issue (missing price, description, and features together) rather than random gaps.

**Threshold tuning:** Tested thresholds 0.3–0.7. Selected **0.4** as the operating threshold — prioritizing recall (catching real defects) since a missed defect (broken listing stays live, uncorrected) is costlier than a false positive (LLM attempts to "fix" an already-fine listing).

| Threshold | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|
| 0.3 | 0.90 | 0.27 | 0.41 | 0.66 |
| **0.4** | **0.85** | **0.30** | **0.45** | **0.72** |
| 0.5 | 0.79 | 0.34 | 0.48 | 0.77 |
| 0.6 | 0.70 | 0.39 | 0.50 | 0.81 |
| 0.7 | 0.55 | 0.46 | 0.50 | 0.85 |

## Stage 3 — GenAI Attribute Correction + Prompt Benchmarking
For listings with real descriptions, masked the ground truth and generated a replacement using an LLM (Groq LLaMA 3.3 70B), testing three prompting strategies on a 30-listing sample:

- **Zero-shot** — direct instruction, no examples
- **Few-shot** — two example title→description pairs provided
- **Chain-of-Thought (CoT)** — reasoning steps (product type → benefits → audience) before generating

Scored using cosine similarity (Sentence Transformers, `all-MiniLM-L6-v2`) between generated and real descriptions.

| Strategy | Avg. Similarity |
|---|---|
| **Zero-shot** | **0.742** |
| Few-shot | 0.738 |
| CoT | 0.730 |

**Finding:** Zero-shot performed marginally best. Likely explanation: product description generation from structured fields (title, category) is a low-complexity generation task, so CoT's added reasoning scaffolding provides no benefit and may introduce phrasing drift. This suggests prompt complexity should be matched to task complexity rather than applied uniformly — a more nuanced takeaway than "more sophisticated prompting always wins."

## Stage 4 — Deployment + Monitoring
Built a Streamlit app that:
1. Accepts a listing's attributes as input
2. Scores it with the trained XGBoost model (threshold = 0.4)
3. If flagged defective, generates a description using the winning zero-shot strategy
4. Logs every check (timestamp, inputs, defect probability, output) to `monitoring_log.csv` for ongoing performance tracking

## Key Takeaways
- Demonstrates the classical ML vs. GenAI trade-off explicitly: ML for structured classification (fast, cheap, interpretable), LLM only invoked selectively where generation is actually needed
- Threshold and model choices were driven by business cost reasoning, not just accuracy
- Offline evaluation (prompt benchmarking) used a practical "mask and test" methodology applicable when no labeled eval set exists
- End-to-end: labeling logic → model training/tuning → LLM generation → deployment → monitoring
