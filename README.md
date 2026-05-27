# QuickBooks Customer Intelligence Pipeline

An end-to-end machine learning project simulating the work of a Senior AI Scientist
at Intuit — built on mock QuickBooks customer data across 2,000 accounts and 57,000+ transactions.

## Project Overview

This project demonstrates the full ML workflow from raw data to deployed insights:
- **Phase 1 — ETL Pipeline** — data generation, cleaning, and SQLite database setup
- **Phase 2 — Customer Segmentation** — unsupervised K-Means clustering
- **Phase 3 — Churn Prediction** — supervised classification with Logistic Regression and Random Forest
- **Phase 4 — Anomaly Detection** — Isolation Forest for flagging suspicious transactions
- **Phase 5 — NLP on Support Tickets** — TF-IDF, LDA topic modeling, churn by topic
- **Phase 6 — A/B Test Analysis** — statistical hypothesis testing on product experiments
- **Phase 7 — Recommender System** — popularity baseline and SVD matrix factorization

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models |
| SQLite | Database |
| scipy | Statistical testing |
| matplotlib / seaborn | Visualization |
| Faker | Mock data generation |
| Google Colab | Development environment |
| GitHub | Version control |

## Key Results

### Phase 1 — ETL
- Generated 4 relational tables: 2,000 customers, 2,000 subscriptions, 57,000+ transactions, 5,000+ support tickets
- Loaded into SQLite with SQL verification queries confirming data integrity
- Overall churn rate: 21.4% — realistic for a SaaS product

### Phase 2 — Customer Segmentation
- Used elbow method and silhouette scores to select K=4 clusters
- Identified four actionable business segments:

| Segment | Size | Churn Rate | Key Trait | Action |
|---|---|---|---|---|
| High Value Champions | 437 | 6% | Advanced plan, highest spend | Protect |
| Reliable Regulars | 810 | 12% | Steady, low support volume | Upsell |
| Power Users | 308 | 18% | Massive transaction volume | Retain |
| At Risk | 445 | 56% | Frustrated, disengaged | Intervene |

### Phase 3 — Churn Prediction
- Trained Logistic Regression and Random Forest classifiers
- Logistic Regression outperformed Random Forest on recall (8 vs 12 missed churners)
- Top churn predictors: support ticket volume, days since last transaction, total spend
- Key insight: churners telegraph departure through frustration and disengagement before cancelling
- At Intuit's scale of 1M customers the recall difference = ~40,000 additional undetected churners

### Phase 4 — Anomaly Detection
- Isolation Forest flagged 1,183 suspicious transactions at 2% contamination rate
- Top anomalies were 5-11x each customer's normal transaction amount
- Engineered relative features (amount vs customer average) rather than absolute thresholds
- Model correctly detected both suspiciously large AND suspiciously small transactions

### Phase 5 — NLP on Support Tickets
- TF-IDF vectorization reduced 5,297 tickets to 134 meaningful terms
- LDA topic modeling discovered 4 natural topics from raw ticket text:

| Topic | Key Words | Churn Rate |
|---|---|---|
| Frustrated Users | crashes, downgrade, plan | Highest |
| Billing Complaints | price, without notice, why | High |
| Billing Disputes | cancelled, billed, reverse charge | Medium |
| Feature Requests | export, excel, add | Lowest |

- Key insight: Feature Request customers churn least — they are engaged and want the product improved

### Phase 6 — A/B Test Analysis
- Simulated experiment: new onboarding flow vs existing for 1,000 customers per group
- Treatment group averaged 4.75 more transactions in first 90 days
- t-statistic: 7.297, p-value < 0.0001, Cohen's d = 0.326 (medium effect)
- 95% confidence interval: [3.48, 6.03] additional transactions
- Conclusion: new onboarding genuinely increases engagement — recommend full rollout

### Phase 7 — Recommender System
- Built customer-product interaction matrix (2,000 x 4) from behavioral signals
- Popularity baseline recommends most popular plan per segment
- SVD matrix factorization with k=2 latent factors explains 89.1% of variance
- 34.8% agreement between baseline and SVD — SVD finds individual nuance baseline misses
- Recommended validation approach: A/B test both systems on live customers

## Project Structure

    qb-customer-intelligence/
    ├── data/
    │   ├── generate_mock_data.py
    │   ├── customers.csv
    │   ├── subscriptions.csv
    │   ├── transactions.csv
    │   └── support_tickets.csv
    ├── db/
    │   ├── setup_db.py
    │   └── qb_customers.db
    ├── notebooks/
    │   ├── 01_ETL_QB.ipynb
    │   ├── 02_Segmentation_QB.ipynb
    │   ├── 03_Churn_Prediction_QB.ipynb
    │   ├── 04_Anomaly_Detection_QB.ipynb
    │   ├── 05_NLP_QB.ipynb
    │   ├── 06_AB_Test_QB.ipynb
    │   └── 07_Recommender_QB.ipynb
    ├── reports/
    └── requirements.txt

## How to Run

1. Clone the repo
2. Open any notebook in Google Colab
3. Run cells top to bottom — each notebook is self-contained
4. Data regenerates automatically using seed=42 for full reproducibility

## Business Context

Built to mirror real-world problems faced by Intuit's AI team:
- Identifying at-risk customers before they cancel
- Segmenting customers for targeted retention campaigns
- Detecting fraudulent transactions automatically
- Extracting insight from unstructured support ticket text
- Proving product changes work through rigorous statistical testing
- Recommending the right product to the right customer

## Author
Weston White | [GitHub](https://github.com/WestonWhiteUCD)
