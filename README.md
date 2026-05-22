# QuickBooks Customer Intelligence Pipeline

An end-to-end machine learning project built on mock QuickBooks customer data across 2,000 accounts and 57,000+ transactions.

## Project Overview

This project demonstrates the full ML workflow from raw data to deployable insights:
- **ETL pipeline** — data generation, cleaning, and SQLite database setup
- **Customer Segmentation** — unsupervised K-Means clustering to identify customer groups
- **Churn Prediction** — supervised classification with Logistic Regression and Random Forest
- **Anomaly Detection** — Isolation Forest for flagging suspicious transactions *(Phase 4)*
- **NLP on Support Tickets** — topic modeling and sentiment analysis *(Phase 5)*
- **A/B Test Analysis** — statistical hypothesis testing on product experiments *(Phase 6)*
- **Recommender System** — product recommendations using matrix factorization *(Phase 7)*

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models |
| SQLite | Database |
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
- Identified four actionable segments:

| Segment | Size | Churn Rate | Key Trait |
|---|---|---|---|
| High Value Champions | 437 | 6% | Advanced plan, highest spend |
| Reliable Regulars | 810 | 12% | Steady, low support volume |
| Power Users | 308 | 18% | Massive transaction volume |
| At Risk | 445 | 56% | Frustrated, disengaged |

### Phase 3 — Churn Prediction
- Trained Logistic Regression and Random Forest classifiers
- Logistic Regression outperformed Random Forest on recall (8 vs 12 missed churners)
- Top churn predictors: support ticket volume, days since last transaction, total spend
- Key insight: churners telegraph departure through frustration and disengagement before cancelling

## Project Structure

qb-customer-intelligence/
├── data/
│   ├── generate_mock_data.py   # Generates all mock CSV files
│   ├── customers.csv
│   ├── subscriptions.csv
│   ├── transactions.csv
│   └── support_tickets.csv
├── db/
│   ├── setup_db.py             # Loads CSVs into SQLite
│   └── qb_customers.db
├── notebooks/
│   ├── 01_ETL_QB.ipynb         # Data generation and database setup
│   ├── 02_Segmentation_QB.ipynb # K-Means customer segmentation
│   └── 03_Churn_Prediction_QB.ipynb # Churn prediction models
├── reports/                    # Output reports and visualizations
└── requirements.txt

## How to Run

1. Clone the repo
2. Open any notebook in Google Colab
3. Run cells top to bottom — each notebook is self-contained

## Business Context

Built to mirror real-world problems faced by Intuit's AI team:
- Identifying at-risk customers before they cancel
- Segmenting customers for targeted retention campaigns  
- Detecting fraudulent transactions automatically
- Extracting insight from unstructured support ticket text

## Author
Weston White | [GitHub](https://github.com/WestonWhiteUCD)
