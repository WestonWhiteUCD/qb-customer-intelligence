import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random, os, uuid

# ── Reproducibility ────────────────────────────────────────────────────────────
# Seeds mean every run produces identical data — critical for teamwork and
# for reproducing your ML results later.
fake = Faker()
np.random.seed(42)
random.seed(42)
Faker.seed(42)

NUM_CUSTOMERS = 2000
OUTPUT_DIR    = "data"

# ── Helper: random date within a range ────────────────────────────────────────
def random_date(start_days_ago, end_days_ago=0):
    start = datetime.now() - timedelta(days=start_days_ago)
    end   = datetime.now() - timedelta(days=end_days_ago)
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.randint(0, max(1, int(delta))))

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: CUSTOMERS
# One row per QuickBooks account. Describes who each customer is.
# These columns will become features in our ML models later.
# ══════════════════════════════════════════════════════════════════════════════
print("Generating customers...")

business_types = ["Retail","Restaurant","Consulting","Freelance",
                  "Construction","Healthcare","Legal","E-commerce"]
states         = ["CA","TX","NY","FL","WA","IL","GA","CO","AZ","NC"]

customers = []
for i in range(NUM_CUSTOMERS):
    signup_date = random_date(start_days_ago=730, end_days_ago=30)
    customers.append({
        "customer_id":   f"CUST_{i+1:05d}",
        "name":          fake.company(),
        "email":         fake.company_email(),
        "state":         random.choice(states),
        "business_type": random.choice(business_types),
        "signup_date":   signup_date.strftime("%Y-%m-%d"),
        "num_employees": max(1, int(np.random.lognormal(mean=1.5, sigma=0.8))),
        "annual_revenue": round(random.uniform(50_000, 5_000_000), 2),
    })

customers_df = pd.DataFrame(customers)
customers_df.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
print(f"  -> {len(customers_df):,} customers written")
print(customers_df.head(3))
