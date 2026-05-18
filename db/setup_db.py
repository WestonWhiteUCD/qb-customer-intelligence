import pandas as pd
import sqlite3
import os

DB_PATH  = "db/qb_customers.db"
DATA_DIR = "data"

conn = sqlite3.connect(DB_PATH)

tables = {
    "customers":       "customers.csv",
    "subscriptions":   "subscriptions.csv",
    "transactions":    "transactions.csv",
    "support_tickets": "support_tickets.csv",
}

for table_name, csv_file in tables.items():
    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"  Loaded {table_name}: {len(df):,} rows")

conn.close()
print("✓ Database ready at db/qb_customers.db")
