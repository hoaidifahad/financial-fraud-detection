import pandas as pd
import sqlite3


paysim_df = pd.read_csv('paysim.csv', nrows=100000)
print("Data loaded successfully!")

conn = sqlite3.connect('fraud_database.db')

print("Creating SQL Database...")
paysim_df.to_sql('transactions', conn, if_exists='replace', index=False)
print("Database created successfully!")

query = """
SELECT step, type, amount, nameOrig, nameDest, isFraud 
FROM transactions 
LIMIT 5;
"""

sample_data = pd.read_sql_query(query, conn)
print("\n--- Sample Data from SQL ---")
print(sample_data)

conn.close()
