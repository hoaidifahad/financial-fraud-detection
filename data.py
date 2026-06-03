import pandas as pd
import sqlite3

# 1. قراءة البيانات (سنقرأ جزء منها فقط في البداية لتسريع العمل وتجربة الكود)
# يمكنك إزالة nrows لاحقاً لقراءة الملف كاملاً إذا كان جهازك يتحمل
paysim_df = pd.read_csv('paysim.csv', nrows=100000)  # قراءة أول 100,000 صف فقط
print("Data loaded successfully!")

# 2. إنشاء اتصال بقاعدة بيانات محلية (سيتم إنشاء ملف باسم fraud_database.db في نفس المجلد)
conn = sqlite3.connect('fraud_database.db')

# 3. حفظ إطار البيانات (DataFrame) داخل قاعدة البيانات كجدول اسمه 'transactions'
print("Creating SQL Database...")
paysim_df.to_sql('transactions', conn, if_exists='replace', index=False)
print("Database created successfully!")

# 4. لنتأكد أن النظام يعمل بكتابة استعلام SQL بسيط لاستخراج أول 5 عمليات
query = """
SELECT step, type, amount, nameOrig, nameDest, isFraud 
FROM transactions 
LIMIT 5;
"""

sample_data = pd.read_sql_query(query, conn)
print("\n--- Sample Data from SQL ---")
print(sample_data)

# إغلاق الاتصال بقاعدة البيانات
conn.close()