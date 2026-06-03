# =============================================================================
# All imports 
# =============================================================================
import pandas as pd
import sqlite3
import requests
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# =============================================================================
# configuration section.
# =============================================================================
DB_PATH       = 'fraud_database.db'   # Path to your SQLite database
MODEL_PATH    = 'fraud_model.joblib'  # Where to save the trained model
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3"
OLLAMA_TIMEOUT = 60  # seconds to wait before giving up on the LLM


# =============================================================================
# STEP 1: Connect to database and load data
# =============================================================================
print("Connecting to database and extracting data...")

try:
    conn = sqlite3.connect(DB_PATH)

    # We focus on TRANSFER and CASH_OUT only since they are the most relevant for fraud detection.
    query = """
    SELECT type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud
    FROM transactions
    WHERE type IN ('TRANSFER', 'CASH_OUT')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

except Exception as e:
    print(f"\n[ERROR] Could not load data from database: {e}")
    print(f"  → Make sure '{DB_PATH}' exists and contains a 'transactions' table.")
    sys.exit(1)

print(f"Data loaded successfully! Total records: {len(df)}")


# =============================================================================
# STEP 2: Data validation — check for missing values
# =============================================================================
missing = df.isnull().sum()
if missing.any():
    print("\n[WARNING] Missing values detected in the following columns:")
    print(missing[missing > 0])
    print("  → Please clean your data before training. Exiting.")
    sys.exit(1)
else:
    print("No missing values found. Data is clean.")


# =============================================================================
# STEP 3: Feature Engineering
# =============================================================================
print("\nEngineering new features...")

# 'errorBalanceOrig': (newBalance + amount) - oldBalance should equal 0.
# Any non-zero result means the sender's balance was manipulated.
df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']

# 'errorBalanceDest': same check but for the receiver's account.
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

# Encode 'type' column (TRANSFER / CASH_OUT) into numeric 0/1
df = pd.get_dummies(df, columns=['type'], drop_first=True)

print("\n--- Data after Feature Engineering (first 5 rows) ---")
print(df.head())


# =============================================================================
# STEP 4: Train/Test Split and Model Training
# =============================================================================
print("\nPreparing data for training...")

# Separate features (X) from the target label (y = isFraud)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training the Random Forest model... (This might take a minute)")

model = RandomForestClassifier(
    n_estimators=100,          
    class_weight='balanced',
    random_state=42,
    n_jobs=-1                 
)
model.fit(X_train, y_train)

print("Model trained successfully!\n")


# =============================================================================
# STEP 5: Save the trained model
# =============================================================================
joblib.dump(model, MODEL_PATH)
print(f"Model saved to '{MODEL_PATH}' — reload it next run with joblib.load()")


# =============================================================================
# STEP 6: Evaluate the model
# =============================================================================
y_pred = model.predict(X_test)

print("--- Model Evaluation Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))



print("\n--- Top Feature Importances ---")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).to_string())


# =============================================================================
# STEP 7: Select a confirmed fraud case for GenAI analysis
# =============================================================================
print("\nML model caught fraud transactions. Selecting a case for GenAI analysis...")

true_fraud_indices = X_test[(y_pred == 1) & (y_test == 1)].index

if len(true_fraud_indices) == 0:
    print("   No true positives found. Falling back to any flagged case...")
    true_fraud_indices = X_test[y_pred == 1].index

sample_fraud = X_test.loc[true_fraud_indices[0]]


# =============================================================================
# STEP 8: Build the investigation prompt for the LLM
# =============================================================================
def generate_investigation_prompt(transaction_data):
    prompt = f"""
    You are an expert Anti-Money Laundering (AML) and Fraud Investigator.
    Our Machine Learning system has flagged the following transaction as HIGHLY SUSPICIOUS.

    Transaction Details:
    - Amount: ${transaction_data['amount']:,.2f}
    - Sender Old Balance: ${transaction_data['oldbalanceOrg']:,.2f}
    - Sender New Balance: ${transaction_data['newbalanceOrig']:,.2f}
    - Receiver Old Balance: ${transaction_data['oldbalanceDest']:,.2f}
    - Receiver New Balance: ${transaction_data['newbalanceDest']:,.2f}
    - Mathematical Error in Sender's Balance: ${transaction_data['errorBalanceOrig']:,.2f}

    Task:
    Write a brief, professional alert summary (1 paragraph) explaining WHY this transaction
    is suspicious based on the mathematical errors or empty balances.
    Provide the summary in both English and Arabic.
    """
    return prompt

prompt_text = generate_investigation_prompt(sample_fraud)


# =============================================================================
# STEP 9: Send prompt to local LLM (Llama 3 via Ollama)
# =============================================================================
print(f"\nSending data to Local LLM ({OLLAMA_MODEL}) via Ollama API...")

def get_local_llm_response(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        return response.json()['response']

    except requests.exceptions.Timeout:
        return (f"[ERROR] Request timed out after {OLLAMA_TIMEOUT}s.\n"
                f"  → Is Ollama running? Try: ollama serve")
    except Exception as e:
        return (f"[ERROR] Could not reach local LLM: {e}\n"
                f"  → Make sure Ollama is running and '{OLLAMA_MODEL}' is installed.\n"
                f"  → Install with: ollama pull {OLLAMA_MODEL}")


ai_report = get_local_llm_response(prompt_text)

print("\n" + "="*50)
print("             REAL LOCAL GenAI REPORT             ")
print("="*50)
print(ai_report)
print("="*50)
