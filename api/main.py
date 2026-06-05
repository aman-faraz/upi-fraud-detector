from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import sqlite3
import shap
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="UPI Fraud Detector API",
    description=(
        "Real-time UPI transaction fraud scoring. "
        "Send sender_vpa + receiver_vpa + amount — "
        "all graph and velocity features are auto-looked up."
    ),
    version="2.0.0"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'processed')
DB_PATH    = os.path.join(BASE_DIR, 'api', 'fraud_results.db')

# ── Load model files ──────────────────────────────────────────────────────────
print("Loading model files...")
model     = joblib.load(os.path.join(MODELS_DIR, 'xgb_fraud_model.pkl'))
threshold = joblib.load(os.path.join(MODELS_DIR, 'threshold.pkl'))
features  = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
print(f"✅ Model loaded | Threshold: {threshold:.4f} | Features: {len(features)}")

# ── Load account feature store ────────────────────────────────────────────────
print("Loading account feature store...")
store_path = os.path.join(DATA_DIR, 'account_feature_store.csv')
feat_store = pd.read_csv(store_path)

# Build VPA → features lookup dictionary
# Key   : vpa string  e.g. "raj.kumar221@okicici"
# Value : dict of all pre-computed features
vpa_lookup = {}
for _, row in feat_store.iterrows():
    vpa_lookup[row['vpa']] = {
        'account_id':      row['account_id'],
        'pagerank':        row['pagerank'],
        'in_degree':       int(row['in_degree']),
        'out_degree':      int(row['out_degree']),
        'avg_velocity_1h': row['avg_velocity_1h'],
        'avg_velocity_24h':row['avg_velocity_24h'],
        'max_velocity_1h': row['max_velocity_1h'],
        'is_mule':         int(row['is_mule']),
    }

print(f"✅ Feature store loaded | {len(vpa_lookup):,} accounts indexed")

# Default features for unknown VPAs
# If a VPA not in store (new account) — use conservative defaults
DEFAULT_FEATURES = {
    'pagerank':         0.0002,
    'in_degree':        5,
    'out_degree':       5,
    'avg_velocity_1h':  1.0,
    'avg_velocity_24h': 3.0,
    'max_velocity_1h':  2.0,
    'is_mule':          0,
}

def lookup_vpa(vpa: str) -> dict:
    """
    Look up pre-computed features for a VPA.
    Returns default features if VPA not found (new account).
    """
    if vpa in vpa_lookup:
        return vpa_lookup[vpa]
    # New/unknown account — flag with defaults
    # In production: trigger real-time graph update
    return DEFAULT_FEATURES.copy()

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id          TEXT,
            amount          REAL,
            sender_vpa      TEXT,
            receiver_vpa    TEXT,
            fraud_score     REAL,
            is_fraud        INTEGER,
            risk_label      TEXT,
            top_reasons     TEXT,
            sender_is_mule  INTEGER,
            receiver_is_mule INTEGER,
            timestamp       TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()
print("✅ Database ready")

# ── Request schema — simplified ───────────────────────────────────────────────
# Caller only needs to send these 6 fields
# Everything else is auto-computed from feature store
class Transaction(BaseModel):
    txn_id:       str
    sender_vpa:   str
    receiver_vpa: str
    amount:       float
    hour:         int
    txn_type:     str   # "P2P", "MERCHANT", or "BILL"

class BatchRequest(BaseModel):
    transactions: list[Transaction]

# ── /predict endpoint ─────────────────────────────────────────────────────────
@app.post("/predict")
def predict(txn: Transaction):
    """
    Score a UPI transaction for fraud.
    Only requires sender_vpa, receiver_vpa, amount, hour, txn_type.
    All graph and velocity features auto-looked up from feature store.
    """

    # Step 1 — Look up sender and receiver features
    sender_feats   = lookup_vpa(txn.sender_vpa)
    receiver_feats = lookup_vpa(txn.receiver_vpa)

    sender_known   = txn.sender_vpa   in vpa_lookup
    receiver_known = txn.receiver_vpa in vpa_lookup

    # Step 2 — Derive time features
    day_of_week = datetime.now().weekday()
    is_weekend  = 1 if day_of_week >= 5 else 0
    is_night    = 1 if txn.hour >= 22 or txn.hour <= 5 else 0
    near_limit  = 1 if txn.amount >= 95000 else 0

    # Step 3 — Transaction type encoding
    is_p2p      = 1 if txn.txn_type.upper() == "P2P"      else 0
    is_merchant = 1 if txn.txn_type.upper() == "MERCHANT" else 0
    is_bill     = 1 if txn.txn_type.upper() == "BILL"     else 0

    # Step 4 — Build full feature vector
    data = {
        "hour":              txn.hour,
        "day_of_week":       day_of_week,
        "is_weekend":        is_weekend,
        "is_night":          is_night,
        "log_amount":        np.log1p(txn.amount),
        "near_limit":        near_limit,
        "is_p2p":            is_p2p,
        "is_merchant":       is_merchant,
        "is_bill":           is_bill,
        # Velocity — from sender's historical profile
        "velocity_1h":       sender_feats['avg_velocity_1h'],
        "velocity_24h":      sender_feats['avg_velocity_24h'],
        # Graph — sender features
        "sender_pagerank":   sender_feats['pagerank'],
        "sender_in_degree":  sender_feats['in_degree'],
        "sender_out_degree": sender_feats['out_degree'],
        # Graph — receiver features
        "receiver_pagerank": receiver_feats['pagerank'],
        "receiver_in_degree":receiver_feats['in_degree'],
    }

    # Step 5 — Score
    X        = pd.DataFrame([data])[features]
    score    = float(model.predict_proba(X)[0, 1])
    is_fraud = int(score >= threshold)

    # Step 6 — Risk label
    if score >= 0.75:
        risk_label = "HIGH"
    elif score >= 0.40:
        risk_label = "MEDIUM"
    else:
        risk_label = "LOW"

    # Step 7 — SHAP explanation
    shap_vals   = explainer.shap_values(X)[0]
    top_indices = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_reasons = []
    for i in top_indices:
        fname     = features[i]
        fval      = round(float(data[fname]), 4)
        sval      = round(float(shap_vals[i]), 4)
        direction = "↑ fraud" if shap_vals[i] > 0 else "↓ legit"
        top_reasons.append(
            f"{fname}={fval} (impact={sval} {direction})"
        )

    # Step 8 — Save to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''INSERT INTO predictions
           (txn_id, amount, sender_vpa, receiver_vpa,
            fraud_score, is_fraud, risk_label, top_reasons,
            sender_is_mule, receiver_is_mule, timestamp)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
        (
            txn.txn_id, txn.amount,
            txn.sender_vpa, txn.receiver_vpa,
            round(score, 4), is_fraud, risk_label,
            " | ".join(top_reasons),
            sender_feats.get('is_mule', 0),
            receiver_feats.get('is_mule', 0),
            datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()

    # Step 9 — Return
    return {
        "txn_id":          txn.txn_id,
        "fraud_score":     round(score, 4),
        "is_fraud":        is_fraud,
        "risk_label":      risk_label,
        "threshold":       round(float(threshold), 4),
        "top_reasons":     top_reasons,
        "sender_known":    sender_known,
        "receiver_known":  receiver_known,
        "sender_is_mule":  sender_feats.get('is_mule', 0),
        "receiver_is_mule":receiver_feats.get('is_mule', 0),
        "features_used":   data,
        "timestamp":       datetime.now().isoformat()
    }

# ── /predict/batch endpoint ───────────────────────────────────────────────────
@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """
    Score multiple transactions at once.
    Returns list of results in same order as input.
    """
    results = []
    for txn in req.transactions:
        result = predict(txn)
        results.append(result)
    return {
        "total":   len(results),
        "flagged": sum(1 for r in results if r['is_fraud']),
        "results": results
    }

# ── /account/{vpa} endpoint ───────────────────────────────────────────────────
@app.get("/account/{vpa:path}")
def get_account(vpa: str):
    """
    Look up pre-computed features for any VPA.
    Useful for debugging and dashboard account search.
    """
    feats = lookup_vpa(vpa)
    known = vpa in vpa_lookup
    return {
        "vpa":     vpa,
        "known":   known,
        "features": feats,
        "risk_profile": (
            "HIGH — Mule account detected"   if feats.get('is_mule') else
            "HIGH — Very active account"     if feats.get('avg_velocity_1h', 0) > 10 else
            "MEDIUM — Moderately active"     if feats.get('avg_velocity_1h', 0) > 5 else
            "LOW — Normal account"
        )
    }

# ── /transactions endpoint ────────────────────────────────────────────────────
@app.get("/transactions")
def get_transactions(limit: int = 50, fraud_only: bool = False):
    conn  = sqlite3.connect(DB_PATH)
    where = "WHERE is_fraud = 1" if fraud_only else ""
    rows  = conn.execute(
        f'''SELECT * FROM predictions
            {where}
            ORDER BY id DESC LIMIT {limit}'''
    ).fetchall()
    conn.close()
    cols = ['id','txn_id','amount','sender_vpa','receiver_vpa',
            'fraud_score','is_fraud','risk_label','top_reasons',
            'sender_is_mule','receiver_is_mule','timestamp']
    return {
        "total":        len(rows),
        "transactions": [dict(zip(cols, r)) for r in rows]
    }

# ── /stats endpoint ───────────────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('''
        SELECT
            COUNT(*)                    as total,
            SUM(is_fraud)               as total_fraud,
            ROUND(AVG(is_fraud)*100, 2) as fraud_rate_pct,
            ROUND(AVG(fraud_score), 4)  as avg_score,
            ROUND(MAX(fraud_score), 4)  as max_score,
            COUNT(CASE WHEN risk_label="HIGH"   THEN 1 END) as high_risk,
            COUNT(CASE WHEN risk_label="MEDIUM" THEN 1 END) as medium_risk,
            COUNT(CASE WHEN risk_label="LOW"    THEN 1 END) as low_risk,
            SUM(sender_is_mule)         as mule_senders,
            SUM(receiver_is_mule)       as mule_receivers
        FROM predictions
    ''').fetchone()
    conn.close()
    cols = ['total','total_fraud','fraud_rate_pct','avg_score',
            'max_score','high_risk','medium_risk','low_risk',
            'mule_senders','mule_receivers']
    return dict(zip(cols, rows))

# ── /health endpoint ──────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status":           "running",
        "model":            "XGBoost UPI Fraud Detector v2.0",
        "threshold":        round(float(threshold), 4),
        "features":         len(features),
        "accounts_indexed": len(vpa_lookup),
    }