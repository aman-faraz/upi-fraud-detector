import requests
import random
import time
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ── Config ────────────────────────────────────────────────────────────────────
API_URL       = "http://localhost:8000"
SLEEP_SECONDS = 1      # 1 transaction per second
FRAUD_RATE    = 0.15   # 15% will be fraud scenarios

# ── Load real VPAs from feature store ────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STORE_PATH  = os.path.join(BASE_DIR, 'data', 'processed',
                            'account_feature_store.csv')

print("Loading account feature store...")
store       = pd.read_csv(STORE_PATH)
normal_vpas = store[store['is_mule']==0]['vpa'].tolist()
mule_vpas   = store[store['is_mule']==1]['vpa'].tolist()

print(f"✅ Loaded {len(normal_vpas):,} normal VPAs")
print(f"✅ Loaded {len(mule_vpas):,}  mule VPAs")
print()

# ── Transaction generators ────────────────────────────────────────────────────
def make_legit_transaction():
    """Normal person paying a merchant or friend"""
    hour   = datetime.now().hour
    amount = round(np.random.lognormal(6.0, 0.8), 2)
    amount = min(amount, 50000)

    return {
        "txn_id":       f"TXN_{uuid.uuid4().hex[:10].upper()}",
        "sender_vpa":   random.choice(normal_vpas),
        "receiver_vpa": random.choice(normal_vpas),
        "amount":       amount,
        "hour":         hour,
        "txn_type":     random.choices(
                            ["P2P","MERCHANT","BILL"],
                            weights=[50, 40, 10]
                        )[0]
    }

def make_fraud_transaction(fraud_type=None):
    """
    Fraud scenarios using real mule VPAs from feature store.
    Three patterns:
      A — Normal sender → mule receiver (mule routing)
      B — High amount just under limit
      C — Night time suspicious transfer
    """
    hour        = datetime.now().hour
    fraud_type  = fraud_type or random.choice(['A','B','C'])

    if fraud_type == 'A':
        # Mule routing — normal sends to mule
        return {
            "txn_id":       f"TXN_{uuid.uuid4().hex[:10].upper()}",
            "sender_vpa":   random.choice(normal_vpas),
            "receiver_vpa": random.choice(mule_vpas),  # ← mule receiver
            "amount":       round(random.uniform(5000, 50000), 2),
            "hour":         hour,
            "txn_type":     "P2P"
        }

    elif fraud_type == 'B':
        # Amount just under ₹1 lakh limit
        return {
            "txn_id":       f"TXN_{uuid.uuid4().hex[:10].upper()}",
            "sender_vpa":   random.choice(normal_vpas),
            "receiver_vpa": random.choice(mule_vpas),
            "amount":       round(random.uniform(95000, 99999), 2),
            "hour":         hour,
            "txn_type":     "P2P"
        }

    else:
        # Mule sending to another mule (layering)
        return {
            "txn_id":       f"TXN_{uuid.uuid4().hex[:10].upper()}",
            "sender_vpa":   random.choice(mule_vpas),
            "receiver_vpa": random.choice(mule_vpas),
            "amount":       round(random.uniform(20000, 80000), 2),
            "hour":         hour,
            "txn_type":     "P2P"
        }

# ── Stream loop ───────────────────────────────────────────────────────────────
print("=" * 55)
print("  UPI Transaction Stream Started")
print(f"  Speed     : 1 transaction / {SLEEP_SECONDS} second")
print(f"  Fraud rate: {FRAUD_RATE*100:.0f}%")
print(f"  API       : {API_URL}")
print("  Press Ctrl+C to stop")
print("=" * 55)
print()

count       = 0
fraud_count = 0
error_count = 0

while True:
    try:
        # Decide fraud or legit
        is_fraud_scenario = random.random() < FRAUD_RATE

        if is_fraud_scenario:
            txn = make_fraud_transaction()
        else:
            txn = make_legit_transaction()

        # Send to API
        resp   = requests.post(
            f"{API_URL}/predict",
            json=txn,
            timeout=3
        )
        result = resp.json()

        count += 1
        if result['is_fraud']:
            fraud_count += 1

        # Console output
        flag  = "🔴 FRAUD" if result['is_fraud'] else "✅  legit"
        risk  = result['risk_label']
        score = result['fraud_score']
        amt   = txn['amount']
        svpa  = txn['sender_vpa'][:20]
        rvpa  = txn['receiver_vpa'][:20]

        print(
            f"[{count:05d}] {flag} | "
            f"Score:{score:.4f} | "
            f"Risk:{risk:<6} | "
            f"₹{amt:>10,.2f} | "
            f"{svpa:<20} → {rvpa}"
        )

        # Summary every 20 transactions
        if count % 20 == 0:
            rate = fraud_count / count * 100
            print()
            print(f"  ── Summary: {count} scored | "
                  f"{fraud_count} fraud ({rate:.1f}%) | "
                  f"{error_count} errors ──")
            print()

    except KeyboardInterrupt:
        print()
        print("=" * 55)
        print(f"  Stream stopped by user")
        print(f"  Total scored  : {count:,}")
        print(f"  Fraud flagged : {fraud_count:,} "
              f"({fraud_count/max(count,1)*100:.1f}%)")
        print(f"  Errors        : {error_count:,}")
        print("=" * 55)
        break

    except Exception as e:
        error_count += 1
        print(f"  ⚠️  Error [{error_count}]: {e}")
        if error_count >= 5:
            print("  Too many errors — is the API running?")
            print(f"  Start it: cd api && uvicorn main:app --reload")
            break

    time.sleep(SLEEP_SECONDS)
    