import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlite3
import time
import os
import streamlit.components.v1 as components
from pyvis.network import Network

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL  = "https://upi-fraud-api-sm90.onrender.com"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, 'api', 'fraud_results.db')
STORE_PATH = os.path.join(BASE_DIR, 'data', 'processed',
                          'account_feature_store.csv')

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .header-bar {
        background: linear-gradient(90deg, #0C2340, #1A56A0);
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    .risk-high   { color:#E63946; font-weight:bold; }
    .risk-medium { color:#FFC107; font-weight:bold; }
    .risk-low    { color:#28A745; font-weight:bold; }
    div[data-testid="metric-container"] {
        background: #F8FAFF;
        border: 0.5px solid #D0DCF0;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/", timeout=2)
        return r.status_code == 200
    except:
        return False

def get_stats():
    try:
        return requests.get(f"{API_URL}/stats", timeout=2).json()
    except:
        return {}

def load_db(limit=500, fraud_only=False):
    try:
        params = {"limit": limit, "fraud_only": fraud_only}
        r = requests.get(
            f"{API_URL}/transactions",
            params=params,
            timeout=10
        )
        data = r.json()
        if "transactions" in data and len(data["transactions"]) > 0:
            return pd.DataFrame(data["transactions"])
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def load_store():
    try:
        return pd.read_csv(STORE_PATH)
    except:
        return pd.DataFrame()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <h1 style="margin:0; font-size:26px;">
        🔍 UPI Fraud Detection — Live Dashboard
    </h1>
    <p style="margin:4px 0 0; opacity:0.8; font-size:13px;">
        Real-time scoring powered by XGBoost + Graph Features + SHAP
    </p>
</div>
""", unsafe_allow_html=True)

# ── API status check ──────────────────────────────────────────────────────────
if not check_api():
    st.error("❌ API not reachable — run this in terminal first:")
    st.code("cd api && uvicorn main:app --reload --port 8000")
    st.stop()

st.success("✅ API connected — scoring engine live")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    auto_refresh = st.toggle("Auto Refresh (5s)", value=False)
    fraud_only   = st.toggle("Show Fraud Only",   value=False)
    limit        = st.slider("Rows to load", 50, 500, 100)
    st.divider()

    # Model info from API
    st.caption("🤖 Model Info")
    try:
        info = requests.get(f"{API_URL}/").json()
        st.write(f"**Version:** {info['model']}")
        st.write(f"**Threshold:** {info['threshold']}")
        st.write(f"**Features:** {info['features']}")
        st.write(f"**Accounts indexed:** {info['accounts_indexed']:,}")
    except:
        st.write("Could not load model info")

    st.divider()
    st.caption("🔎 Account Lookup")
    lookup_vpa = st.text_input("Enter VPA to inspect", "")
    if st.button("Lookup") and lookup_vpa:
        try:
            r = requests.get(
                f"{API_URL}/account/{lookup_vpa}",
                timeout=3
            ).json()
            st.write(f"**Known:** {r['known']}")
            st.write(f"**Risk:** {r['risk_profile']}")
            st.write(f"**PageRank:** {r['features']['pagerank']:.6f}")
            st.write(f"**In-degree:** {r['features']['in_degree']}")
            st.write(f"**Avg velocity 1h:** {r['features']['avg_velocity_1h']:.2f}")
            st.write(f"**Is Mule:** {'🔴 Yes' if r['features']['is_mule'] else '🟢 No'}")
        except:
            st.error("VPA not found or API error")

# ── Load data ─────────────────────────────────────────────────────────────────
stats = get_stats()
df    = load_db(limit=limit, fraud_only=fraud_only)
store = load_store()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Monitor",
    "🧪 Score Transaction",
    "🕸️ Network Graph",
    "📋 Account Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Metric cards
    st.subheader("Live Statistics")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Scored",   f"{int(stats.get('total',0) or 0):,}")
    c2.metric("Fraud Flagged",  f"{int(stats.get('total_fraud',0) or 0):,}")
    c3.metric("Fraud Rate",     f"{stats.get('fraud_rate_pct',0) or 0}%")
    c4.metric("High Risk",      f"{int(stats.get('high_risk',0) or 0):,}")
    c5.metric("Mule Senders",   f"{int(stats.get('mule_senders',0) or 0):,}")
    c6.metric("Mule Receivers", f"{int(stats.get('mule_receivers',0) or 0):,}")

    st.divider()

    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour']      = df['timestamp'].dt.hour

        # Row 1 charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fraud Score Distribution")
            fig = px.histogram(
                df, x='fraud_score',
                color='is_fraud',
                nbins=40,
                color_discrete_map={0:'#2C7BE5', 1:'#E63946'},
                barmode='overlay',
                labels={'is_fraud':'Label','fraud_score':'Score'}
            )
            fig.for_each_trace(lambda t: t.update(
                name='Fraud' if t.name=='1' else 'Legit'
            ))
            fig.update_layout(height=280, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Fraud Rate by Hour")
            hourly = df.groupby('hour').agg(
                total=('is_fraud','count'),
                fraud=('is_fraud','sum')
            ).reset_index()
            hourly['rate'] = (hourly['fraud']/hourly['total']*100).round(2)
            fig2 = px.bar(
                hourly, x='hour', y='rate',
                color='rate',
                color_continuous_scale='RdYlGn_r',
                labels={'rate':'Fraud Rate (%)','hour':'Hour'}
            )
            fig2.update_layout(
                height=280, margin=dict(t=10,b=10),
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Row 2 charts
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Risk Label Breakdown")
            risk = df['risk_label'].value_counts().reset_index()
            risk.columns = ['Risk','Count']
            fig3 = px.pie(
                risk, names='Risk', values='Count',
                color='Risk',
                color_discrete_map={
                    'HIGH':'#E63946',
                    'MEDIUM':'#FFC107',
                    'LOW':'#28A745'
                },
                hole=0.45
            )
            fig3.update_layout(height=280, margin=dict(t=10,b=10))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.subheader("Amount by Risk Level")
            fig4 = px.box(
                df, x='risk_label', y='amount',
                color='risk_label',
                color_discrete_map={
                    'HIGH':'#E63946',
                    'MEDIUM':'#FFC107',
                    'LOW':'#28A745'
                },
                labels={'amount':'Amount (₹)','risk_label':'Risk'}
            )
            fig4.update_layout(
                height=280, margin=dict(t=10,b=10),
                showlegend=False
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.divider()

        # Fraud queue table
        st.subheader("🚨 Transaction Queue")
        show_cols = [c for c in [
            'txn_id','amount','sender_vpa','receiver_vpa',
            'fraud_score','is_fraud','risk_label',
            'sender_is_mule','receiver_is_mule',
            'top_reasons','timestamp'
        ] if c in df.columns]

        def colour_risk(val):
            if val == 'HIGH':   return 'color:#E63946;font-weight:bold'
            if val == 'MEDIUM': return 'color:#FFC107;font-weight:bold'
            return 'color:#28A745'

        styled = df[show_cols].style \
        .map(colour_risk, subset=['risk_label']) \
        .format({'amount':'₹{:,.2f}','fraud_score':'{:.4f}'})

        st.dataframe(styled, use_container_width=True, height=320)
        st.caption(
            f"Showing {len(df):,} | "
            f"Fraud: {df['is_fraud'].sum():,} | "
            f"Legit: {(df['is_fraud']==0).sum():,}"
        )

    else:
        st.info("No transactions yet — run simulate_stream.py to start")
        st.code("python simulate_stream.py")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MANUAL SCORING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🧪 Score Any Transaction")
    st.caption(
        "Enter sender VPA + receiver VPA — all graph and velocity "
        "features are auto-looked up from the feature store."
    )

    # Show sample VPAs from store
    if len(store) > 0:
        with st.expander("📋 Sample VPAs you can use"):
            col_n, col_m = st.columns(2)
            col_n.write("**Normal accounts:**")
            normal_vpas = store[store['is_mule']==0]['vpa'].head(5).tolist()
            for v in normal_vpas:
                col_n.code(v)
            col_m.write("**Mule accounts (high risk):**")
            mule_vpas = store[store['is_mule']==1]['vpa'].head(5).tolist()
            for v in mule_vpas:
                col_m.code(v)

    with st.form("score_form"):
        col_a, col_b = st.columns(2)
        sender_vpa   = col_a.text_input(
            "Sender VPA",
            value=normal_vpas[0] if len(store)>0 else "sender@okicici"
        )
        receiver_vpa = col_b.text_input(
            "Receiver VPA",
            value=mule_vpas[0] if len(store)>0 else "receiver@paytm"
        )

        col_c, col_d, col_e = st.columns(3)
        amount   = col_c.number_input("Amount (₹)", 1.0, 100000.0, 98500.0)
        hour     = col_d.slider("Hour of Day", 0, 23, 2)
        txn_type = col_e.selectbox("Transaction Type",
                                    ["P2P","MERCHANT","BILL"])

        submitted = st.form_submit_button(
            "🔍 Score Transaction",
            use_container_width=True
        )

    if submitted:
        payload = {
            "txn_id":       f"TXN_MANUAL_{int(time.time())}",
            "sender_vpa":   sender_vpa,
            "receiver_vpa": receiver_vpa,
            "amount":       amount,
            "hour":         hour,
            "txn_type":     txn_type
        }

        with st.spinner("Scoring..."):
            try:
                resp   = requests.post(
                    f"{API_URL}/predict",
                    json=payload, timeout=5
                )
                result = resp.json()

                # Result metrics
                r1, r2, r3 = st.columns(3)
                r1.metric("Fraud Score", result['fraud_score'])
                r2.metric("Decision",
                    "🔴 FRAUD" if result['is_fraud'] else "🟢 LEGIT")
                r3.metric("Risk Level", result['risk_label'])

                # Mule flags
                m1, m2 = st.columns(2)
                m1.info(
                    f"Sender is mule: "
                    f"{'🔴 YES' if result['sender_is_mule'] else '🟢 No'} | "
                    f"Known account: {result['sender_known']}"
                )
                m2.info(
                    f"Receiver is mule: "
                    f"{'🔴 YES' if result['receiver_is_mule'] else '🟢 No'} | "
                    f"Known account: {result['receiver_known']}"
                )

                # SHAP reasons
                st.subheader("🔎 Why this decision?")
                for i, reason in enumerate(result['top_reasons'], 1):
                    icon = "🔴" if "↑ fraud" in reason else "🟢"
                    st.write(f"{icon} **Reason {i}:** {reason}")

                # Gauge chart
                fig_g = go.Figure(go.Indicator(
                    mode  = "gauge+number+delta",
                    value = result['fraud_score'],
                    delta = {'reference': result['threshold']},
                    title = {'text': "Fraud Probability"},
                    gauge = {
                        'axis': {'range':[0,1]},
                        'bar':  {'color':'#E63946'},
                        'steps':[
                            {'range':[0.00,0.40],'color':'#28A745'},
                            {'range':[0.40,0.75],'color':'#FFC107'},
                            {'range':[0.75,1.00],'color':'#E63946'},
                        ],
                        'threshold':{
                            'line':{'color':'black','width':4},
                            'thickness':0.75,
                            'value':result['threshold']
                        }
                    }
                ))
                fig_g.update_layout(
                    height=300,
                    margin=dict(t=40,b=10)
                )
                st.plotly_chart(fig_g, use_container_width=True)

                # Features used
                with st.expander("🔬 Full feature vector used"):
                    feat_df = pd.DataFrame(
                        result['features_used'].items(),
                        columns=['Feature','Value']
                    )
                    st.dataframe(feat_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NETWORK GRAPH
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🕸️ Transaction Network — Top Risk Accounts")
    st.caption(
        "Visualises the top 30 highest-PageRank accounts and "
        "their connections. Red = mule account."
    )

    if len(store) > 0:
        if st.button("Generate Network Graph"):
            with st.spinner("Building network..."):

                # Top 30 accounts by pagerank
                top_accounts = store.nlargest(30, 'pagerank')

                # Load recent transactions for edges
                df_txns = load_db(limit=200)

                # Build PyVis network
                net = Network(
                    height="500px", width="100%",
                    bgcolor="#0C2340",
                    font_color="white"
                )
                net.barnes_hut()

                # Add nodes
                for _, acc in top_accounts.iterrows():
                    color = "#E63946" if acc['is_mule'] else "#2C7BE5"
                    size  = max(10, min(40, acc['in_degree']))
                    net.add_node(
                        acc['account_id'],
                        label=acc['account_id'],
                        color=color,
                        size=size,
                        title=(
                            f"VPA: {acc['vpa']}\n"
                            f"PageRank: {acc['pagerank']:.6f}\n"
                            f"In-degree: {acc['in_degree']}\n"
                            f"Mule: {'Yes' if acc['is_mule'] else 'No'}\n"
                            f"Avg velocity 1h: {acc['avg_velocity_1h']:.1f}"
                        )
                    )

                # Add edges from recent transactions
                if len(df_txns) > 0 and 'sender_vpa' in df_txns.columns:
                    top_ids = set(top_accounts['account_id'].tolist())
                    df_check = load_db(limit=500)
                    # Use accounts CSV to map VPA to account_id
                    accounts_path = os.path.join(
                        BASE_DIR, 'data', 'processed', 'account_feature_store.csv'
                    )
                    acc_df = pd.read_csv(accounts_path)
                    vpa_to_id = dict(zip(
                        acc_df['vpa'],
                        acc_df['account_id']
                    ))

                    added = 0
                    for _, row in df_check.iterrows():
                        s = vpa_to_id.get(row.get('sender_vpa',''))
                        r = vpa_to_id.get(row.get('receiver_vpa',''))
                        if s in top_ids and r in top_ids and s != r:
                            net.add_edge(
                                s, r,
                                color="#FFC107" if row['is_fraud'] else "#555",
                                width=2 if row['is_fraud'] else 1
                            )
                            added += 1
                        if added >= 60:
                            break

                # Save and display
                graph_path = os.path.join(BASE_DIR, 'dashboard', 'graph.html')
                net.save_graph(graph_path)
                with open(graph_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                components.html(html, height=520)

                st.caption(
                    "🔴 Red nodes = mule accounts | "
                    "🔵 Blue = normal | "
                    "Node size = in-degree | "
                    "🟡 Yellow edges = fraud transactions"
                )
    else:
        st.info("Feature store not loaded — run feature engineering first")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ACCOUNT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Account Feature Store Explorer")

    if len(store) > 0:
        col_f1, col_f2, col_f3 = st.columns(3)
        show_mules = col_f1.toggle("Mule accounts only", False)
        sort_by    = col_f2.selectbox(
            "Sort by",
            ["pagerank","in_degree","avg_velocity_1h","max_velocity_1h"]
        )
        top_n = col_f3.slider("Show top N", 10, 200, 50)

        display = store.copy()
        if show_mules:
            display = display[display['is_mule']==1]
        display = display.nlargest(top_n, sort_by)

        show_store_cols = [
            'account_id','vpa','is_mule','pagerank',
            'in_degree','out_degree',
            'avg_velocity_1h','avg_velocity_24h','max_velocity_1h'
        ]
        available_cols = [c for c in show_store_cols if c in display.columns]

        def highlight_mule(row):
            if row.get('is_mule', 0) == 1:
                return ['background-color: #FAE8E8'] * len(row)
            return [''] * len(row)

        styled_store = display[available_cols].style\
            .apply(highlight_mule, axis=1)\
            .format({
                'pagerank':         '{:.6f}',
                'avg_velocity_1h':  '{:.2f}',
                'avg_velocity_24h': '{:.2f}',
                'max_velocity_1h':  '{:.2f}',
            })

        st.dataframe(styled_store, use_container_width=True, height=400)
        st.caption(
            f"Showing {len(display):,} accounts | "
            f"Mules highlighted in red"
        )

        # Distribution charts
        st.divider()
        st.subheader("Feature Distributions")
        dc1, dc2 = st.columns(2)

        with dc1:
            fig_pr = px.histogram(
                store, x='pagerank',
                color='is_mule',
                nbins=50,
                color_discrete_map={0:'#2C7BE5', 1:'#E63946'},
                title='PageRank Distribution',
                labels={'is_mule':'Is Mule'}
            )
            fig_pr.for_each_trace(lambda t: t.update(
                name='Mule' if t.name=='1' else 'Normal'
            ))
            fig_pr.update_layout(height=280, margin=dict(t=30,b=10))
            st.plotly_chart(fig_pr, use_container_width=True)

        with dc2:
            fig_v = px.histogram(
                store, x='avg_velocity_1h',
                color='is_mule',
                nbins=50,
                color_discrete_map={0:'#2C7BE5', 1:'#E63946'},
                title='Avg Velocity 1h Distribution',
                labels={'is_mule':'Is Mule'}
            )
            fig_v.for_each_trace(lambda t: t.update(
                name='Mule' if t.name=='1' else 'Normal'
            ))
            fig_v.update_layout(height=280, margin=dict(t=30,b=10))
            st.plotly_chart(fig_v, use_container_width=True)

    else:
        st.warning("Feature store CSV not found")

# ── Auto refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()