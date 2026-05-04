import streamlit as st
import httpx
import pandas as pd
from datetime import datetime
import json
import uuid

st.set_page_config(page_title="Finance Anomaly Detector", page_icon="💸", layout="wide")

st.title("💸 Finance Anomaly Detector")
st.markdown("Detect anomalous personal finance transactions using Isolation Forest and SHAP explainability.")

API_URL = "http://localhost:8000/api/v1/predict"

def predict_transaction(transactions_df):
    try:
        # Convert dataframe to list of dicts
        payload = transactions_df.to_dict(orient='records')
        
        # Add required fields if missing
        for i, row in enumerate(payload):
            if 'transaction_id' not in row:
                row['transaction_id'] = str(uuid.uuid4())
            if 'timestamp' not in row:
                row['timestamp'] = datetime.now().isoformat()
            elif isinstance(row['timestamp'], pd.Timestamp):
                row['timestamp'] = row['timestamp'].isoformat()
        
        response = httpx.post(API_URL, json=payload, params={"include_explanation": True}, timeout=10.0)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.error(f"Model not ready: {response.json().get('detail')}")
            return None
        else:
            st.error(f"Error from API: {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend API. Make sure FastAPI is running. Error: {e}")
        return None

tab1, tab2 = st.tabs(["Single Transaction", "Batch Upload (CSV)"])

with tab1:
    st.subheader("Analyze a Single Transaction")
    
    with st.form("single_transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.01, value=50.0, step=1.0)
            category = st.selectbox("Category", ["Groceries", "Dining", "Entertainment", "Utilities", "Travel", "Shopping", "Transfer", "Other"])
            
        with col2:
            merchant = st.text_input("Merchant", value="Supermarket")
            timestamp = st.date_input("Date", value=datetime.today())
            
        submitted = st.form_submit_button("Analyze Transaction")
        
        if submitted:
            df = pd.DataFrame([{
                "amount": amount,
                "category": category,
                "merchant": merchant,
                "timestamp": timestamp.isoformat() + "T00:00:00"
            }])
            
            with st.spinner("Analyzing..."):
                results = predict_transaction(df)
                
                if results:
                    result = results[0]
                    st.divider()
                    
                    # Display results
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        if result["is_anomaly"]:
                            st.error("🚨 **ANOMALY DETECTED**")
                        else:
                            st.success("✅ **NORMAL TRANSACTION**")
                        
                        st.metric("Anomaly Score (Lower is more anomalous)", f"{result['score']:.4f}")
                    
                    with col_res2:
                        st.subheader("Explanation (SHAP)")
                        exp = result.get("explanation", {})
                        if exp:
                            # Render explanation dict simply
                            st.json(exp)
                        else:
                            st.info("No explanation available.")

with tab2:
    st.subheader("Batch Upload Transactions")
    st.markdown("Upload a CSV file with columns: `amount`, `category`, `merchant`. (Optional: `timestamp`, `transaction_id`)")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            if st.button("Analyze Batch"):
                with st.spinner("Analyzing..."):
                    max_rows = 1000
                    if len(df) > max_rows:
                        st.warning(f"File too large. Only analyzing the first {max_rows} rows.")
                        df = df.head(max_rows)
                        
                    results = predict_transaction(df)
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        
                        st.subheader("Results")
                        
                        anomalies = results_df[results_df['is_anomaly'] == True]
                        st.metric("Total Anomalies Found", len(anomalies))
                        
                        # Merge the results back with the original df for display
                        display_df = df.copy()
                        display_df['is_anomaly'] = results_df['is_anomaly']
                        display_df['score'] = results_df['score']
                        
                        st.dataframe(display_df.style.apply(lambda x: ['background: #ffcccc' if x['is_anomaly'] else '' for _ in x], axis=1))
                            
                        st.download_button(
                            label="Download Full Results",
                            data=display_df.to_csv(index=False).encode('utf-8'),
                            file_name='anomaly_results.csv',
                            mime='text/csv',
                        )
        except Exception as e:
            st.error(f"Error reading file: {e}")
