import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from src.utils.logger import setup_logger
from src.utils.config import settings

logger = setup_logger(__name__)

def generate_transactions(n_samples: int = 10000, anomaly_fraction: float = 0.05) -> pd.DataFrame:
    np.random.seed(42)
    
    categories = ['groceries', 'dining', 'entertainment', 'utilities', 'rent', 'shopping']
    merchants = ['walmart', 'amazon', 'target', 'starbucks', 'uber', 'netflix', 'local_restaurant']
    
    # Generate normal transactions
    n_normal = int(n_samples * (1 - anomaly_fraction))
    n_anomalies = n_samples - n_normal
    
    start_date = datetime.now() - timedelta(days=365)
    
    data = []
    
    # Normal data
    for _ in range(n_normal):
        cat = np.random.choice(categories)
        merch = np.random.choice(merchants)
        amount = np.random.lognormal(mean=3.0, sigma=0.8) # Right skewed
        
        if cat == 'rent':
            amount = np.random.normal(1500, 200)
        elif cat == 'utilities':
            amount = np.random.normal(100, 30)
            
        data.append({
            'transaction_id': f"txn_{np.random.randint(1000000, 9999999)}",
            'timestamp': start_date + timedelta(minutes=np.random.randint(0, 365*24*60)),
            'amount': max(1.0, round(amount, 2)),
            'category': cat,
            'merchant': merch,
            'is_anomaly': 0
        })
        
    # Anomaly data
    for _ in range(n_anomalies):
        cat = np.random.choice(categories)
        merch = np.random.choice(merchants)
        
        if np.random.rand() > 0.5:
            amount = np.random.uniform(5000, 20000) # Unusually high
        else:
            amount = np.random.uniform(0.01, 1.0) # Unusually low
            
        data.append({
            'transaction_id': f"txn_{np.random.randint(1000000, 9999999)}",
            'timestamp': start_date + timedelta(minutes=np.random.randint(0, 365*24*60)),
            'amount': round(amount, 2),
            'category': cat,
            'merchant': merch,
            'is_anomaly': 1
        })
        
    df = pd.DataFrame(data)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    # Give it a proper timestamp format as iso
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    return df

def main():
    logger.info(f"Generating synthetic data...")
    df = generate_transactions()
    
    os.makedirs(os.path.dirname(settings.data_path), exist_ok=True)
    df.to_csv(settings.data_path, index=False)
    logger.info(f"Saved {len(df)} transactions to {settings.data_path}")
    
if __name__ == "__main__":
    main()
