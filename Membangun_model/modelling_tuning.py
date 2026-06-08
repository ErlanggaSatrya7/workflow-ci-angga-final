import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import dagshub

# 1. Koneksi DagsHub
dagshub.init(repo_owner='anggasatrya007', repo_name='telco-churn-mlops', mlflow=True)

# 2. Load data
df = pd.read_csv('data_siap_latih.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Setup MLflow
mlflow.set_experiment("Telco-Churn-Tuning")

# --- KUNCI KRITERIA 2 ---
mlflow.sklearn.autolog()

with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Simpan dataset sebagai bukti tambahan
    mlflow.log_artifact("data_siap_latih.csv")
    
    print("Pelatihan selesai. GitHub Actions akan mengurus upload-nya!")