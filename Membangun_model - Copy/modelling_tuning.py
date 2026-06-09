import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub

# 1. Koneksikan ke DagsHub (GANTI DENGAN MILIKMU)
dagshub.init(repo_owner='anggasatrya007', repo_name='telco-churn-mlops', mlflow=True)

# 2. Load data
df = pd.read_csv('data_siap_latih.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Buat eksperimen di MLflow
mlflow.set_experiment("Telco-Churn-Tuning")

with mlflow.start_run() as run:
    # 1. Training model
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # 2. Persiapkan file-file wajib secara lokal agar bisa di-upload
    # Pastikan file requirements.txt ada di direktori kerja Anda
    
    # 3. Log model dengan struktur folder 'model'
    # Parameter artifact_path='model' akan membuat folder fisik bernama 'model'
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="model",
        registered_model_name="TelcoChurnModel"
    )

    # 4. Log file pendukung lainnya ke dalam folder 'model' agar masuk ke dalam satu folder
    # Ini memastikan MLmodel, conda.yaml, dsb. berada di tempat yang benar
    # Catatan: MLflow secara otomatis memasukkan MLmodel, conda.yaml, python_env.yaml
    # saat log_model dipanggil.
    
    # Jika requirements.txt belum masuk otomatis, log manual ke folder 'model':
    mlflow.log_artifact("requirements.txt", artifact_path="model")
    mlflow.log_artifact("data_siap_latih.csv")
    mlflow.log_artifact("confusion_matrix.png")