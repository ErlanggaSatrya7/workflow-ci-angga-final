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

with mlflow.start_run():
    # 4. Hyperparameter Tuning (Mencari parameter terbaik)
    rf = RandomForestClassifier(random_state=42)
    params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    
    grid = GridSearchCV(rf, params, cv=3)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    
    # 5. Manual Logging (Sesuai instruksi Kriteria Advanced)
    mlflow.log_param("best_n_estimators", grid.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid.best_params_['max_depth'])
    
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    
    # 6. Log Artifact (Menyimpan file tambahan)
    # Artefak 1: File Model
    mlflow.sklearn.log_model(best_model, "model")
    
    # Artefak 2: Gambar Confusion Matrix
    disp = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Artefak 3: Data Latih
    mlflow.log_artifact("data_siap_latih.csv")
    
    print("Pelatihan Advanced selesai! Cek hasilnya di website DagsHub kamu.")