    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import mlflow

    # 1. Load data matang dari Colab tadi
    df = pd.read_csv('data_siap_latih.csv')
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 2. Split data (80% untuk latihan, 20% untuk ujian)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Fitur Autolog dari MLflow (Sesuai instruksi Kriteria Basic)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        # 4. Pilih algoritma dan mulai latihan
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        
        print("Pelatihan model standar selesai dan log berhasil dicatat!")