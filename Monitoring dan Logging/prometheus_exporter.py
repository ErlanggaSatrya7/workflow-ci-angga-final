from prometheus_client import start_http_server, Counter
import time

# 1. Definisikan metrik: ini adalah nama yang akan muncul di Grafana nanti
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total jumlah prediksi model')

if __name__ == '__main__':
    # 2. Buka "pintu" di port 8001 agar Prometheus bisa masuk dan mengambil data
    start_http_server(8001)
    print("Prometheus Exporter sedang berjalan di port 8001...")
    
    # 3. Simulasi: setiap 5 detik kita buat seolah-olah model melakukan prediksi
    while True:
        PREDICTION_COUNTER.inc() # Menambah angka metrik +1
        time.sleep(5)