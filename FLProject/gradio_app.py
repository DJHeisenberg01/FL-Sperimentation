import gradio as gr
import json
import subprocess
import threading
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_PATH = "cfg/config.json"
CSV_DIR = "csv"
SERVER_SCRIPT = "start_server.py"
CLIENT_SCRIPT = "start_clients.py"

# --- Utility per gestire config.json ---
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

# --- Funzioni per avviare server/client ---
server_process = None
client_process = None
server_log = []
client_log = []

def run_script(script, args, log_list):
    proc = subprocess.Popen(
        ["python", script] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    for line in proc.stdout:
        log_list.append(line)
    proc.wait()

def start_server():
    global server_process, server_log
    server_log = []
    args = ["--config", CONFIG_PATH]
    server_process = threading.Thread(
        target=run_script,
        args=(SERVER_SCRIPT, args, server_log),
        daemon=True
    )
    server_process.start()
    return "Server avviato!"

def start_clients(num_clients, mode, policy):
    global client_process, client_log
    client_log = []
    # Fallback se num_clients è None
    if num_clients is None:
        num_clients = 4  # default
    if mode is None:
        mode = "standard"
    if policy is None:
        policy = "uniform"
    args = [
        "--config", CONFIG_PATH,
        "--num-clients", str(num_clients),
        "--mode", str(mode),
        "--policy", str(policy)
    ]
    client_process = threading.Thread(
        target=run_script,
        args=(CLIENT_SCRIPT, args, client_log),
        daemon=True
    )
    client_process.start()
    return "Client avviati!"

def stop_all():
    # Non si può killare direttamente i thread, serve kill process se serve
    return "Per fermare i processi, chiudi la webapp o usa Ctrl+C nel terminale."

# --- Funzioni per visualizzare log e risultati ---
def get_server_log():
    return "".join(server_log[-50:])  # Ultime 50 righe

def get_client_log():
    return "".join(client_log[-50:])



def read_metrics():
    metrics = {}
    metrics_dir = "metrics"
    if not os.path.exists(metrics_dir):
        return metrics
    for fname in os.listdir(metrics_dir):
        if fname.endswith(".json"):
            with open(os.path.join(metrics_dir, fname)) as f:
                data = json.load(f)
                metrics[fname] = data
    return metrics

def show_table(metrics):
    rows = []
    for client, rounds in metrics.items():
        for round_num, data in rounds.items():
            rows.append({
                "Client": data.get("client_name", client),
                "Round": round_num,
                "Train Loss": data.get("train_loss", "-"),
                "Avg F1": data.get("avg_f1", "-"),
                "Avg Acc": data.get("avg_acc", "-"),
                "Train Size": data.get("train_size", "-"),
                "CPU": data.get("compute_power", "-"),
                "Bandwidth": data.get("bandwidth", "-"),
                "Reliability": data.get("reliability", "-"),
                "Training Time": data.get("training_time", "-"),
            })
    return pd.DataFrame(rows)

def plot_metrics(metrics):
    plt.figure(figsize=(10,5))
    for client, rounds in metrics.items():
        x = []
        y = []
        for round_num, data in rounds.items():
            x.append(int(round_num))
            y.append(data.get("avg_acc", 0))
        if x and y:
            plt.plot(x, y, label=f"{client} Accuracy")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Client Accuracy per Round")
    plt.tight_layout()
    return plt.gcf()

def read_server_metrics():
    metrics_path = "metrics/server_metrics.json"
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, "r") as f:
        return json.load(f)

def show_server_table():
    data = read_server_metrics()
    rows = []
    for round_num, metrics in data.items():
        row = {"Round": int(round_num)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)

def plot_server_metrics():
    df = show_server_table()
    plt.figure(figsize=(8,4))
    if not df.empty:
        if "test_acc" in df:
            plt.plot(df["Round"], df["test_acc"], label="Test Accuracy")
        if "test_loss" in df:
            plt.plot(df["Round"], df["test_loss"], label="Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.title("Server Metrics")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

# --- Interfaccia Gradio ---
def update_config(
    ip_address, port, model_name, device, global_epoch, local_epoch, num_clients,
    batch_size, learning_rate, client_quality
):
    config = load_config()
    config["ip_address"] = ip_address
    config["port"] = int(port)
    config["model_name"] = model_name
    config["device"] = device
    config["global_epoch"] = int(global_epoch)
    config["local_epoch"] = int(local_epoch)
    config["num_clients"] = int(num_clients)
    config["batch_size"] = int(batch_size)
    config["learning_rate"] = float(learning_rate)
    config["client_configuration"]["num_clients"] = int(num_clients)
    config["client_configuration"]["client_quality"] = client_quality
    config["MIN_NUM_WORKERS"] = int(num_clients)
    save_config(config)
    return "Configurazione aggiornata!"

with gr.Blocks() as demo:
    gr.Markdown("# Federated Learning WebApp")
    gr.Markdown("Personalizza i parametri, avvia server/client e visualizza i risultati!")

    with gr.Tab("Configurazione"):
        config = load_config()
        ip_address = gr.Textbox(value=config["ip_address"], label="IP Server")
        port = gr.Number(value=config["port"], label="Porta Server")
        model_name = gr.Dropdown(["ResNet18", "Altro"], value=config["model_name"], label="Modello")
        device = gr.Dropdown(["cpu", "gpu"], value=config["device"], label="Device")
        global_epoch = gr.Number(value=config["global_epoch"], label="Global Epoch")
        local_epoch = gr.Number(value=config["local_epoch"], label="Local Epoch")
        num_clients = gr.Number(value=config["num_clients"], label="Numero Client")
        batch_size = gr.Number(value=config["batch_size"], label="Batch Size")
        learning_rate = gr.Number(value=config["learning_rate"], label="Learning Rate")
        client_quality = gr.Dropdown(
            config["client_configuration"]["_quality_options"],
            value=config["client_configuration"]["client_quality"],
            label="Profilo Client"
        )
        update_btn = gr.Button("Aggiorna Configurazione")
        update_output = gr.Textbox(label="Stato aggiornamento")
        update_btn.click(
            update_config,
            inputs=[
                ip_address, port, model_name, device, global_epoch, local_epoch,
                num_clients, batch_size, learning_rate, client_quality
            ],
            outputs=update_output
        )

    with gr.Tab("Avvio Server/Client"):
        gr.Markdown("Avvia il server e i client federati")
        server_btn = gr.Button("Avvia Server")
        server_status = gr.Textbox(label="Stato Server")
        server_btn.click(start_server, outputs=server_status)

        client_mode = gr.Dropdown(["standard", "resource"], value="standard", label="Modalità Client")
        client_policy = gr.Dropdown(
            ["uniform", "power", "reliability", "bandwidth", "hybrid"],
            value="uniform", label="Policy Aggregazione"
        )
        client_btn = gr.Button("Avvia Client")
        client_status = gr.Textbox(label="Stato Client")
        client_btn.click(
            start_clients,
            inputs=[num_clients, client_mode, client_policy],
            outputs=client_status
        )

        stop_btn = gr.Button("Stop Tutto")
        stop_status = gr.Textbox(label="Stato Stop")
        stop_btn.click(stop_all, outputs=stop_status)

    with gr.Tab("Log & Output"):
        gr.Markdown("Visualizza log e risultati intermedi/finali")
        server_log_box = gr.Textbox(label="Server Log", lines=10)
        client_log_box = gr.Textbox(label="Client Log", lines=10)
        refresh_btn = gr.Button("Aggiorna Log")
        refresh_btn.click(get_server_log, outputs=server_log_box)
        refresh_btn.click(get_client_log, outputs=client_log_box)

    

    with gr.Tab("Metriche"):
        gr.Markdown("Visualizza le metriche dei client")
        metrics = read_metrics()
        metrics_table = show_table(metrics)
        metrics_output = gr.Dataframe(metrics_table)
        refresh_metrics_btn = gr.Button("Aggiorna Metriche")
        refresh_metrics_btn.click(
            lambda: show_table(read_metrics()),
            outputs=metrics_output
        )

    with gr.Tab("Dashboard"):
        gr.Markdown("# Federated Learning Dashboard")
        gr.Markdown("Monitoraggio live delle metriche dei client")

        refresh_btn = gr.Button("Aggiorna Dashboard")
        
        plot_box = gr.Plot(label="Grafico Accuracy")
        table_box = gr.Dataframe(label="Tabella Metriche")
        metrics_box = gr.Textbox(label="Metriche Raw", lines=10)

        def update_dashboard():
            metrics = read_metrics()
            return json.dumps(metrics, indent=2), plot_metrics(metrics), show_table(metrics)

        refresh_btn.click(update_dashboard, outputs=[metrics_box, plot_box, table_box])

        

    with gr.Tab("Server Training Metrics"):
        gr.Markdown("Metriche di training del server (live da file JSON)")
        server_table = gr.Dataframe(label="Server Metrics")
        server_plot = gr.Plot(label="Server Metrics Plot")
        refresh_server_btn = gr.Button("Aggiorna Server Metrics")

        def refresh_server():
            return show_server_table(), plot_server_metrics()

        refresh_server_btn.click(refresh_server, outputs=[server_table, server_plot])

demo.launch()