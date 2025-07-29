import json
import threading
import time
import itertools
import csv
import os
import federated_server
import run_multiple_clients
from utilities import pickle_file_to_obj

GRID_SEARCH_CFG = './cfg/grid_search_config.json'

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def load_existing_configs(csv_path):
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        existing = []
        for row in reader:
            try:
                existing.append({
                    'model_name': row['model_name'],
                    'global_epoch': float(row['global_epoch']),
                    'models_percentage': float(row['models_percentage']),
                    'local_epoch': float(row['local_epoch']),
                    'batch_size': float(row['batch_size']),
                    'MIN_NUM_WORKERS': float(row['MIN_NUM_WORKERS']),
                    'learning_rate': float(row['learning_rate']),
                })
            except ValueError:
                continue  # Salta righe malformate
        return existing

def is_duplicate(config, existing_configs):
    # Mappa dei nomi alternativi: chiave = nome usato nel check, valore = nome nel CSV
    key_mapping = {
        # 'MIN_NUM_WORKERS': 'MIN_NUM_WORKERS'
    }
    print(config)
    for existing in existing_configs:
        match = True
        for k in config:
            # Usa la chiave mappata se esiste, altrimenti la stessa
            csv_key = key_mapping.get(k, k)
            if k not in config or csv_key not in existing:
                match = False
                break
            if config[k] != existing[csv_key]:
                match = False
                break
        if match:
            return True
    return False

def start_server(config_path):
    server = federated_server.FederatedServer(config_path)
    server.run()

def build_config_file():
    grid_param = load_json(GRID_SEARCH_CFG)

    # Parametri fissi
    fixed_params = {
        'ip_address': grid_param['ip_address'],
        'port': grid_param['port'],
        'csv_output_path': grid_param['csv_output_path'],
        'model_output_path': grid_param['model_output_path'],
        'early_stop_patience': grid_param['early_stop_patience'],
        'weighted_aggregation': grid_param['weighted_aggregation'],
        'device': 'gpu'
    }

    # Search space
    search_space = {
        k: v for k, v in grid_param.items()
        if isinstance(v, list) and len(v) > 1
    }

    # Parametri singoli -> fissi
    fixed_params.update({
        k: v[0] for k, v in grid_param.items()
        if isinstance(v, list) and len(v) == 1
    })

    # Carica configurazioni già eseguite
    existing_configs = load_existing_configs("./csv/training_parameters.csv")

    keys, values = zip(*search_space.items()) if search_space else ([], [])

    for combo in itertools.product(*values):
        config = fixed_params.copy()
        config.update(dict(zip(keys, combo)))
        config['MIN_NUM_WORKERS'] = config['num_clients']

        # Costruisci chiave identificativa per il confronto
        check_config = {
            'model_name': config['model_name'],
            'global_epoch': float(config['global_epoch']),
            'models_percentage': float(config['models_percentage']),
            'MIN_NUM_WORKERS': float(config['MIN_NUM_WORKERS']),
            'local_epoch': float(config['local_epoch']),
            'batch_size': float(config['batch_size']),
            'learning_rate': float(config['learning_rate']),
        }

        if is_duplicate(check_config, existing_configs):
            print(f"Configurazione già eseguita, si salta: {check_config}")
            continue

        # Scrivi config
        config_path = grid_param['config_file']
        with open(config_path, 'w') as fp:
            json.dump(config, fp, indent=4)

        # Avvia server
        thread = threading.Thread(target=start_server, args=(config_path,))
        thread.start()
        time.sleep(5)

        # Avvia client
        run_multiple_clients.main(config_path, grid_param['splitting_dir'])

        thread.join()
        # weights = pickle_file_to_obj("weights.pkl")

        # weights = pickle_file_to_obj("weights.pkl")

if __name__ == '__main__':
    build_config_file()
