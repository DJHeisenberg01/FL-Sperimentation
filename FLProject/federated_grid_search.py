import json
import threading
import time
import itertools
from pathlib import Path

import federated_server
import run_multiple_clients

GRID_SEARCH_CFG = Path('cfg/grid_search_config.json')


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def start_server(config_path):
    config = load_json(config_path)
    server = federated_server.FederatedServer(config)
    server.run()


def generate_grid_configs(base_config, param_grid):
    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        config = base_config.copy()
        config.update(dict(zip(keys, values)))
        yield config


def write_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def build_and_run():
    grid_param = load_json(GRID_SEARCH_CFG)

    base_config = {
        'ip_address': grid_param['ip_address'],
        'port': grid_param['port'],
        'csv_output_path': grid_param['csv_output_path'],
        'model_output_path': grid_param['model_output_path'],
        'early_stop_patience': grid_param['early_stop_patience'],
        'weighted_aggregation': grid_param['weighted_aggregation'],
        'device': 'gpu'
    }

    param_grid = {
        'model_name': grid_param['model_name'],
        'global_epoch': grid_param['global_epoch'],
        'local_epoch': grid_param['local_epoch'],
        'num_clients': grid_param['num_clients'],
        'models_percentage': grid_param['models_percentage'],
        'learning_rate': grid_param['learning_rate'],
        'batch_size': grid_param['batch_size']
    }

    config_path = Path(grid_param['config_file'])

    for config in generate_grid_configs(base_config, param_grid):
        config['MIN_NUM_WORKERS'] = config['num_clients']

        write_config(config, config_path)

        server_thread = threading.Thread(target=start_server, args=(config_path,))
        server_thread.start()
        time.sleep(2)  # Ensure server has time to start

        run_multiple_clients.main(str(config_path), grid_param['splitting_dir'])

        server_thread.join()


if __name__ == '__main__':
    build_and_run()
