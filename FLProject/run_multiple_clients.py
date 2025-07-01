import os
import time
from dataset_splitter_clients import DatasetSplitterClients
from federated_client import FederatedClient
import threading
import json

IMAGES_PATH = 'dataset/Cropped_ROI'
CLIENTS_PATHS = "dataset/Clients"
CONFIG_FILE = 'cfg/config.json'


def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def start_client(config_file, dataset_path, client_id):
    FederatedClient(config_file, dataset_path, client_id)

def main(config_file=CONFIG_FILE, splitting_dir=CLIENTS_PATHS):
    num_clients = load_json(config_file)['num_clients']
    # Split Dataset and Erase old files
    splitter = DatasetSplitterClients(splitting_dir, IMAGES_PATH, num_clients)
    print("Split the Dataset into", num_clients, "Groups")
    splitter.split()

    threads = []
    for i in range(num_clients):
        print(f"Starting client {i} of {num_clients}")
        dataset_path = os.path.join(splitting_dir, 'client_' + str(i))
        thread = threading.Thread(target=start_client, args=(config_file, dataset_path, i))
        thread.start()
        time.sleep(5)
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
