Per eseguire la grid search si deve modificare il file grid_search_config.json nella cartella cfg per modificare i valori per ogni parametro, per poi eseguire lo script "federated_grid_search.py" che si occuperà automaticamente a creare i client, il server, split del dataset ecc.

Per eseguire invece la rete federata come un client server con parametri statici si possono tranquillamente eseguire gli script "federated_client" e "federated_server" dopo aver configurato i parametri nel file "config.json" nella cartella cfg.

------

Versioni Funzionanti:
Python 3.11
Per installare correttamente cuda e pytorch:
pip install -r requirements.txt

pip uninstall torch torchvideo torchaudio

Python Version: 3.11
Pytorch version: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
CUDA Version: 12.6 https://developer.nvidia.com/cuda-12-6-0-download-archive)