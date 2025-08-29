import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from sklearn.model_selection import StratifiedKFold


class DatasetSplitterClients:

    def __init__(self, CLIENTS_PATHS, IMAGES_PATH, num_clients):
        self.CLIENTS_PATHS = CLIENTS_PATHS
        self.IMAGES_PATH = IMAGES_PATH
        self.num_clients = num_clients
        os.makedirs(self.CLIENTS_PATHS, exist_ok=True)
        data_path = os.path.join(self.IMAGES_PATH, 'roi_annotation.csv')
        self.df = pd.read_csv(data_path)

    def save_images(self, indexes, client_path, dir_name):
        for index in indexes:
            filename_value = self.df.loc[index, 'filename']
            class_value = self.df.loc[index, 'class']

            if class_value == 0:
                dir = "damaged"
                dest_path = os.path.join(client_path, dir_name, dir)
            else:
                dir = "healthy"
                dest_path = os.path.join(client_path, dir_name, dir)

            os.makedirs(dest_path, exist_ok=True)

            # Costruisci il percorso sorgente dell'immagine
            image_path = os.path.join(self.IMAGES_PATH, dir, filename_value)
            dest_file_path = os.path.join(dest_path, filename_value)
            
            # Controlla se il file sorgente esiste prima di copiarlo
            if os.path.exists(image_path):
                try:
                    shutil.copy(image_path, dest_file_path)
                    print(f"Copiato: {filename_value} -> client {os.path.basename(client_path)}/{dir_name}/{dir}")
                except Exception as e:
                    print(f"ERRORE nella copia di {filename_value}: {e}")
            else:
                print(f"AVVISO: File non trovato: {image_path}")
                # Opzionale: cerca il file in tutte le sottodirectory
                found = self.find_file_in_subdirs(self.IMAGES_PATH, filename_value)
                if found:
                    try:
                        shutil.copy(found, dest_file_path)
                        print(f"Trovato e copiato da percorso alternativo: {found} -> {dest_file_path}")
                    except Exception as e:
                        print(f"ERRORE nella copia alternativa di {filename_value}: {e}")

    def find_file_in_subdirs(self, base_path, filename):
        """Cerca un file in tutte le sottodirectory del percorso base"""
        for root, dirs, files in os.walk(base_path):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def split(self):
        self.delete_files()

        # Verifica che il CSV esista e contenga dati
        if self.df.empty:
            raise ValueError(f"Il file CSV {os.path.join(self.IMAGES_PATH, 'roi_annotation.csv')} è vuoto o non valido")

        print(f"Dataset caricato: {len(self.df)} campioni")
        print(f"Classi presenti: {self.df['class'].value_counts().to_dict()}")

        skf = StratifiedKFold(n_splits=self.num_clients, random_state=42, shuffle=True)

        for i, (train_index, test_index) in enumerate(skf.split(self.df['filename'], self.df['class'])):
            client_path = os.path.normpath(os.path.join(self.CLIENTS_PATHS, "client_" + str(i)))
            os.makedirs(client_path, exist_ok=True)

            train, valid = train_test_split(test_index, test_size=0.1, random_state=42)
            
            print(f"\nProcessing Client {i}:")
            print(f"  Train samples: {len(train)}")
            print(f"  Valid samples: {len(valid)}")
            
            self.save_images(train, client_path, "train")
            self.save_images(valid, client_path, "valid")

    def delete_files(self):
        if not os.path.exists(self.CLIENTS_PATHS):
            return
            
        for root, dirs, files in os.walk(self.CLIENTS_PATHS):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"Deleted directory and all its contents: {dir_path}")

    def verify_dataset_structure(self):
        """Verifica la struttura del dataset e stampa informazioni diagnostiche"""
        print("=== VERIFICA STRUTTURA DATASET ===")
        print(f"Directory base: {self.IMAGES_PATH}")
        print(f"File CSV: {os.path.join(self.IMAGES_PATH, 'roi_annotation.csv')}")
        
        # Verifica esistenza directory
        if not os.path.exists(self.IMAGES_PATH):
            print(f"ERRORE: Directory {self.IMAGES_PATH} non esiste!")
            return False
            
        # Verifica esistenza CSV
        csv_path = os.path.join(self.IMAGES_PATH, 'roi_annotation.csv')
        if not os.path.exists(csv_path):
            print(f"ERRORE: File CSV {csv_path} non esiste!")
            return False
            
        # Analizza contenuto CSV
        print(f"Righe nel CSV: {len(self.df)}")
        print(f"Colonne: {list(self.df.columns)}")
        
        # Verifica directory damaged/healthy
        damaged_dir = os.path.join(self.IMAGES_PATH, 'damaged')
        healthy_dir = os.path.join(self.IMAGES_PATH, 'healthy')
        
        print(f"Directory 'damaged' esiste: {os.path.exists(damaged_dir)}")
        print(f"Directory 'healthy' esiste: {os.path.exists(healthy_dir)}")
        
        if os.path.exists(damaged_dir):
            damaged_files = os.listdir(damaged_dir)
            print(f"File in 'damaged': {len(damaged_files)}")
            
        if os.path.exists(healthy_dir):
            healthy_files = os.listdir(healthy_dir)
            print(f"File in 'healthy': {len(healthy_files)}")
            
        # Verifica alcuni file dal CSV
        print("\n=== VERIFICA CAMPIONI FILE ===")
        missing_files = []
        for i, row in self.df.head(10).iterrows():
            filename = row['filename']
            class_val = row['class']
            dir_name = 'damaged' if class_val == 0 else 'healthy'
            full_path = os.path.join(self.IMAGES_PATH, dir_name, filename)
            exists = os.path.exists(full_path)
            print(f"File: {filename}, Classe: {class_val}, Esiste: {exists}")
            if not exists:
                missing_files.append(full_path)
                
        if missing_files:
            print(f"\nATTENZIONE: {len(missing_files)} file non trovati nei primi 10 campioni!")
            
        return True