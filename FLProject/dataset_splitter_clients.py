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
            # print(type(class_value))

            if class_value == 0:
                dir = "damaged"
                dest_path = os.path.join(client_path, dir_name, dir)

            else:
                dir = "healthy"
                dest_path = os.path.join(client_path, dir_name, dir)

            os.makedirs(dest_path, exist_ok=True)

            image_path = os.path.join(self.IMAGES_PATH, dir, filename_value)
            shutil.copy(image_path, os.path.join(dest_path,filename_value))

    def split(self):

        self.delete_files()

        skf = StratifiedKFold(n_splits=self.num_clients, random_state=42, shuffle=True)

        for i, (train_index, test_index) in enumerate(skf.split(self.df['filename'], self.df['class'])):
            client_path = os.path.join(self.CLIENTS_PATHS, "client_" + str(i))
            os.makedirs(client_path, exist_ok=True)

            train, valid = train_test_split(test_index, test_size=0.1, random_state=42)

            self.save_images(train, client_path, "train")
            self.save_images(valid, client_path, "valid")

    def delete_files(self):
        for root, dirs, files in os.walk(self.CLIENTS_PATHS):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)
                print(f"Deleted directory and all its contents: {dir_path}")