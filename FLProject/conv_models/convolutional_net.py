import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, ResNeXt50_32X4D_Weights, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary
import os
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import rexnet.rexnetv1_lite as rexnet

# HYPERPARAMETERS
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_CLASSES = 2
LEARNING_RATE = 0.001


class ConvolutionalNet:

    def __init__(self, dataset_path, name='ResNet50', device='gpu', ):
        # Declare What type of net want to build between ResNet50, ResNext50 and RexNet
        self.name = name
        if self.name == 'ResNet50':
            print('Loading ResNet50 model...')
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
        elif self.name == 'ResNet18':
            print('Loading ResNet18 model...')
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
        elif self.name == 'ResNet10':
            print('Loading ResNet10 model...')
            self.model = models.resnet10(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
        elif self.name == 'ResNext50':
            print('Loading ResNext50 model...')
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d',
                                        weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
        elif self.name == 'RexNet':
            print('Loading RexNet model...')
            self.model = rexnet.ReXNetV1_lite(multiplier=1.0)
            self.model.load_state_dict(
                torch.load('rexnet/rexnet_lite_1.0.pth',
                           map_location=torch.device('cpu')))
            self.classifier = nn.Sequential(nn.Linear(1000, NUM_CLASSES))
        else:
            raise ValueError('Invalid model name')
        print(device)
        if device == 'gpu' and self.name != 'RexNet':

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.dataset_path = dataset_path
        self.training_transform_pipeline = transforms.Compose([transforms.Resize((224, 224)),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225]),
                                                               ])

    def transform_train(self, image_path, batch_size=32):
        train_dataset = ImageFolder(root=image_path, transform=self.training_transform_pipeline)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader

    def train(self, epochs=NUM_EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(f'Training model {self.name}...')
        time_tot = 0
        train_map = {'f1_score': [], 'accuracy_score': [], 'precision_score': [], 'recall_score': []}
        train_loss = 0
        train_path_loader = os.path.join(self.dataset_path, 'train')
        train_loader = self.transform_train(train_path_loader, batch_size)
        for epoch in range(epochs):
            t = time.time()
            train_loss = self._epoch_train(train_loader, optimizer)
            valid_loss, metric_score , valid_time, valid_size = self.validate(batch_size)
            train_map['f1_score'].append(metric_score['f1_score'])
            train_map['accuracy_score'].append(metric_score['accuracy'])
            train_map['precision_score'].append(metric_score['precision'])
            train_map['recall_score'].append(metric_score['recall'])

            print('-------------------------------------')
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Training time : {time.time() - t:.4f} s')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}')
            print(f'Validation Time : {valid_time:.4f} s')
            print(f'F1Score Validation: {metric_score["f1_score"]:.4f}')
            print(f'Accuracy Validation: {metric_score["accuracy"]:.4f}')
            print(f'Precision Validation: {metric_score["precision"]:.4f}')
            print(f'Recall Validation: {metric_score["recall"]:.4f}')
            print(f'Total Time: {time.time() - t:.4f} s')
            print('--------------------------------------')

            time_tot += time.time() - t
        return time_tot, train_map, train_loss, len(train_loader.dataset)

    def validate(self, batch_size=BATCH_SIZE):
        valid_path = os.path.join(self.dataset_path, "valid")
        valid_loader = self.transform_train(valid_path, batch_size)
        self.model.eval()
        t = time.time()

        predictions = []
        true_values = []
        running_loss = 0
        valid_loss = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                if self.name == 'RexNet':
                    outputs = self.classifier(outputs)

                _, preds = torch.max(outputs, 1)

                true_values.extend(labels.cpu().numpy())
                predictions.extend(preds.cpu().numpy())

                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

            valid_loss = running_loss / len(valid_loader.dataset)
            time_tot = time.time() - t
            f1 = f1_score(predictions, true_values)
            acc = accuracy_score(predictions, true_values)
            prec = precision_score(predictions, true_values)
            recall = recall_score(predictions, true_values, zero_division=True)

            metric_score = {'f1_score': f1, 'accuracy': acc, 'precision': prec, 'recall': recall}

        return valid_loss, metric_score, time_tot, len(valid_loader.dataset)

    def validate_path(self, test_path, batch_size=BATCH_SIZE):
        valid_loader = self.transform_train(test_path, batch_size)
        self.model.eval()
        t = time.time()

        predictions = []
        true_values = []
        running_loss = 0
        valid_loss = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                if self.name == 'RexNet':
                    outputs = self.classifier(outputs)

                _, preds = torch.max(outputs, 1)

                true_values.extend(labels.cpu().numpy())
                predictions.extend(preds.cpu().numpy())

                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

            valid_loss = running_loss / len(valid_loader.dataset)
            time_tot = time.time() - t
            f1 = f1_score(predictions, true_values)
            acc = accuracy_score(predictions, true_values)
            prec = precision_score(predictions, true_values)
            recall = recall_score(predictions, true_values)

            metric_score = {'f1_score': f1, 'accuracy': acc, 'precision': prec, 'recall': recall}

        return valid_loss, metric_score, time_tot, len(valid_loader.dataset)

    def test(self, batch_size=BATCH_SIZE):
        print(f"Testing model {self.name}...")
        valid_loss, metric_score, time_tot, leng = self.validate(batch_size)
        print('----------------------------')
        print(f'Test Time : {time_tot:.4f} s')
        print(f'Test F1 Score : {metric_score["f1_score"]:.4f}')
        print(f'Test Accuracy : {metric_score["accuracy"]:.4f}')
        print('-----------------------------')
        return time_tot, metric_score

    def save_model(self, path):
        save_path = path + f'{self.name}.pth'
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path + f'/{self.name}.pth'))

    def get_weight(self):
        params = [param.data.cpu().numpy()
                  for param in self.model.parameters()]
        return params
        # return copy.deepcopy(self.model.state_dict())

    def set_weight(self, parameters):
        for i, param in enumerate(self.model.parameters()):
            param_ = torch.from_numpy(parameters[i])
            param.data.copy_(param_)
        # self.model.load_state_dict(weights)

    def _epoch_train(self, train_loader, optimizer):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(inputs)
            if self.name == 'RexNet':
                outputs = self.classifier(outputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def _show_model(self):
        print(summary(model=self.model, input_size=(3, 224, 224)))
