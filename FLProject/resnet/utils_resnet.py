import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split


class ImageNetProcessor:
    def __init__(
        self,
        num_samples=1000,
        calibration_samples=100,
        batch_size=32,
        num_workers=4,
        image_size=224,
        seed=42,
        data_dir=None,  # directory con le immagini divise per classi
    ):
        self.num_samples = num_samples
        self.calibration_samples = calibration_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        if data_dir is None:
            raise ValueError("`data_dir` deve essere specificato per caricare le immagini locali.")

        torch.manual_seed(self.seed)

        # Trasformazioni immagini
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Carica tutte le immagini con etichette da sottocartelle
        full_dataset = datasets.ImageFolder(root=data_dir)

        # Estrai gli indici casuali per main + calibrazione
        total = num_samples + calibration_samples
        indices = list(range(len(full_dataset)))
        if total > len(indices):
            raise ValueError("Numero richiesto di campioni maggiore di quelli disponibili.")

        selected_indices, _ = train_test_split(indices, train_size=total, random_state=seed, shuffle=True)
        main_indices = selected_indices[:num_samples]
        calib_indices = selected_indices[num_samples:num_samples + calibration_samples]

        self.main_dataset = Subset(full_dataset, main_indices)
        self.calibration_dataset = Subset(full_dataset, calib_indices)

        self.dataloader = self._create_dataloader(self.main_dataset)
        self.calibration_dataloader = self._create_dataloader(self.calibration_dataset)

    def preprocess(self, example):
        image, label = example
        return {
            "pixel_values": self.transform(image),
            "cls": label,
        }

    def get_calibration_tensor(self):
        calibration_samples = [self.preprocess(example) for example in self.calibration_dataset]
        calibration_tensor = torch.stack([sample["pixel_values"] for sample in calibration_samples])
        return calibration_tensor

    def _create_dataloader(self, dataset):
        def collate_fn(examples):
            processed = [self.preprocess(example) for example in examples]
            pixel_values = torch.stack([e["pixel_values"] for e in processed])
            labels = torch.tensor([e["cls"] for e in processed])
            return {"pixel_values": pixel_values, "labels": labels}

        generator = torch.Generator().manual_seed(self.seed)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            generator=generator,
            worker_init_fn=lambda worker_id: torch.manual_seed(self.seed + worker_id),
        )

    @staticmethod
    def compute_accuracy(outputs, targets, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = targets.size(0)
            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @classmethod
    def accuracy(cls, outputs, targets):
        return cls.compute_accuracy(outputs, targets, topk=(1,))[0].item()

    @classmethod
    def accuracy_top2(cls, outputs, targets):
        return cls.compute_accuracy(outputs, targets, topk=(2,))[0].item()
