import pickle
from typing import Callable, Optional
import os
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from ..data import AudioDataset

class DataPipeline(LightningDataModule):
    def __init__(self, feature_path, labels, input_length, sr) -> None:
        super(DataPipeline, self).__init__()
        self.feature_path = feature_path
        self.labels = labels
        self.input_length = sr
        self.sr = sr
        if dataset_type == "MTAT":
            self.dataset_builder = AudioDataset
        # elif

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.feature_path,
                self.labels,
                "TRAIN",
                self.input_length, 
                self.sr
            )

            self.val_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.feature_path,
                self.labels,
                "VALID",
                self.input_length, 
                self.sr
            )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.feature_path,
                self.labels,
                "TEST",
                self.input_length, 
                self.sr
            )

    def train_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, feature_path, labels, split, input_length, sr) -> Dataset:
        dataset = dataset_builder(feature_path, labels, split, input_length, sr)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool, **kwargs) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, **kwargs)