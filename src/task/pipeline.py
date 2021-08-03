import pickle
from typing import Callable, Optional
import os
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from ..data import MTATDataset, MSDDataset, JamendoDataset

class DataPipeline(LightningDataModule):
    def __init__(self, dataset_type, audio_path, split_path, sr, duration, batch_size, num_workers) -> None:
        super(DataPipeline, self).__init__()
        self.audio_path = audio_path
        self.split_path = split_path
        self.sr = sr
        self.duration =duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        if dataset_type == "mtat":
            self.dataset_builder = MTATDataset
        elif dataset_type == "msd":
            self.dataset_builder = MSDDataset
        elif dataset_type == "jamendo":
            self.dataset_builder = JamendoDataset

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.audio_path,
                self.split_path,
                "TRAIN",
                self.sr, 
                self.duration
            )

            self.val_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.audio_path,
                self.split_path,
                "VALID",
                self.sr, 
                self.duration
            )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.audio_path,
                self.split_path,
                "TEST",
                self.sr, 
                self.duration
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
    def get_dataset(cls, dataset_builder: Callable, audio_path, split_path, split, sr, duration) -> Dataset:
        dataset = dataset_builder(audio_path, split_path, split, sr, duration)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool, **kwargs) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, **kwargs)