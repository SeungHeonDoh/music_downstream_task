import os
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils import data
import torch
from .utils import torch_sox_effect_load


class MTATDataset(data.Dataset):
	def __init__(self, audio_path, split_path, split, sr, duration):
		self.audio_path = audio_path
		self.split_path = split_path
		self.split = split
		self.sr =sr
		self.input_length = sr * duration
		self.get_fl()

	def __getitem__(self, index):
		instance = self.fl.iloc[index]
		binary = instance.values
		fpath = instance.name
		tensor_audio = torch_sox_effect_load(Path(self.audio_path, fpath), self.sr)
		audio_input = self.get_audio(tensor_audio)
		return audio_input.to(dtype=torch.float32), binary.astype("float32")
		
	def get_fl(self):
		if self.split == "TRAIN":
			self.fl = pd.read_csv(Path(self.split_path, "train.csv"), index_col=0)
		elif self.split == "VALID":
			self.fl = pd.read_csv(Path(self.split_path, "valid.csv"), index_col=0)
		elif self.split == "TEST":
			self.fl = pd.read_csv(Path(self.split_path, "test.csv"), index_col=0)
		else:
			print("Split should be one of [TRAIN, VALID, TEST]")

	def get_audio(self, tensor_audio):
		mono_sample = tensor_audio.mean(0, False)
		if self.split == "TEST":
			audio_split = mono_sample.split(self.input_length)
			audio_input = torch.stack(list(audio_split)[:-1])
		else:
			random_idx = random.choice(range(len(mono_sample) - self.input_length))
			audio_input = mono_sample[random_idx:random_idx+self.input_length]
		return audio_input

	def __len__(self):
		return len(self.fl)

class JamendoDataset(data.Dataset):
	def __init__(self, audio_path, split_path, split, sr, duration):
		self.audio_path = audio_path
		self.split_path = split_path
		self.split = split
		self.sr =sr
		self.input_length = sr * duration
		self.track_to_path = pickle.load(open(Path(self.split_path, "track_to_path.pkl"),'rb'))
		self.get_fl()

	def __getitem__(self, index):
		instance = self.fl.iloc[index]
		binary = instance.values
		fname = instance.name
		fpath = self.track_to_path[fname]
		tensor_audio = torch_sox_effect_load(Path(audio_path, fpath), self.sr)
		audio_input = self.get_audio(tensor_audio)
		return audio_input.to(dtype=torch.float32), binary.astype("float32")
		
	def get_fl(self):
		if self.split == "TRAIN":
			self.fl = pd.read_csv(Path(self.split_path, "train.csv"), index_col=0)
		elif self.split == "VALID":
			self.fl = pd.read_csv(Path(self.split_path, "valid.csv"), index_col=0)
		elif self.split == "TEST":
			self.fl = pd.read_csv(Path(self.split_path, "test.csv"), index_col=0)
		else:
			print("Split should be one of [TRAIN, VALID, TEST]")

	def get_audio(self, tensor_audio):
		mono_sample = tensor_audio.mean(0, False)
		if self.split == "TEST":
			audio_split = mono_sample.split(self.input_length)
			audio_input = torch.stack(list(audio_split)[:-1])
		else:
			random_idx = random.choice(range(len(mono_sample) - self.input_length))
			audio_input = mono_sample[random_idx:random_idx+self.input_length]
		return audio_input

	def __len__(self):
		return len(self.fl)

class MSDDataset(data.Dataset):
	def __init__(self, audio_path, split_path, split, sr, duration):
		self.audio_path = audio_path
		self.split_path = split_path
		self.split = split
		self.sr =sr
		self.input_length = sr * duration
		self.track_to_path = pickle.load(open(Path(self.split_path, "track_to_path.pkl"),'rb'))
		self.get_fl()

	def __getitem__(self, index):
		instance = self.fl.iloc[index]
		binary = instance.values
		fname = instance.name
		fpath = self.track_to_path[fname]
		tensor_audio = torch_sox_effect_load(Path(audio_path, fpath), self.sr)
		audio_input = self.get_audio(tensor_audio)
		return audio_input.to(dtype=torch.float32), binary.astype("float32")
		
	def get_fl(self):
		if self.split == "TRAIN":
			self.fl = pd.read_csv(Path(self.split_path, "train.csv"), index_col=0)
		elif self.split == "VALID":
			self.fl = pd.read_csv(Path(self.split_path, "valid.csv"), index_col=0)
		elif self.split == "TEST":
			self.fl = pd.read_csv(Path(self.split_path, "test.csv"), index_col=0)
		else:
			print("Split should be one of [TRAIN, VALID, TEST]")

	def get_audio(self, tensor_audio):
		mono_sample = tensor_audio.mean(0, False)
		if self.split == "TEST":
			audio_split = mono_sample.split(self.input_length)
			audio_input = torch.stack(list(audio_split)[:-1])
		else:
			random_idx = random.choice(range(len(mono_sample) - self.input_length))
			audio_input = mono_sample[random_idx:random_idx+self.input_length]
		return audio_input

	def __len__(self):
		return len(self.fl)