import os
import numpy as np
from torch.utils import data

class AudioFolder(data.Dataset):
	def __init__(self, root, split, input_length, sr):
		self.root = root
		self.split = split
        self.sr = sr
		self.input_length = input_length
		self.get_songlist()
		self.binary = np.load('./../split/mtat/binary.npy')

	def __getitem__(self, index):
		npy, tag_binary = self.get_npy(index)
		return npy.astype('float32'), tag_binary.astype('float32')

	def get_songlist(self):
		if self.split == 'TRAIN':
			self.fl = np.load('./../split/mtat/train.npy')
		elif self.split == 'VALID':
			self.fl = np.load('./../split/mtat/valid.npy')
		elif self.split == 'TEST':
			self.fl = np.load('./../split/mtat/test.npy')
		else:
			print('Split should be one of [TRAIN, VALID, TEST]')

	def get_npy(self, index):
		ix, fn = self.fl[index].split('\t')
		npy_path = os.path.join(self.root, 'mtat', 'npy', fn.split('/')[1])
		waveform = self.torch_sox_effect_load(npy_path, mmap_mode='r')
		random_idx = int(torch.floor(np.random.random(1) * (waveform.shape[1] - self.input_length)))
		waveform = waveform[:,random_idx:random_idx+self.input_length]
		tag_binary = self.binary[int(ix)]
		return waveform, tag_binary

	def __len__(self):
		return len(self.fl)

    def torch_sox_effect_load(self, mp3_path):
        effects = [
            ['rate', str(self.sr)]
        ]
        waveform, source_sr = torchaudio.load(mp3_path)
        if source_sr != self.sr:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, source_sr, effects, channels_first=True)
        return waveform