import torchvision
from torch.utils.data import Dataset
from pathlib import Path
import re
import numpy as np
from audio_converter import audio_to_melspectrogram
import audioread
import soundfile


class PhonemeSet(Dataset):
    def __init__(self):
        self.mel_specs = []
        self.phonemes = []
        self.phones = []
        with open('cmudict-0.7b', encoding="ISO-8859-1") as f:
            for line in f:
                if not line.startswith(';;;'):
                    print(line)
                    word = line.split('  ')[0].lower()
                    file_path = 'pronunciations/'+word+'.mp3'
                    if Path(file_path).is_file():
                        try:
                            mel_spec = audio_to_melspectrogram(file_path, display=False)
                        except audioread.exceptions.NoBackendError:
                            continue
                        except soundfile.LibsndfileError:
                            continue
                        phoneme = line.split('  ')[1].split()
                        self.mel_specs.append(mel_spec)
                        self.phonemes.append(phoneme)
        for phoneme in self.phonemes:
            for phone in phoneme:
                m = re.match(r'([A-Z]+)\d*', phone)
                if m.group(1) not in self.phones:
                    self.phones.append(m.group(1))
    def __len__(self):
        return len(self.mel_specs)

    def __getitem__(self, idx):
        phonemes = [0 for x in self.phones]
        for phone in self.phonemes[idx]:
            m = re.match(r'([A-Z]+)\d*', phone)
            phonemes[self.phones.index(m.group(1))] = 1
        phonemes = np.array(phonemes)
        return self.mel_specs[idx], phonemes

#model = torchvision.models.vgg19_bn()

