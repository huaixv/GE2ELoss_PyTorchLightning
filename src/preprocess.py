from collections import defaultdict

import librosa
import torch
from torchaudio.datasets.utils import walk_files,makedir_exist_ok

from src.config import *


def vad(X, sr):
    intervals = librosa.effects.split(X, top_db=TOP_DB, frame_length=WIN_LEN, hop_length=WIN_HOP)
    return [X[x:y] for (x, y) in intervals]


def logmel(X, sr):
    mel = librosa.feature.melspectrogram(X, sr, n_fft=NFFT, win_length=WIN_LEN, hop_length=WIN_HOP, n_mels=NUM_MELS)
    mel = torch.log(torch.tensor(mel + 1e-6))
    return mel


def utt2mel(X, sr):
    frags = vad(X, sr)
    mels = []
    for frag in frags:
        if len(frag) > WIN_LEN:
            mels.append(logmel(frag, sr))
    mels = torch.cat(mels, dim=1)
    mels = mels.transpose(0, 1)  # (n_frames, n_mels)
    return mels


def load_timit_item(path):
    X, sr = librosa.core.load(path, sr=None)
    mels = utt2mel(X, sr)
    mels = mels[0:NUM_FRAMES, :]
    label = path.split('/')[-2]
    return mels, label


def preprocess(root):
    files = walk_files(root, ('.WAV', '.wav'), prefix=True)
    ddict = defaultdict(list)
    for path in files:
        mels, label = load_timit_item(path)
        if (mels.size()[0] >= NUM_FRAMES):
            ddict[label].append(mels)

    X = list(ddict.values())
    X = [torch.stack(x) for x in X]
    y = list(range(len(X)))
    return X, y


if __name__ == '__main__':
    TRAIN_ROOT = "TIMIT/TRAIN"
    TEST_ROOT = "TIMIT/TEST"

    X_train, y_train = preprocess(TRAIN_ROOT)
    makedir_exist_ok("data/train")
    torch.save(X_train, "data/train/X.pt")
    torch.save(y_train, "data/train/y.pt")

    X_test, y_test = preprocess(TEST_ROOT)
    makedir_exist_ok("data/test")
    torch.save(X_test, "data/test/X.pt")
    torch.save(y_test, "data/test/y.pt")
