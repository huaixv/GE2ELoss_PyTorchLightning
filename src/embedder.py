import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from src.config import *
from src.dataset import TIMITP


class Embedder(LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(NUM_MELS, NUM_HIDDEN_NODES, NUM_LAYERS)
        self.fc = torch.nn.Linear(NUM_HIDDEN_NODES, NUM_PROJ)
        self.w = torch.nn.Parameter(torch.tensor([10.]), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor([-5.]), requires_grad=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=SGD_LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SGD_EPOCH_HALF, SGD_LR_GAMMA)
        # optimizer = torch.optim.Adam(self.parameters(), lr=ADAM_LR)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ADAM_EPOCH_HALF, ADAM_LR_GAMMA)
        return [optimizer], [scheduler]

    # ----------------------------------------
    # Embedding network
    # ----------------------------------------
    def forward(self, X):
        # Reshape input:
        # (n_spk, n_utt, n_frame, n_mels) -> (ttl_utt, n_frame, n_mels)
        size = list(X.size())
        X = X.reshape([-1] + size[-2:])
        # LSTM
        E, _ = self.lstm(X)
        E = E[:, -1]
        # Linear
        E = self.fc(E)
        # Normalize
        E = F.normalize(E, p=2, dim=-1)
        # Reshape output:
        E = E.reshape(size[0:-2] + [-1])
        return E

    # ----------------------------------------
    # Centroids && Similarity matrix
    # ----------------------------------------
    def cent(self, E):
        n_spk, n_utt, n_dim = E.size()
        C = E.mean(dim=1)
        C = F.normalize(C, p=2, dim=-1)
        return C

    def simmat(self, E, C=None):
        # We use broadcasting to calculate similarity matrix
        n_spk, n_utt, n_dim = E.size()
        # Mean, normalize, reshape
        if C is None:
            C = self.cent(E)
        # Reshape
        E = E.reshape([n_spk, n_utt, 1, n_dim])
        C = C.reshape([1, 1, n_spk, n_dim])
        # Cosine similarity
        S = F.cosine_similarity(E, C, dim=-1)
        return S

    # ----------------------------------------
    # Loss function
    # ----------------------------------------
    def softmax(self, S):
        idx = list(range(S.size(0)))
        pos = S[idx, :, idx]
        neg = (torch.exp(S).sum(dim=2) + 1e-6).log_()
        L = -1 * (pos - neg)
        l = L.sum()
        return l, L

    def ge2eloss(self, E):
        torch.clamp(self.w, 1e-6)
        S = self.simmat(E)
        Sp = self.w * S + self.b

        # Loss
        l, _ = self.softmax(Sp)
        return l, S

    # ----------------------------------------
    # ArcFace Loss
    # ----------------------------------------
    def arcfaceloss(self, E):
        torch.clamp(self.w, 1e-6)

        S = self.simmat(E)
        I = list(range(S.size(0)))

        # ArcSoftmax loss
        pos = S[I, :, I]
        pos = torch.acos(pos)
        pos += LMS_M2
        pos = torch.cos(pos)

        S[I, :, I] = pos
        S = self.w * S + self.b

        pos = S[I, :, I]
        neg = (S.exp().sum(dim=2) + 1e-6).log()
        L = neg - pos
        l = L.sum()
        return l, S

    # A simple loss wrapper
    def loss(self, X):
        return self.ge2eloss(X)

    # ----------------------------------------
    # Train
    # ----------------------------------------
    def train_dataloader(self):
        return DataLoader(TIMITP("data/train", TRAIN_UTTS), batch_size=TRAIN_SPKS, num_workers=0)

    def training_step(self, batch, batch_idx):
        X, y = batch
        E = self(X)
        L, _ = self.loss(E)
        log = {
            'train_loss': L
        }
        return {'loss': L, 'log': log}

    # ----------------------------------------
    # Validation
    # ----------------------------------------
    def val_dataloader(self):
        return DataLoader(TIMITP("data/test", TEST_UTTS), batch_size=TEST_SPKS, num_workers=0)

    def validation_step(self, batch, batch_idx):
        """
        Calculate validation loss and EER etc.
        Adapted from https://github.com/HarryVolek/PyTorch_Speaker_Verification/blob/master/train_speech_embedder.py
        """
        X, y = batch
        E = self(X)
        L, S = self.loss(E)

        # Randomize
        size = E.size()
        perm = torch.randperm(size[1])
        Ee = E[:, perm[:TEST_ENROLL], :]
        Ev = E[:, perm[TEST_ENROLL:], :]

        Ce = self.cent(Ee)
        S = self.simmat(Ev, Ce)

        y = torch.zeros(S.shape).to(S.device)
        idx = list(range(S.size(0)))
        y[idx, :, idx] = True

        diff = torch.tensor(1.).to(S.device)
        EER = torch.tensor(1.).to(S.device)
        EER_thresh = torch.tensor(1.).to(S.device)
        EER_FAR = torch.tensor(1.).to(S.device)
        EER_FRR = torch.tensor(1.).to(S.device)
        for th in [0.01 * i for i in range(100)]:
            y_pred = S > th

            fp = (y_pred == 1) & (y == 0)
            fptn = y == 0

            fn = (y_pred == 0) & (y == 1)
            fntp = y == 1
            FAR = fp.sum().float() / fptn.sum().float()
            FRR = fn.sum().float() / fntp.sum().float()

            # Save threshold when FAR = FRR (=EER)
            if diff > abs(FAR - FRR):
                diff = abs(FAR - FRR)
                EER = (FAR + FRR) / 2
                EER_thresh = torch.tensor(th).to(S.device)
                EER_FAR = FAR
                EER_FRR = FRR

        return {
            'val_loss': L,
            'eer': EER,
            'thres': EER_thresh,
            'far': EER_FAR,
            'frr': EER_FRR
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_eer = torch.stack([x['eer'] for x in outputs]).mean()
        avg_thres = torch.stack([x['thres'] for x in outputs]).mean()
        avg_far = torch.stack([x['far'] for x in outputs]).mean()
        avg_frr = torch.stack([x['frr'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss,
                            'val_err': avg_eer,
                            'val_thres': avg_thres,
                            'val_far': avg_far,
                            'val_frr': avg_frr}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':
    net = Embedder()

    if RESUME:
        trainer = Trainer(gpus=1, resume_from_checkpoint=CKPT_PATH, max_epochs=NUM_EPOCH, gradient_clip_val=GRAD_CLIP)
    else:
        trainer = Trainer(gpus=1, max_epochs=NUM_EPOCH, gradient_clip_val=GRAD_CLIP)

    trainer.fit(net)
    # trainer.save_checkpoint(CKPT_PATH)
