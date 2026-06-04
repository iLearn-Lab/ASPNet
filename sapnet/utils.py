import atexit
import os
import random
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import config
from sapnet.datasets import CmuStyleDataset, IemocapDataset
from sapnet.model import PromptFusionNetwork


class Logger:
    def __init__(self, filename="default.log", stream=sys.stdout):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.terminal = stream
        self.log = open(filename, "w", encoding="utf-8")
        self._closed = False
        atexit.register(self.close)

    def write(self, message):
        if self._closed:
            return
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        if self._closed:
            return
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self._closed:
            return
        self.flush()
        self.log.close()
        self._closed = True


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_loaders(
    dataset_name: str,
    audio_root: str,
    text_root: str,
    video_root: str,
    batch_size: int,
    num_workers: int,
    prompt_root: Optional[str] = None,
):
    if dataset_name in config.REGRESSION_DATASETS:
        dataset = CmuStyleDataset(
            label_path=config.PATH_TO_LABEL[dataset_name],
            audio_root=audio_root,
            text_root=text_root,
            video_root=video_root,
            prompt_root=prompt_root,
        )
        train_num = len(dataset.train_vids)
        val_num = len(dataset.val_vids)
        test_num = len(dataset.test_vids)
        splits = [
            (
                list(range(0, train_num)),
                list(range(train_num, train_num + val_num)),
                list(range(train_num + val_num, train_num + val_num + test_num)),
            )
        ]
    elif dataset_name in config.CLASSIFICATION_DATASETS:
        dataset = IemocapDataset(
            label_path=config.PATH_TO_LABEL[dataset_name],
            audio_root=audio_root,
            text_root=text_root,
            video_root=video_root,
        )
        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):
            session = int(vid[4]) - 1
            session_to_idx.setdefault(session, []).append(idx)
        if len(session_to_idx) != 5:
            raise ValueError("IEMOCAP must split into five sessions.")

        splits = []
        for test_session in range(5):
            valid_session = (test_session + 1) % 5
            train_idxs = []
            for session in range(5):
                if session not in (test_session, valid_session):
                    train_idxs.extend(session_to_idx[session])
            splits.append((train_idxs, session_to_idx[valid_session], session_to_idx[test_session]))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loaders, valid_loaders, test_loaders = [], [], []
    for train_idxs, valid_idxs, test_idxs in splits:
        train_loaders.append(_loader(dataset, train_idxs, batch_size, num_workers))
        valid_loaders.append(_loader(dataset, valid_idxs, batch_size, num_workers))
        test_loaders.append(_loader(dataset, test_idxs, batch_size, num_workers))

    return train_loaders, valid_loaders, test_loaders, dataset.feature_dims()


def _loader(dataset, indices, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices),
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_model(args, adim: int, tdim: int, vdim: int, pdim: int = 0):
    model = PromptFusionNetwork(
        args,
        adim,
        tdim,
        vdim,
        args.hidden,
        pdim=pdim,
        n_classes=args.n_classes,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=1,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        no_cuda=args.no_cuda,
    )
    print(f"Model parameters: {sum(x.numel() for x in model.parameters())}")
    return model


def generate_mask(seq_len: int, batch_size: int, condition: str, first_stage: bool):
    if first_stage:
        audio = np.ones(seq_len * batch_size, dtype=np.int64)
        text = np.ones(seq_len * batch_size, dtype=np.int64)
        visual = np.ones(seq_len * batch_size, dtype=np.int64)
    else:
        audio = np.full(seq_len * batch_size, int("a" in condition), dtype=np.int64)
        text = np.full(seq_len * batch_size, int("t" in condition), dtype=np.int64)
        visual = np.full(seq_len * batch_size, int("v" in condition), dtype=np.int64)
    return [audio, text, visual]


def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    host = torch.cat([audio_host, text_host, visual_host], dim=2)
    guest = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    feat_dim = host.size(-1)
    speaker_mask = qmask.transpose(0, 1).unsqueeze(2).repeat(1, 1, feat_dim)
    return [torch.where(speaker_mask == 0, host, guest)]
