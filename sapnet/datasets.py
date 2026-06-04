import glob
import os
import pickle
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _read_cmu_features(label_path: str, feature_root: str) -> Tuple[Dict[str, np.ndarray], int]:
    names = []
    video_ids, *_rest = pickle.load(open(label_path, "rb"), encoding="latin1")
    for vid in video_ids:
        names.extend(video_ids[vid])

    features = []
    feature_dim = -1
    for name in names:
        path = os.path.join(feature_root, f"{name}.npy")
        feature_dir = os.path.join(feature_root, name)
        values = []
        if os.path.exists(path):
            value = np.load(path).squeeze()
            values.append(value)
            feature_dim = max(feature_dim, value.shape[-1])
        elif os.path.isdir(feature_dir):
            for filename in sorted(os.listdir(feature_dir)):
                value = np.load(os.path.join(feature_dir, filename))
                values.append(value)
                feature_dim = max(feature_dim, value.shape[-1])
        else:
            raise FileNotFoundError(f"Missing feature for sample {name}: {path} or {feature_dir}")

        value = np.array(values).squeeze()
        if len(value) == 0:
            value = np.zeros((feature_dim,))
        elif len(value.shape) == 2:
            value = np.mean(value, axis=0)
        features.append(value)

    print(f"Input feature {os.path.basename(feature_root)}: dim={feature_dim}; samples={len(names)}")
    return {name: feature for name, feature in zip(names, features)}, feature_dim


def _read_iemocap_features(label_path: str, feature_root: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    names, speakers = [], []
    video_ids, _labels, video_speakers, _sentences, train_vids, test_vids = pickle.load(
        open(label_path, "rb"), encoding="latin1"
    )
    for vid in sorted(train_vids | test_vids):
        names.extend(video_ids[vid])
        speakers.extend(video_speakers[vid])

    features = []
    feature_dim = -1
    for name, speaker in zip(names, speakers):
        matches = glob.glob(os.path.join(feature_root, f"{name}*"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one feature path for {name}, found {len(matches)}")
        feature_path = matches[0]

        item = {"F": [], "M": []}
        if feature_path.endswith(".npy"):
            value = np.load(feature_path).squeeze()
            item[speaker].append(value)
            feature_dim = max(feature_dim, value.shape[-1])
        else:
            for filename in sorted(os.listdir(feature_path)):
                if "F" not in filename and "M" not in filename:
                    raise ValueError(f"Cannot infer IEMOCAP speaker from feature file: {filename}")
                value = np.load(os.path.join(feature_path, filename))
                feature_dim = max(feature_dim, value.shape[-1])
                item["F" if "F" in filename else "M"].append(value)

        for speaker_id in item:
            value = np.array(item[speaker_id]).squeeze()
            if len(value) == 0:
                value = np.zeros((feature_dim,))
            elif len(value.shape) == 2:
                value = np.mean(value, axis=0)
            item[speaker_id] = value
        features.append(item)

    print(f"Input feature {os.path.basename(feature_root)}: dim={feature_dim}; samples={len(names)}")
    return {name: feature for name, feature in zip(names, features)}, feature_dim


class CmuStyleDataset(Dataset):
    """Utterance-level CMU-MOSI and CMU-MOSEI feature dataset."""

    def __init__(
        self,
        label_path: str,
        audio_root: str,
        text_root: str,
        video_root: str,
        prompt_root: Optional[str] = None,
    ):
        name2audio, self.adim = _read_cmu_features(label_path, audio_root)
        name2text, self.tdim = _read_cmu_features(label_path, text_root)
        name2video, self.vdim = _read_cmu_features(label_path, video_root)
        name2prompt, self.pdim = (None, 0)
        if prompt_root is not None:
            name2prompt, self.pdim = _read_cmu_features(label_path, prompt_root)
        self.use_prompt = name2prompt is not None

        (
            self.video_ids,
            self.video_labels,
            self.video_speakers,
            _video_sentences,
            self.train_vids,
            self.val_vids,
            self.test_vids,
        ) = pickle.load(open(label_path, "rb"), encoding="latin1")

        self.vids = []
        for split_vids in (self.train_vids, self.val_vids, self.test_vids):
            self.vids.extend(sorted(split_vids))

        self.audio_host, self.text_host, self.video_host, self.prompt = {}, {}, {}, {}
        self.audio_guest, self.text_guest, self.video_guest = {}, {}, {}
        self.labels, self.speakers = {}, {}
        self.max_len = 0

        for vid in sorted(self.video_ids):
            uids = self.video_ids[vid]
            labels = self.video_labels[vid]
            speakers = self.video_speakers[vid]
            self.max_len = max(self.max_len, len(uids))
            speaker_map = {"": 0}

            self.audio_host[vid], self.text_host[vid], self.video_host[vid], self.prompt[vid] = [], [], [], []
            self.audio_guest[vid], self.text_guest[vid], self.video_guest[vid] = [], [], []
            self.labels[vid], self.speakers[vid] = [], []

            for idx, uid in enumerate(uids):
                self.audio_host[vid].append(name2audio[uid])
                self.text_host[vid].append(name2text[uid])
                self.video_host[vid].append(name2video[uid])
                if self.use_prompt:
                    self.prompt[vid].append(name2prompt[uid])
                self.audio_guest[vid].append(np.zeros((self.adim,)))
                self.text_guest[vid].append(np.zeros((self.tdim,)))
                self.video_guest[vid].append(np.zeros((self.vdim,)))
                self.labels[vid].append(labels[idx])
                self.speakers[vid].append(speaker_map[speakers[idx]])

            for store in (
                self.audio_host,
                self.text_host,
                self.video_host,
                self.audio_guest,
                self.text_guest,
                self.video_guest,
                self.labels,
                self.speakers,
            ):
                store[vid] = np.array(store[vid])
            if self.use_prompt:
                self.prompt[vid] = np.array(self.prompt[vid])

    def __getitem__(self, index):
        vid = self.vids[index]
        items = [
            torch.FloatTensor(self.audio_host[vid]),
            torch.FloatTensor(self.text_host[vid]),
            torch.FloatTensor(self.video_host[vid]),
            torch.FloatTensor(self.audio_guest[vid]),
            torch.FloatTensor(self.text_guest[vid]),
            torch.FloatTensor(self.video_guest[vid]),
            torch.FloatTensor(self.speakers[vid]),
            torch.FloatTensor([1] * len(self.labels[vid])),
            torch.FloatTensor(self.labels[vid]),
        ]
        if self.use_prompt:
            items.append(torch.FloatTensor(self.prompt[vid]))
        items.append(vid)
        return tuple(items)

    def __len__(self):
        return len(self.vids)

    def feature_dims(self):
        print(f"audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}")
        if self.use_prompt:
            print(f"prompt dimension: {self.pdim}")
            return self.adim, self.tdim, self.vdim, self.pdim
        return self.adim, self.tdim, self.vdim

    def collate_fn(self, data):
        result = []
        frame = pd.DataFrame(data)
        for idx in frame:
            if idx <= 5:
                result.append(pad_sequence(frame[idx]))
            elif idx <= 8:
                result.append(pad_sequence(frame[idx], True))
            elif self.use_prompt and idx == 9:
                result.append(pad_sequence(frame[idx], True))
            else:
                result.append(frame[idx].tolist())
        return result


class IemocapDataset(Dataset):
    """IEMOCAP four-class or six-class feature dataset."""

    def __init__(self, label_path: str, audio_root: str, text_root: str, video_root: str):
        name2audio, self.adim = _read_iemocap_features(label_path, audio_root)
        name2text, self.tdim = _read_iemocap_features(label_path, text_root)
        name2video, self.vdim = _read_iemocap_features(label_path, video_root)

        (
            self.video_ids,
            self.video_labels,
            self.video_speakers,
            _video_sentences,
            self.train_vids,
            self.test_vids,
        ) = pickle.load(open(label_path, "rb"), encoding="latin1")

        self.vids = sorted(list(self.train_vids | self.test_vids))
        self.audio_host, self.text_host, self.video_host = {}, {}, {}
        self.audio_guest, self.text_guest, self.video_guest = {}, {}, {}
        self.labels, self.speakers = {}, {}
        speaker_map = {"F": 0, "M": 1}

        for vid in self.vids:
            self.audio_host[vid], self.text_host[vid], self.video_host[vid] = [], [], []
            self.audio_guest[vid], self.text_guest[vid], self.video_guest[vid] = [], [], []
            self.labels[vid], self.speakers[vid] = [], []
            for idx, uid in enumerate(self.video_ids[vid]):
                self.audio_host[vid].append(name2audio[uid]["F"])
                self.text_host[vid].append(name2text[uid]["F"])
                self.video_host[vid].append(name2video[uid]["F"])
                self.audio_guest[vid].append(name2audio[uid]["M"])
                self.text_guest[vid].append(name2text[uid]["M"])
                self.video_guest[vid].append(name2video[uid]["M"])
                self.labels[vid].append(self.video_labels[vid][idx])
                self.speakers[vid].append(speaker_map[self.video_speakers[vid][idx]])

            for store in (
                self.audio_host,
                self.text_host,
                self.video_host,
                self.audio_guest,
                self.text_guest,
                self.video_guest,
                self.labels,
                self.speakers,
            ):
                store[vid] = np.array(store[vid])

    def __getitem__(self, index):
        vid = self.vids[index]
        return (
            torch.FloatTensor(self.audio_host[vid]),
            torch.FloatTensor(self.text_host[vid]),
            torch.FloatTensor(self.video_host[vid]),
            torch.FloatTensor(self.audio_guest[vid]),
            torch.FloatTensor(self.text_guest[vid]),
            torch.FloatTensor(self.video_guest[vid]),
            torch.FloatTensor(self.speakers[vid]),
            torch.FloatTensor([1] * len(self.labels[vid])),
            torch.LongTensor(self.labels[vid]),
            vid,
        )

    def __len__(self):
        return len(self.vids)

    def feature_dims(self):
        print(f"audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}")
        return self.adim, self.tdim, self.vdim

    def collate_fn(self, data):
        result = []
        frame = pd.DataFrame(data)
        for idx in frame:
            if idx <= 5:
                result.append(pad_sequence(frame[idx]))
            elif idx <= 8:
                result.append(pad_sequence(frame[idx], True))
            else:
                result.append(frame[idx].tolist())
        return result
