import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from aspnet.modalities import normalize_condition


PROMPT_FILE_NAMES = {
    "a": "audio",
    "t": "text",
    "v": "visual",
    "at": "audio_text",
    "av": "audio_visual",
    "tv": "text_visual",
    "atv": "audio_text_visual",
}


def iter_label_sample_ids(dataset: str) -> Iterable[str]:
    label_data = pickle.load(open(config.PATH_TO_LABEL[dataset], "rb"), encoding="latin1")
    if len(label_data) == 7:
        video_ids, _, _, _, train_vids, val_vids, test_vids = label_data
    elif len(label_data) == 6:
        video_ids, _, _, _, train_vids, test_vids = label_data
        val_vids = set()
    else:
        raise ValueError(f"Unsupported label file format: {config.PATH_TO_LABEL[dataset]}")
    for vid in sorted(train_vids | val_vids | test_vids):
        for sample_id in video_ids[vid]:
            yield sample_id


def load_prompt_files(prompt_dir: str, conditions: List[str]) -> Dict[str, str]:
    prompts = {}
    for split in ("train", "valid", "test"):
        for condition in conditions:
            path = os.path.join(prompt_dir, f"{split}_{PROMPT_FILE_NAMES[condition]}_fixed.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing fixed prompt file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    sample_id = item["sample_id"]
                    prompt = item.get("final_answer") or ""
                    prompts[sample_id] = prompt.strip()
    return prompts


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def cls_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0]


def parse_args():
    parser = argparse.ArgumentParser(description="Encode generated semantic prompts into utterance-level feature files.")
    parser.add_argument("--dataset", default="CMU-MOSI", choices=["CMU-MOSI", "CMU-MOSEI", "IEMOCAP"])
    parser.add_argument("--iemocap-classes", type=int, choices=[4, 6], default=4)
    parser.add_argument("--prompt-dir", default="prompt_outputs")
    parser.add_argument("--conditions", nargs="+", default=["atv"], choices=list(PROMPT_FILE_NAMES))
    parser.add_argument("--model-path", required=True, help="Local HuggingFace text encoder directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output feature directory. If one condition is requested, features are written here. "
            "If multiple conditions are requested, condition suffixes are appended. "
            "Default: dataset/<dataset>/features/prompt-deberta-large-4-UTT-<condition>"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def condition_output_dir(args, condition: str) -> str:
    base_dir = args.output_dir or os.path.join(
        config.PATH_TO_FEATURES[args.dataset], "prompt-deberta-large-4-UTT"
    )
    if args.output_dir is not None and len(args.conditions) == 1:
        return base_dir
    return f"{base_dir}-{condition}"


def main():
    args = parse_args()
    args.dataset = config.normalize_dataset_name(args.dataset, args.iemocap_classes)
    args.conditions = [normalize_condition(condition) for condition in args.conditions]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True).to(device)
    model.eval()

    sample_ids = list(iter_label_sample_ids(args.dataset))
    pool_fn = mean_pool if args.pooling == "mean" else cls_pool

    for condition in args.conditions:
        output_dir = condition_output_dir(args, condition)
        os.makedirs(output_dir, exist_ok=True)
        prompts = load_prompt_files(args.prompt_dir, [condition])
        work_items = []
        for sample_id in sample_ids:
            out_path = os.path.join(output_dir, f"{sample_id}.npy")
            if os.path.exists(out_path) and not args.overwrite:
                continue
            prompt = prompts.get(sample_id)
            if not prompt:
                raise KeyError(f"Missing prompt for sample_id={sample_id} condition={condition}")
            work_items.append((sample_id, prompt))

        print(
            f"condition={condition}; dataset={args.dataset}; samples={len(sample_ids)}; "
            f"to_write={len(work_items)}"
        )
        print(f"output_dir={output_dir}")

        with torch.no_grad():
            for start in range(0, len(work_items), args.batch_size):
                batch = work_items[start : start + args.batch_size]
                sample_batch = [x[0] for x in batch]
                text_batch = [x[1] for x in batch]
                encoded = tokenizer(
                    text_batch,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = model(**encoded)
                feats = pool_fn(output.last_hidden_state, encoded["attention_mask"]).detach().cpu().numpy()
                feats = feats.astype(np.float32)
                for sample_id, feat in zip(sample_batch, feats):
                    np.save(os.path.join(output_dir, f"{sample_id}.npy"), feat)
                print(f"[{condition}] wrote {min(start + len(batch), len(work_items))}/{len(work_items)}")


if __name__ == "__main__":
    main()
