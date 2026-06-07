from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from aspnet.modalities import CONDITION_TO_MODALITIES, normalize_condition


ALL_MODALITIES: Tuple[str, ...] = ("text", "audio", "visual")


STAGE_PROMPT_TEMPLATE = """
You are a multimodal sentiment analysis expert. Generate an evidence-grounded affective semantic prompt for downstream incomplete multimodal sentiment analysis.

Rules:
1. Use only the modalities explicitly marked as available.
2. Do not describe missing modalities as observed evidence.
3. Do not infer exact wording from non-text modalities.
4. Return only the requested XML-style section and no Markdown.

Stage to generate: {stage}

Stage definitions:
- STATE: identify available/missing modalities and define the MSA analysis objective.
- EVIDENCE: extract sentiment-relevant evidence only from available modalities.
- INFERENCE: Explain how observed cues correspond, agree, differ, or remain ambiguous across modalities in a step-by-step thought process.
- PROMPT: write the final compact semantic prompt for downstream MSA feature extraction.

Previous generated sections:
{previous_sections}

Available modality code: {available_modality_codes}
Available modalities: {available_modalities}
Missing modalities: {missing_modalities}

Text evidence:
{text_input}

Audio evidence:
{audio_input}

Visual evidence:
{visual_input}

Media access:
{media_information}

Return format:
<{stage}>
[content]
</{stage}>
""".strip()


JUDGE_PROMPT_TEMPLATE = """
You are selecting the best semantic prompt candidate for incomplete multimodal sentiment analysis.

Available modalities: {available_modalities}
Missing modalities: {missing_modalities}

Selection criteria:
1. The candidate must obey the missing-modality constraint.
2. It must not fabricate unavailable audio, visual, or textual evidence.
3. It should contain clear affective information useful for downstream MSA.
4. It should be concise and suitable for text feature extraction.

Candidates:
{candidates}

Return only the integer index of the best candidate.
""".strip()


STAGES: Tuple[str, ...] = ("STATE", "EVIDENCE", "INFERENCE", "PROMPT")
STAGE_KEYS = {
    "STATE": "state",
    "EVIDENCE": "evidence",
    "INFERENCE": "inference",
    "PROMPT": "prompt",
}
STAGE_END_TAGS = {stage: f"</{stage}>" for stage in STAGES}


@dataclass
class MediaPackage:
    content_items: List[Dict[str, Any]]
    use_audio_in_video: bool
    access_mode: str
    source_paths: List[str]

    @property
    def has_media(self) -> bool:
        return bool(self.content_items)


def iter_split_samples(dataset: str, split: str) -> Iterator[Dict[str, Any]]:
    label_path = config.PATH_TO_LABEL[dataset]
    with open(label_path, "rb") as f:
        label_data = pickle.load(f, encoding="latin1")

    if len(label_data) == 7:
        video_ids, _labels, video_speakers, video_sentences, train_vids, val_vids, test_vids = label_data
    elif len(label_data) == 6:
        video_ids, _labels, video_speakers, video_sentences, train_vids, test_vids = label_data
        val_vids = set()
    else:
        raise ValueError(f"Unsupported label file format: {label_path}")

    split_to_vids = {
        "train": sorted(train_vids),
        "valid": sorted(val_vids),
        "test": sorted(test_vids),
    }
    for vid in split_to_vids[split]:
        uids = video_ids[vid]
        sentences = video_sentences[vid]
        speakers = video_speakers[vid]
        for idx, uid in enumerate(uids):
            yield {
                "sample_id": str(uid),
                "video_id": str(vid),
                "clip_id": str(uid).rsplit("_", 1)[-1],
                "text": str(sentences[idx]),
                "speaker": speakers[idx],
            }


def resolve_video_path(raw_data_dir: str, sample: Mapping[str, Any]) -> Path:
    root = Path(raw_data_dir)
    candidates = [
        root / str(sample["video_id"]) / f'{sample["clip_id"]}.mp4',
        root / str(sample["video_id"]) / f'{sample["sample_id"]}.mp4',
        root / f'{sample["sample_id"]}.mp4',
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def extract_audio_to_wav(video_path: Path, output_path: Path) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required but was not found in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to extract audio from {video_path}: {exc.stderr.strip()}") from exc
    return output_path


def prepare_media_package(
    sample: Mapping[str, Any],
    condition: str,
    raw_data_dir: str,
    audio_cache_dir: str,
    fps: float = 2.0,
    max_frames: int = 16,
) -> MediaPackage:
    available = set(CONDITION_TO_MODALITIES[condition])
    video_path = resolve_video_path(raw_data_dir, sample)
    if not video_path.exists():
        raise FileNotFoundError(f"Raw video not found for sample {sample['sample_id']}: {video_path}")

    has_audio = "audio" in available
    has_visual = "visual" in available
    if has_audio and has_visual:
        return MediaPackage(
            [{"type": "video", "video": str(video_path), "fps": fps, "max_frames": max_frames}],
            True,
            "video_with_audio",
            [str(video_path)],
        )
    if has_audio:
        wav_path = Path(audio_cache_dir) / f'{sample["sample_id"]}.wav'
        return MediaPackage(
            [{"type": "audio", "audio": str(extract_audio_to_wav(video_path, wav_path))}],
            False,
            "isolated_audio_only",
            [str(wav_path)],
        )
    if has_visual:
        return MediaPackage(
            [{"type": "video", "video": str(video_path), "fps": fps, "max_frames": max_frames}],
            False,
            "silent_video_frames_only",
            [str(video_path)],
        )
    return MediaPackage([], False, "text_only", [])


def extract_tagged_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for tag in STAGES:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            sections[STAGE_KEYS[tag]] = match.group(1).strip()
    return sections


def media_instruction(package: MediaPackage) -> str:
    if package.access_mode == "text_only":
        return "No raw media is provided because only text is available."
    if package.access_mode == "video_with_audio":
        return "A video with audio is provided; use both vocal and visual evidence."
    if package.access_mode == "isolated_audio_only":
        return "Isolated audio is provided; use audio evidence only."
    if package.access_mode == "silent_video_frames_only":
        return "Silent video frames are provided; use visual evidence only."
    return package.access_mode


def modality_codes(available: Sequence[str]) -> str:
    code_map = {"audio": "a", "text": "t", "visual": "v"}
    return "".join(code_map[m] for m in ALL_MODALITIES if m in available)


def format_previous_sections(sections: Mapping[str, str]) -> str:
    if not sections:
        return "None."
    lines = []
    for stage in STAGES:
        key = STAGE_KEYS[stage]
        if key in sections and sections[key].strip():
            lines.append(f"<{stage}>\n{sections[key].strip()}\n</{stage}>")
    return "\n\n".join(lines) if lines else "None."


def build_stage_prompt(
    sample: Mapping[str, Any],
    condition: str,
    media: MediaPackage,
    stage: str,
    previous_sections: Mapping[str, str],
) -> str:
    available = list(CONDITION_TO_MODALITIES[condition])
    missing = [m for m in ALL_MODALITIES if m not in available]

    if "text" in available:
        text_information = f'Observed text: "{sample["text"]}"; speaker: {sample.get("speaker") or "unknown"}.'
    else:
        text_information = "MISSING"

    if "audio" in available:
        audio_information = "Audio is available. Describe prosody, pitch, energy, pauses, laughter, rhythm, and vocal attitude."
    else:
        audio_information = "MISSING"

    visual_information = (
        "Visual input is available. Describe facial expressions, eye contact, posture, head motion, and gestures."
        if "visual" in available
        else "MISSING"
    )

    return STAGE_PROMPT_TEMPLATE.format(
        stage=stage,
        previous_sections=format_previous_sections(previous_sections),
        available_modality_codes=modality_codes(available),
        available_modalities=", ".join(available),
        missing_modalities=", ".join(missing) or "none",
        text_input=text_information,
        audio_input=audio_information,
        visual_input=visual_information,
        media_information=media_instruction(media),
    )


def parse_stage_response(raw_text: str, stage: str) -> str:
    raw_text = trim_after_stage(raw_text, stage)
    sections = extract_tagged_sections(raw_text)
    key = STAGE_KEYS[stage]
    if sections.get(key):
        return sections[key].strip()
    return raw_text.strip().replace("\n", " ")


def trim_after_stage(raw_text: str, stage: str) -> str:
    end_tag = STAGE_END_TAGS[stage]
    match = re.search(re.escape(end_tag), raw_text, flags=re.IGNORECASE)
    if not match:
        return raw_text
    return raw_text[: match.end()]


def is_valid_stage_section(section: str, stage: str) -> bool:
    if not section or not section.strip():
        return False
    if f"<{stage}>" in section.upper() or f"</{stage}>" in section.upper():
        return False
    return True


def build_judge_prompt(candidates: Sequence[str], available: Sequence[str], missing: Sequence[str]) -> str:
    formatted = "\n\n".join(f"{idx}. {candidate}" for idx, candidate in enumerate(candidates))
    return JUDGE_PROMPT_TEMPLATE.format(
        available_modalities=", ".join(available) or "none",
        missing_modalities=", ".join(missing) or "none",
        candidates=formatted,
    )


def parse_judge_index(raw_text: str, candidate_count: int) -> int:
    match = re.search(r"\d+", raw_text)
    if not match:
        return 0
    index = int(match.group(0))
    return index if 0 <= index < candidate_count else 0


class OmniPromptGenerator:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        disable_talker: bool = True,
        use_flash_attention_2: bool = False,
        candidate_temperature: float = 0.7,
        candidate_top_p: float = 0.9,
    ) -> None:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.candidate_temperature = candidate_temperature
        self.candidate_top_p = candidate_top_p
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "local_files_only": True,
        }
        if use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        if disable_talker and hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

    def _generate(
        self,
        prompt: str,
        media: MediaPackage,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
    ) -> str:
        content = list(media.content_items)
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Use only the provided modalities for affective semantic analysis."}],
            },
            {"role": "user", "content": content},
        ]
        chat_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if media.has_media:
            from qwen_omni_utils import process_mm_info

            audios, images, videos = process_mm_info(messages, use_audio_in_video=media.use_audio_in_video)
            inputs = self.processor(
                text=chat_text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=media.use_audio_in_video,
            )
        else:
            inputs = self.processor(text=chat_text, return_tensors="pt", padding=True)

        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with self.torch.no_grad():
            generation_kwargs: Dict[str, Any] = {
                "use_audio_in_video": media.use_audio_in_video,
                "thinker_max_new_tokens": max_new_tokens or self.max_new_tokens,
                "do_sample": do_sample,
                "return_audio": False,
            }
            if do_sample:
                generation_kwargs.update(
                    {
                        "temperature": self.candidate_temperature,
                        "top_p": self.candidate_top_p,
                    }
                )
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        if isinstance(output_ids, (tuple, list)):
            output_ids = output_ids[0]
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_section(self, prompt: str, media: MediaPackage, do_sample: bool = False) -> str:
        return self._generate(prompt, media, max_new_tokens=self.max_new_tokens, do_sample=do_sample)

    def choose_candidate(self, judge_prompt: str) -> str:
        return self._generate(judge_prompt, MediaPackage([], False, "text_only", []), max_new_tokens=16)


def load_done_sample_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    done: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["sample_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_error(path: str, item: Mapping[str, Any]) -> None:
    if not path:
        return
    error_parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(error_parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")


def generate_staged_prompt(
    generator: OmniPromptGenerator,
    sample: Mapping[str, Any],
    condition: str,
    media: MediaPackage,
    num_candidates: int,
    candidate_stage: str,
    max_retries: int,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    sections: Dict[str, str] = {}
    diagnostics: Dict[str, Any] = {
        "candidate_stage": candidate_stage,
        "num_candidates": num_candidates,
        "max_retries": max_retries,
        "media_access_mode": media.access_mode,
        "media_source_paths": media.source_paths,
    }
    available = list(CONDITION_TO_MODALITIES[condition])
    missing = [modality for modality in ALL_MODALITIES if modality not in available]

    for stage in STAGES:
        stage_prompt = build_stage_prompt(sample, condition, media, stage, sections)
        if stage == candidate_stage and num_candidates > 1:
            candidates = []
            candidate_attempts = num_candidates + max(0, max_retries)
            for attempt in range(candidate_attempts):
                raw_response = generator.generate_section(stage_prompt, media, do_sample=attempt > 0)
                candidate = parse_stage_response(raw_response, stage)
                if is_valid_stage_section(candidate, stage):
                    candidates.append(candidate)
                if len(candidates) >= num_candidates:
                    break
            if not candidates:
                raise RuntimeError(f"Failed to generate a valid {stage} section for sample {sample['sample_id']}")
            judge_prompt = build_judge_prompt(candidates, available, missing)
            judge_response = generator.choose_candidate(judge_prompt)
            best_index = parse_judge_index(judge_response, len(candidates))
            sections[STAGE_KEYS[stage]] = candidates[best_index]
            diagnostics["candidates"] = candidates
            diagnostics["judge_response"] = judge_response
            diagnostics["selected_index"] = best_index
        else:
            section = ""
            raw_response = ""
            for attempt in range(max(1, max_retries + 1)):
                raw_response = generator.generate_section(stage_prompt, media, do_sample=attempt > 0)
                section = parse_stage_response(raw_response, stage)
                if is_valid_stage_section(section, stage):
                    break
            if not is_valid_stage_section(section, stage):
                raise RuntimeError(
                    f"Failed to generate a valid {stage} section for sample {sample['sample_id']}: {raw_response[:200]}"
                )
            sections[STAGE_KEYS[stage]] = section

    sections["final_answer"] = sections.get("prompt", "")
    return sections, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate semantic prompts for incomplete multimodal sentiment analysis.")
    parser.add_argument("--dataset", default="CMU-MOSI", choices=["CMU-MOSI", "CMU-MOSEI", "IEMOCAP"])
    parser.add_argument("--iemocap-classes", type=int, choices=[4, 6], default=4)
    parser.add_argument("--split", choices=["train", "valid", "test"], required=True)
    parser.add_argument("--condition", required=True, choices=list(CONDITION_TO_MODALITIES))
    parser.add_argument("--raw-data-dir", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio-cache-dir", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--candidate-stage", choices=STAGES, default="PROMPT")
    parser.add_argument("--candidate-temperature", type=float, default=0.7)
    parser.add_argument("--candidate-top-p", type=float, default=0.9)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--use-flash-attention-2", action="store_true")
    parser.add_argument("--keep-talker", action="store_true")
    parser.add_argument("--start-sample-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-generation-input", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--error-path", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dataset = config.normalize_dataset_name(args.dataset, args.iemocap_classes)
    condition = normalize_condition(args.condition)
    available = list(CONDITION_TO_MODALITIES[condition])

    if any(modality in available for modality in ("audio", "visual")) and not args.raw_data_dir:
        raise ValueError("--raw-data-dir is required when audio or visual modalities are available.")

    output_parent = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_parent, exist_ok=True)
    audio_cache_dir = args.audio_cache_dir or os.path.join(output_parent, ".audio_cache")

    done = load_done_sample_ids(args.output_path) if args.resume else set()
    generator = OmniPromptGenerator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        disable_talker=not args.keep_talker,
        use_flash_attention_2=args.use_flash_attention_2,
        candidate_temperature=args.candidate_temperature,
        candidate_top_p=args.candidate_top_p,
    )

    mode = "a" if args.resume else "w"
    with open(args.output_path, mode, encoding="utf-8") as out:
        started = not bool(args.start_sample_id)
        written = 0
        for sample in iter_split_samples(args.dataset, args.split):
            sample_id = str(sample["sample_id"])
            if not started:
                started = sample_id == args.start_sample_id
            if not started or sample_id in done:
                continue

            try:
                media = prepare_media_package(
                    sample=sample,
                    condition=condition,
                    raw_data_dir=args.raw_data_dir,
                    audio_cache_dir=audio_cache_dir,
                    fps=args.fps,
                    max_frames=args.max_frames,
                )

                sections, diagnostics = generate_staged_prompt(
                    generator=generator,
                    sample=sample,
                    condition=condition,
                    media=media,
                    num_candidates=max(1, args.num_candidates),
                    candidate_stage=args.candidate_stage,
                    max_retries=max(0, args.max_retries),
                )
            except Exception as exc:
                if not args.skip_errors:
                    raise
                append_error(
                    args.error_path or f"{args.output_path}.errors",
                    {
                        "sample_id": sample_id,
                        "dataset": args.dataset,
                        "split": args.split,
                        "condition": condition,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                continue

            item: Dict[str, Any] = {
                "sample_id": sample_id,
                "dataset": args.dataset,
                "split": args.split,
                "condition": condition,
                "state": sections.get("state", ""),
                "evidence": sections.get("evidence", ""),
                "inference": sections.get("inference", ""),
                "prompt": sections.get("prompt", ""),
                "final_answer": sections.get("final_answer", ""),
            }
            if args.save_generation_input:
                item["generation_diagnostics"] = diagnostics

            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            if args.limit > 0 and written >= args.limit:
                break


if __name__ == "__main__":
    main()
