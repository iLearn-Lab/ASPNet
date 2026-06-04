CONDITION_TO_MODALITIES = {
    "a": ("audio",),
    "t": ("text",),
    "v": ("visual",),
    "at": ("audio", "text"),
    "av": ("audio", "visual"),
    "tv": ("text", "visual"),
    "atv": ("audio", "text", "visual"),
}


def normalize_condition(condition: str) -> str:
    normalized = "".join(sorted(set(condition.lower()), key="atv".index))
    if normalized not in CONDITION_TO_MODALITIES:
        raise ValueError(f"Unsupported modality condition: {condition}")
    return normalized

