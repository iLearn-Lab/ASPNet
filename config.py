import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
SAVED_ROOT = os.path.join(PROJECT_ROOT, "saved")

MODEL_DIR = os.path.join(SAVED_ROOT, "model")
LOG_DIR = os.path.join(SAVED_ROOT, "log")
NPZ_DIR = os.path.join(SAVED_ROOT, "npz")


DATASET_ALIASES = {
    "CMUMOSI": "CMU-MOSI",
    "CMU_MOSI": "CMU-MOSI",
    "CMU-MOSI": "CMU-MOSI",
    "MOSI": "CMU-MOSI",
    "CMUMOSEI": "CMU-MOSEI",
    "CMU_MOSEI": "CMU-MOSEI",
    "CMU-MOSEI": "CMU-MOSEI",
    "MOSEI": "CMU-MOSEI",
    "IEMOCAP": "IEMOCAP",
    "IEMOCAP4": "IEMOCAP4",
    "IEMOCAPFour": "IEMOCAP4",
    "IEMOCAP-4": "IEMOCAP4",
    "IEMOCAP6": "IEMOCAP6",
    "IEMOCAPSix": "IEMOCAP6",
    "IEMOCAP-6": "IEMOCAP6",
}


def normalize_dataset_name(name: str, iemocap_classes: int = 4) -> str:
    canonical = DATASET_ALIASES.get(name, name)
    if canonical == "IEMOCAP":
        return f"IEMOCAP{iemocap_classes}"
    if canonical not in DATA_DIR:
        raise ValueError(f"Unsupported dataset: {name}")
    return canonical


DATA_DIR = {
    "CMU-MOSI": os.path.join(DATASET_ROOT, "CMUMOSI"),
    "CMU-MOSEI": os.path.join(DATASET_ROOT, "CMUMOSEI"),
    "IEMOCAP4": os.path.join(DATASET_ROOT, "IEMOCAP"),
    "IEMOCAP6": os.path.join(DATASET_ROOT, "IEMOCAP"),
}

PATH_TO_RAW_AUDIO = {name: os.path.join(path, "subaudio") for name, path in DATA_DIR.items()}
PATH_TO_RAW_FACE = {
    "CMU-MOSI": os.path.join(DATA_DIR["CMU-MOSI"], "openface_face"),
    "CMU-MOSEI": os.path.join(DATA_DIR["CMU-MOSEI"], "openface_face"),
    "IEMOCAP4": os.path.join(DATA_DIR["IEMOCAP4"], "subvideofaces"),
    "IEMOCAP6": os.path.join(DATA_DIR["IEMOCAP6"], "subvideofaces"),
}
PATH_TO_FEATURES = {name: os.path.join(path, "features") for name, path in DATA_DIR.items()}
PATH_TO_LABEL = {
    "CMU-MOSI": os.path.join(DATA_DIR["CMU-MOSI"], "CMUMOSI_features_raw_2way.pkl"),
    "CMU-MOSEI": os.path.join(DATA_DIR["CMU-MOSEI"], "CMUMOSEI_features_raw_2way.pkl"),
    "IEMOCAP4": os.path.join(DATA_DIR["IEMOCAP4"], "IEMOCAP_features_raw_4way.pkl"),
    "IEMOCAP6": os.path.join(DATA_DIR["IEMOCAP6"], "IEMOCAP_features_raw_6way.pkl"),
}


REGRESSION_DATASETS = {"CMU-MOSI", "CMU-MOSEI"}
CLASSIFICATION_DATASETS = {"IEMOCAP4", "IEMOCAP6"}
SUPPORTED_DATASETS = tuple(DATA_DIR)
