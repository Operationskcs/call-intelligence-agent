"""Build-time only: install pinned public recognizer weights; no call data."""

import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen

REV = "536b0662742c02347bc0e980a01041f333bce120"
FILES = {
    "config.json": "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828",
    "model.bin": "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
    "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
    "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
}
BASE_FILES = {
    **FILES,
    "config.json": "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a",
    "model.bin": "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
}
for model, rev, paths, dest_dir in (
    ("small", REV, FILES, "/opt/conduct-asr"),
    ("base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66", BASE_FILES, "/opt/conduct-asr-base"),
):
    root = Path(dest_dir)
    root.mkdir(exist_ok=True)
    for name, digest in paths.items():
        url = f"https://huggingface.co/Systran/faster-whisper-{model}/resolve/{rev}/{name}"
        with urlopen(url, timeout=90) as src, (root / name).open("wb") as dest:
            shutil.copyfileobj(src, dest)
        with (root / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                raise ValueError(f"Unexpected public model artifact: {model}/{name}")
