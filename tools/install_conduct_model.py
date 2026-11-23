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
root = Path("/opt/conduct-asr")
root.mkdir(exist_ok=True)
for name, digest in FILES.items():
    url = f"https://huggingface.co/Systran/faster-whisper-small/resolve/{REV}/{name}"
    with urlopen(url, timeout=90) as src, (root / name).open("wb") as dest:
        shutil.copyfileobj(src, dest)
    with (root / name).open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != digest:
            raise ValueError(f"Unexpected public model artifact: {name}")
