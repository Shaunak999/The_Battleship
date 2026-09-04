"""Zip only the backend/ folder into backend.zip for uploading to Google Colab.

Excludes __pycache__ / .pytest_cache directories, *.pyc / *.pyo bytecode,
and pre-trained *.zip model files to keep the upload small.

Run from the project root:
    python make_backend_zip.py
"""

import os
import zipfile

skip_dirs = {"__pycache__", ".pytest_cache"}
skip_ext = {".pyc", ".pyo", ".zip"}

files = []
for root, dirs, fs in os.walk("backend"):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in fs:
        if os.path.splitext(f)[1] not in skip_ext:
            files.append(os.path.join(root, f))

with zipfile.ZipFile("backend.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f)

print(f"Created backend.zip with {len(files)} files, {os.path.getsize('backend.zip') / 1e6:.2f} MB")
