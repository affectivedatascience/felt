"""
Download RAVDESS video ZIP archives from Zenodo.

This utility downloads the RAVDESS Video_Speech and Video_Song actor ZIP files
from the RAVDESS Zenodo record. It is a standalone data-acquisition helper and
is not part of the numbered FELT processing pipeline.

Expected project layout
-----------------------
face-tracking-2024/
├── 01_data/
│   ├── 00_downloads/
│   │   └── ravdess/
│   │       ├── Video_Speech_Actor_01.zip
│   │       ├── Video_Song_Actor_01.zip
│   │       └── ...
│   ├── 01_input/
│   └── 02_output/
└── 02_code/
    └── src/
        ├── tools/
        │   └── download_ravdess_archives.py
        └── utils/
            ├── __init__.py
            └── felt_paths.py

Notes
-----
RAVDESS does not include song files for Actor 18, so
Video_Song_Actor_18.zip is skipped.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm


# =============================================================================
# Make the local utils/ package importable when this script is run directly.
# =============================================================================

CODE_ROOT = Path(__file__).resolve()
for parent in CODE_ROOT.parents:
    if (parent / "utils").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not find code root containing utils/.")

from utils.felt_paths import (
    ACTOR_WITHOUT_SONG,
    DATA_DIR,
    LOG_DIR,
    PROJECT_ROOT,
    configure_logging,
)


# =============================================================================
# User-editable configuration
# =============================================================================

RAVDESS_RECORD_ID = "1188976"
ZENODO_BASE_URL = f"https://zenodo.org/records/{RAVDESS_RECORD_ID}/files"

DOWNLOAD_DIR = DATA_DIR / "00_downloads" / "ravdess"
LOG_FILE = LOG_DIR / "download_ravdess_archives.log"

START_ACTOR = 1
END_ACTOR = 24

SKIP_EXISTING = True


# =============================================================================
# URL helpers
# =============================================================================

def actor_number(actor_id: int) -> str:
    """Return the two-digit RAVDESS actor number."""
    return f"{actor_id:02}"


def speech_zip_name(actor_id: int) -> str:
    """Return the RAVDESS speech ZIP filename for one actor."""
    return f"Video_Speech_Actor_{actor_number(actor_id)}.zip"


def song_zip_name(actor_id: int) -> str:
    """Return the RAVDESS song ZIP filename for one actor."""
    return f"Video_Song_Actor_{actor_number(actor_id)}.zip"


def zenodo_file_url(filename: str) -> str:
    """Return the Zenodo download URL for one RAVDESS archive."""
    return f"{ZENODO_BASE_URL}/{filename}"


# =============================================================================
# Download functions
# =============================================================================

def download_file(url: str, output_path: Path) -> None:
    """Download one file with a progress bar updated every ~25 MB."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if SKIP_EXISTING and output_path.exists():
        logging.info("File already exists; skipping: %s", output_path)
        print(f"Already exists; skipping: {output_path}")
        return

    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    logging.info("Downloading %s to %s", url, output_path)
    print(f"Downloading {url}")
    print(f"    to {output_path}")

    chunk_size = 25 * 1024 * 1024  # 25 MB

    if temp_path.exists():
        logging.info("Removing incomplete prior download: %s", temp_path)
        temp_path.unlink()

    try:
        with urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))

            with open(temp_path, "wb") as file_out, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name,
                miniters=1,
                mininterval=0.5,
                dynamic_ncols=False,
                leave=True,
            ) as progress:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    file_out.write(chunk)
                    progress.update(len(chunk))

        temp_path.replace(output_path)

    except Exception:
        logging.exception("Download failed: %s", url)
        if temp_path.exists():
            logging.warning("Incomplete download retained at: %s", temp_path)
        raise

    logging.info("Downloaded: %s", output_path)
    print(f"Downloaded: {output_path}")


def build_download_tasks() -> list[tuple[str, Path]]:
    """Build download tasks for RAVDESS speech and song archives."""
    tasks: list[tuple[str, Path]] = []

    for actor_id in range(START_ACTOR, END_ACTOR + 1):
        speech_name = speech_zip_name(actor_id)
        tasks.append(
            (
                zenodo_file_url(speech_name),
                DOWNLOAD_DIR / speech_name,
            )
        )

    for actor_id in range(START_ACTOR, END_ACTOR + 1):
        if actor_id == ACTOR_WITHOUT_SONG:
            logging.info("Skipping song archive for Actor %02d; not present in RAVDESS.", actor_id)
            continue

        song_name = song_zip_name(actor_id)
        tasks.append(
            (
                zenodo_file_url(song_name),
                DOWNLOAD_DIR / song_name,
            )
        )

    logging.info("Prepared %d download tasks.", len(tasks))
    return tasks


def main() -> None:
    """Download RAVDESS speech and song video ZIP archives."""
    configure_logging(LOG_FILE)

    print(f"Writing log to: {LOG_FILE}")

    logging.info("Starting RAVDESS archive download.")
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("DOWNLOAD_DIR: %s", DOWNLOAD_DIR)
    logging.info("LOG_FILE: %s", LOG_FILE)
    logging.info("Actors: %02d-%02d", START_ACTOR, END_ACTOR)
    logging.info("SKIP_EXISTING: %s", SKIP_EXISTING)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    tasks = build_download_tasks()

    for index, (url, output_path) in enumerate(tasks, start=1):
        logging.info("Processing download task %d/%d: %s", index, len(tasks), output_path)
        print(f"[{index}/{len(tasks)}]")
        download_file(url, output_path)

    logging.info("RAVDESS archive download complete.")
    print("RAVDESS archive download complete.")


if __name__ == "__main__":
    main()