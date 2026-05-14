"""setup_kaggle.py — curated downloader for the Kaggle QUIC dataset
(adam357/quic-network-capture-data).

The full archive is ~88 GB — far more than we need. This script pulls a
hand-picked subset (~2.5 GB) that gives us ~1.5 GB of benign traffic and
~1 GB of DDoS / GET-FLOOD traffic, which is plenty to lift the benign
class from 306 flows up to "something reasonable" for the Phase 3
rebalancing.

What this script does:
    1. For every entry in BENIGN_FILES / DDOS_FILES below, calls
       `kaggle datasets download -d <id> -f <path>` into a staging dir.
    2. Each download arrives as a zip — extract the contained pcap into
       the right Phase 3 folder:
         data/raw_pcap/benign/kaggle/
         data/raw_pcap/ddos/kaggle/
    3. Delete every zip + the staging dir at the end to reclaim disk.

Prerequisite:
    - Kaggle API token at %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
      or ~/.kaggle/kaggle.json (Unix).
    - `pip install kaggle` so either the CLI or the Python module is available.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_ID = "adam357/quic-network-capture-data"

# ---------------------------------------------------------------------------
# Curated file list — chosen for size, not content. We want enough variety
# (different normal_traffic_* and GET_FLOOD recordings) to avoid overfitting
# to a single capture session, but small enough to be processable locally.
# ---------------------------------------------------------------------------

BENIGN_FILES: list[str] = [
    # normal_traffic_7 — small files, ideal starting point
    "normal_traffic_7/traffic_normal/traffic_record_normal1.pcap",   # 65.7 MB
    "normal_traffic_7/traffic_normal/traffic_record_normal3.pcap",   # 80.4 MB
    "normal_traffic_7/traffic_normal/traffic_record_normal2.pcap",   # 94.5 MB
    "normal_traffic_7/traffic_normal/traffic_record_normal.pcap",    # 118.5 MB
    "normal_traffic_7/traffic_normal/traffic_record_normal4.pcap",   # 223.1 MB
    # one mid-sized file from a different recording session
    "normal_traffic_3/normal_traffic3/normal_traffic_recording_3.pcap",  # 262.8 MB
    # one larger file for additional variety
    "normal_traffic_2/normal_traffic2/normal_traffic_recording_17.pcap", # 617.1 MB
]   # total ~1.46 GB

DDOS_FILES: list[str] = [
    # small GET_FLOOD recordings
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_10.pcap",  # 14.1 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_12.pcap",  # 14.1 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_13.pcap",  # 14.0 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_24.pcap",  # 37.1 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_25.pcap",  # 37.1 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_15.pcap",  # 49.9 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_19.pcap",  # 54.2 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_18.pcap",  # 84.6 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_22.pcap",  # 110.7 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_27.pcap",  # 174.9 MB
    "traffic_recording/traffic_recording/GET_FLOOD_PCAP/recording_3.pcap",   # 421.7 MB
]   # total ~1.01 GB

log = logging.getLogger(__name__)


def kaggle_cli_available() -> bool:
    return shutil.which("kaggle") is not None


def download_one(remote_path: str, staging: Path) -> None:
    """Pull a single file from the dataset via the Kaggle CLI / API.

    Kaggle wraps every single-file download in a zip, regardless of the
    underlying file type. We extract it in extract_one().
    """
    log.info("Downloading %s", remote_path)
    if kaggle_cli_available():
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", DATASET_ID,
             "-f", remote_path,
             "-p", str(staging),
             "--force"],
            check=True,
        )
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(
            DATASET_ID, file_name=remote_path, path=str(staging), force=True,
        )


def extract_one(remote_path: str, staging: Path, target_dir: Path) -> Path:
    """Locate the zip that corresponds to `remote_path` and extract its
    .pcap content into `target_dir`. Returns the final pcap path.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle names the zip after the basename of the remote path.
    pcap_name = Path(remote_path).name
    zip_candidates = [
        staging / f"{pcap_name}.zip",
        staging / f"{Path(remote_path).stem}.zip",
    ]
    zip_path = next((z for z in zip_candidates if z.exists()), None)

    # Some kaggle versions skip zipping for already-compressed files and drop
    # the pcap straight into staging.
    if zip_path is None:
        direct = staging / pcap_name
        if direct.exists():
            target = target_dir / pcap_name
            if target.exists():
                target.unlink()
            shutil.move(str(direct), str(target))
            return target
        raise FileNotFoundError(
            f"Kaggle download did not produce a zip or raw pcap for {remote_path} "
            f"in {staging} (looked for {[p.name for p in zip_candidates]})."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".pcap")]
        if not members:
            raise RuntimeError(f"No .pcap inside {zip_path.name}")
        # Most Kaggle single-file zips contain exactly one entry
        member = members[0]
        extracted = Path(zf.extract(member, staging))

    target = target_dir / pcap_name
    if target.exists():
        target.unlink()
    shutil.move(str(extracted), str(target))

    # cleanup the zip immediately to keep disk usage low while iterating
    zip_path.unlink(missing_ok=True)
    return target


def process_group(files: list[str], staging: Path, target_dir: Path,
                  label: str) -> int:
    n_ok = 0
    for remote in files:
        try:
            download_one(remote, staging)
            extracted = extract_one(remote, staging, target_dir)
            log.info("  [%-6s] %s", label, extracted.name)
            n_ok += 1
        except Exception as exc:
            log.error("  [%-6s] FAILED %s — %s", label, remote, exc)
    return n_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curated downloader for the Kaggle QUIC dataset (Phase 3).",
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent,
                        help="project root (default: script directory)")
    parser.add_argument("--keep-staging", action="store_true",
                        help="don't delete the staging dir after extraction")
    args = parser.parse_args()

    base = args.base_dir.resolve()
    staging = base / "data" / "_kaggle_staging"
    staging.mkdir(parents=True, exist_ok=True)

    benign_dir = base / "data" / "raw_pcap" / "benign" / "kaggle"
    ddos_dir   = base / "data" / "raw_pcap" / "ddos"   / "kaggle"

    log.info("Targeted Kaggle download: %d benign + %d ddos files",
             len(BENIGN_FILES), len(DDOS_FILES))

    n_benign = process_group(BENIGN_FILES, staging, benign_dir, "benign")
    n_ddos   = process_group(DDOS_FILES,   staging, ddos_dir,   "ddos")

    log.info("=" * 56)
    log.info("  Routed %d benign + %d ddos pcap(s)", n_benign, n_ddos)
    log.info("    -> %s", benign_dir)
    log.info("    -> %s", ddos_dir)
    log.info("=" * 56)

    if n_benign == 0 and n_ddos == 0:
        log.error("Nothing was downloaded — check Kaggle auth and the file list.")
        sys.exit(1)

    if not args.keep_staging:
        log.info("Cleaning up staging directory %s", staging)
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
