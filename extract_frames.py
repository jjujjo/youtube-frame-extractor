#!/usr/bin/env python3
"""
YouTube frame extraction pipeline.

Downloads each video in INPUT_FILE with yt-dlp, extracts frames with ffmpeg,
deduplicates near-identical consecutive frames with perceptual hashing, and
deletes the source video to save disk space.

Requires: yt-dlp and ffmpeg on PATH, plus `pip install imagehash pillow`.
"""

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imagehash
from PIL import Image

# --------------------------------------------------------------------------
# Configuration — edit these as needed
# --------------------------------------------------------------------------

INPUT_FILE = Path(r"urls.txt")
OUTPUT_DIR = Path(r"C:\Users\jood1\Downloads\stc-images\youtube")

FRAME_RATE = 0.5            # frames per second extracted by ffmpeg (1 frame / 2s)
DEDUP_THRESHOLD = 5         # max perceptual-hash difference to count as a duplicate
DELAY_BETWEEN_VIDEOS = 8    # seconds to wait between videos (rate-limit friendly)
MAX_HEIGHT = 1080           # cap download quality at 1080p

BOT_CHECK_COOLDOWN = 120    # seconds to pause after YouTube throws a "Sign in to confirm" bot check
BOT_CHECK_MARKER = "Sign in to confirm"

CONSECUTIVE_FAILURE_THRESHOLD = 5   # back off after this many download failures in a row
CONSECUTIVE_FAILURE_COOLDOWN = 180  # seconds to pause when the threshold is hit

# Path to a Netscape-format cookies.txt file exported from your browser (e.g. via
# the "Get cookies.txt LOCALLY" extension while signed into YouTube). Used instead
# of --cookies-from-browser to avoid Windows DPAPI decryption issues.
COOKIES_FILE = Path(r"cookies.txt")

LOG_FILE_NAME = "process_log.txt"

YT_DLP_FORMAT = f"bestvideo[height<={MAX_HEIGHT}]+bestaudio/best[height<={MAX_HEIGHT}]"

VIDEO_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})"
)

# --------------------------------------------------------------------------


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("frame_extractor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(output_dir / LOG_FILE_NAME, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def read_urls(input_file: Path) -> list[str]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def extract_video_id(url: str) -> str | None:
    match = VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def get_video_id_via_ytdlp(url: str) -> str | None:
    """Fallback: ask yt-dlp for the canonical video ID without downloading."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--print", "id", "--skip-download", url],
            capture_output=True, text=True, timeout=60, check=True,
        )
        video_id = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None
        return video_id or None
    except Exception:
        return None


def download_video(url: str, dest_dir: Path, logger: logging.Logger) -> tuple[Path | None, bool]:
    """Returns (video_path, bot_check_triggered)."""
    output_template = str(dest_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", YT_DLP_FORMAT,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--cookies", str(COOKIES_FILE),
        "--remote-components", "ejs:github",
        "-o", output_template,
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip()
        logger.error(f"yt-dlp failed for {url}: {stderr[-500:]}")
        return None, BOT_CHECK_MARKER in stderr
    except subprocess.TimeoutExpired:
        logger.error(f"yt-dlp timed out for {url}")
        return None, False

    downloaded = list(dest_dir.glob("*.mp4"))
    if not downloaded:
        logger.error(f"yt-dlp reported success but no .mp4 found for {url}")
        return None, False
    return downloaded[0], False


def extract_frames(video_path: Path, frames_dir: Path, fps: float, logger: logging.Logger) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(frames_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        output_pattern,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed for {video_path}: {e.stderr.strip()[-500:]}")
        return 0
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timed out for {video_path}")
        return 0

    return len(list(frames_dir.glob("frame_*.jpg")))


def dedupe_frames(frames_dir: Path, threshold: int, logger: logging.Logger) -> int:
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_files) < 2:
        return 0

    removed = 0
    prev_hash = None
    for frame_path in frame_files:
        try:
            with Image.open(frame_path) as img:
                current_hash = imagehash.phash(img)
        except Exception as e:
            logger.warning(f"Could not hash {frame_path.name}: {e}")
            continue

        if prev_hash is not None and (current_hash - prev_hash) < threshold:
            frame_path.unlink()
            removed += 1
        else:
            prev_hash = current_hash

    return removed


def is_already_done(video_output_dir: Path) -> bool:
    return video_output_dir.exists() and any(video_output_dir.glob("frame_*.jpg"))


def process_url(url: str, output_dir: Path, logger: logging.Logger) -> tuple[str, bool]:
    """Returns (status, bot_check_triggered) where status is 'success', 'skipped', or 'failed'."""
    video_id = extract_video_id(url) or get_video_id_via_ytdlp(url)
    if not video_id:
        logger.error(f"Could not determine video ID for {url}; skipping")
        return "failed", False

    video_output_dir = output_dir / video_id

    if is_already_done(video_output_dir):
        logger.info(f"[{video_id}] Already processed, skipping ({url})")
        return "skipped", False

    logger.info(f"[{video_id}] Processing {url}")

    with tempfile.TemporaryDirectory(prefix=f"ytfe_{video_id}_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        video_path, bot_check = download_video(url, tmp_dir, logger)
        if video_path is None:
            logger.error(f"[{video_id}] Download failed; skipping")
            return "failed", bot_check

        frame_count = extract_frames(video_path, video_output_dir, FRAME_RATE, logger)

        try:
            video_path.unlink()
        except OSError as e:
            logger.warning(f"[{video_id}] Could not delete video file: {e}")

        if frame_count == 0:
            logger.error(f"[{video_id}] No frames extracted; skipping")
            shutil.rmtree(video_output_dir, ignore_errors=True)
            return "failed", False

        removed = dedupe_frames(video_output_dir, DEDUP_THRESHOLD, logger)
        remaining = frame_count - removed

        logger.info(
            f"[{video_id}] Done: {frame_count} frames extracted, "
            f"{removed} duplicates removed, {remaining} kept"
        )
        return "success", False


def check_dependencies(logger: logging.Logger) -> bool:
    missing = [tool for tool in ("yt-dlp", "ffmpeg") if shutil.which(tool) is None]
    if missing:
        logger.error(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install them and/or restart your terminal so PATH changes take effect, "
            "then verify with `yt-dlp --version` and `ffmpeg -version`."
        )
        return False

    if not COOKIES_FILE.exists():
        logger.error(
            f"Cookies file not found: {COOKIES_FILE}. Export youtube.com cookies "
            "(e.g. via the \"Get cookies.txt LOCALLY\" Chrome extension while signed "
            "into YouTube) and save them to that path."
        )
        return False

    return True


def main():
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=== Starting YouTube frame extraction run ===")

    if not check_dependencies(logger):
        sys.exit(1)

    urls = read_urls(INPUT_FILE)
    logger.info(f"Loaded {len(urls)} URLs from {INPUT_FILE}")

    stats = {"success": 0, "skipped": 0, "failed": 0}
    consecutive_failures = 0

    for i, url in enumerate(urls, start=1):
        logger.info(f"--- [{i}/{len(urls)}] {url} ---")
        try:
            result, bot_check = process_url(url, OUTPUT_DIR, logger)
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}")
            result, bot_check = "failed", False

        stats[result] += 1
        consecutive_failures = consecutive_failures + 1 if result == "failed" else 0

        if bot_check:
            logger.warning(
                f"YouTube bot check triggered; cooling down for {BOT_CHECK_COOLDOWN}s "
                "(make sure the Chrome profile yt-dlp reads is signed into YouTube)"
            )
            time.sleep(BOT_CHECK_COOLDOWN)
            consecutive_failures = 0
        elif consecutive_failures >= CONSECUTIVE_FAILURE_THRESHOLD:
            logger.warning(
                f"{consecutive_failures} consecutive download failures; cooling down for "
                f"{CONSECUTIVE_FAILURE_COOLDOWN}s (likely an outdated yt-dlp or a systemic "
                "block — consider running `pip install -U yt-dlp`)"
            )
            time.sleep(CONSECUTIVE_FAILURE_COOLDOWN)
            consecutive_failures = 0
        elif result != "skipped" and i < len(urls):
            time.sleep(DELAY_BETWEEN_VIDEOS)

    logger.info(
        f"=== Run complete: {stats['success']} succeeded, "
        f"{stats['skipped']} skipped, {stats['failed']} failed ==="
    )


if __name__ == "__main__":
    main()
