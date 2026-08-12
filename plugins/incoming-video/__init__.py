"""Incoming video auto-processing plugin.

Detects the gateway's "[User sent a video attachment: ...]" note in the
incoming user message, extracts the cached video path, transcribes the audio
track with the configured STT provider (``transcribe_audio`` -> honours
``stt.provider`` in config.yaml, which on this host is openai via openproxy),
extracts a few frames with ffmpeg, and injects the transcript + frame paths
into the turn via the ``pre_llm_call`` hook.

The agent never sees the raw video — it receives the digest already made.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

# Matches the note the gateway prepends in gateway/run.py:16585
#   [The user sent a video attachment: '<display>'. It is saved at: <path>. ...]
_VIDEO_NOTE_RE = re.compile(
    r"\[The user sent a video attachment: '(?P<name>[^']*)'\.\s*"
    r"It is saved at: (?P<path>\S+)\.\s*"
)

# Video extensions the gateway caches (gateway/platforms/base.py VIDEO_EXT_TO_MIME)
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

# Frames to extract based on duration
_FRAMES_BY_DURATION = [(30, 4), (120, 6), (float("inf"), 9)]


def _extract_video_path(user_message: str) -> str | None:
    """Return the cached video path from the gateway attachment note, if any."""
    m = _VIDEO_NOTE_RE.search(user_message)
    if not m:
        return None
    path = m.group("path").strip()
    if not path:
        return None
    ext = Path(path).suffix.lower()
    if ext not in _VIDEO_EXTS:
        # Not a video (e.g. a document) — let the agent handle it normally.
        return None
    if not Path(path).exists():
        logger.warning("[incoming-video] cached path missing: %s", path)
        return None
    return path


def _video_duration(path: str) -> float:
    """Return duration in seconds via ffprobe, or 0.0 on failure."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(out.stdout.strip())
    except Exception as exc:  # noqa: BLE001 — never break the turn
        logger.warning("[incoming-video] ffprobe failed: %s", exc)
        return 0.0


def _frames_for_duration(duration: float) -> int:
    for limit, n in _FRAMES_BY_DURATION:
        if duration <= limit:
            return n
    return 9


def _extract_audio(video_path: str, workdir: Path) -> Path | None:
    """Extract audio track to WAV 16k mono. Returns path or None (no audio)."""
    wav = workdir / "audio.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(wav),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[incoming-video] ffmpeg audio extract failed: %s", exc)
        return None
    if not wav.exists() or wav.stat().st_size < 1024:
        return None
    return wav


def _transcribe(wav_path: Path) -> str:
    """Transcribe with the configured STT provider via transcribe_audio()."""
    try:
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio(str(wav_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[incoming-video] transcribe_audio raised: %s", exc)
        return ""
    if not result.get("success"):
        logger.warning(
            "[incoming-video] transcription failed: %s",
            result.get("error", "unknown"),
        )
        return ""
    return (result.get("transcript") or "").strip()


def _extract_frames(video_path: str, workdir: Path, n: int) -> list[str]:
    """Extract up to n frames (720p) into workdir. Returns list of paths."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "fps=1/5,scale=720:-2",
                "-frames:v", str(n),
                str(workdir / "frame_%02d.png"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[incoming-video] frame extract failed: %s", exc)
        return []
    frames = sorted(str(p) for p in workdir.glob("frame_*.png"))
    return frames[:n]


def _persist_frames(tmp_frames: list[str], video_path: str) -> list[str]:
    """Move frames from the tempdir into the persistent video cache.

    The gateway caches videos under ~/.hermes/cache/videos; frames land next
    to them so the agent can vision_analyze them later (tempdir is wiped).
    """
    cache_dir = Path(os.path.expanduser("~/.hermes/cache/videos"))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[incoming-video] cannot create cache dir: %s", exc)
        return tmp_frames
    stem = Path(video_path).stem
    out: list[str] = []
    for i, fp in enumerate(tmp_frames, start=1):
        dest = cache_dir / f"{stem}_frame_{i:02d}.png"
        try:
            shutil.move(fp, dest)
            out.append(str(dest))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[incoming-video] frame persist failed: %s", exc)
            out.append(fp)
    return out


def _process_video(video_path: str) -> str:
    """Full pipeline. Returns the context block to inject (or '')."""
    parts: list[str] = []

    with tempfile.TemporaryDirectory(prefix="incoming-video-") as tmp:
        workdir = Path(tmp)

        # Transcript
        wav = _extract_audio(video_path, workdir)
        transcript = _transcribe(wav) if wav else ""
        if transcript:
            parts.append(f"[Transcripción del vídeo]\n{transcript}")
        else:
            parts.append("[El vídeo no tiene pista de audio transcribible o la transcripción falló]")

        # Frames
        duration = _video_duration(video_path)
        n = _frames_for_duration(duration)
        tmp_frames = _extract_frames(video_path, workdir, n)
        frames = _persist_frames(tmp_frames, video_path)
        if frames:
            parts.append(
                "[Frames extraídos del vídeo] "
                + ", ".join(frames)
                + " — usa vision_analyze sobre ellos si el usuario pide contenido visual."
            )

    return "\n\n".join(parts)


def _on_pre_llm_call(user_message: str, **kwargs) -> str | None:
    """pre_llm_call hook: transcribe + frame incoming videos before the LLM turn."""
    if not user_message or not isinstance(user_message, str):
        return None

    video_path = _extract_video_path(user_message)
    if not video_path:
        return None

    logger.info("[incoming-video] processing: %s", video_path)
    context = _process_video(video_path)
    if not context:
        return None

    return {
        "context": (
            "[Procesamiento automático del vídeo entrante]\n"
            + context
            + "\n\nEl vídeo original está en: " + video_path
        )
    }


def register(ctx) -> None:
    """Plugin entry point — wire the pre_llm_call hook."""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("[incoming-video] plugin registered (pre_llm_call hook)")
