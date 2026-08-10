"""
AI Horde image generation backend.

Connects to the Horde API (https://aihorde.net/api/v2) — a distributed
inference cluster with 50+ models: Flux, SDXL, Pony, SD15, and many more.

Flow:
1. POST /generate/async  →  {id: "abc123"}
2. GET  /generate/check/{id}  →  {done: false, wait_time: 10, queue_position: 3}
3. GET  /generate/status/{id}  →  {done: true, generations: [...]}
4. Download image URL → save to cache

API key: set HORDE_API_KEY env var or configure via hermes tools.
Without an API key, the Horde assigns one anonymously (0000000000).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Horde API
# ---------------------------------------------------------------------------

HORDE_API_URL = "https://aihorde.net/api/v2"
DEFAULT_TIMEOUT = 300  # 5min max wait for generation
POLL_INTERVAL = 5      # seconds between status checks
REQUEST_TIMEOUT = 60   # http request timeout
# Worker-side NSFW censorship: the worker adds state="censored" to the generation
# when its post-gen NSFW detector trips. We re-submit the same request (different
# worker pool) up to this many times before giving up.
MAX_CENSORED_RETRIES = 3
# Max size for source images (downloaded and sent as base64)
MAX_SOURCE_IMAGE_SIZE = 1024
# A worker can return a "done" job with garbage bytes (not a real image).
# We validate the downloaded bytes and re-submit up to this many times.
MAX_BROKEN_RETRIES = 2
# Minimum plausible size for a generated image (PNG/JPEG header + payload)
MIN_IMAGE_BYTES = 1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _download_and_encode_image(url: str, max_size: int = MAX_SOURCE_IMAGE_SIZE) -> Optional[str]:
    """Download an image from URL (http/https/data: URI/file path) and return base64 JPEG.
    
    Returns None if download or conversion fails.
    """
    import base64
    import io
    from pathlib import Path
    from PIL import Image
    
    try:
        image_bytes = None
        
        # data: URI
        if url.startswith("data:"):
            # data:image/jpeg;base64,/9j/4AAQ...
            if "," in url:
                b64_part = url.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_part)
        
        # Local file path
        elif url.startswith("/") or url.startswith("~") or (len(url) > 1 and url[1] == ":"):
            path = Path(url).expanduser()
            if path.exists():
                image_bytes = path.read_bytes()
        
        # HTTP/HTTPS URL
        elif url.startswith("http://") or url.startswith("https://"):
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        image_bytes = await resp.read()
        
        if not image_bytes:
            return None
        
        # Convert to WEBP base64 (Horde API requires webp: "The Base64-encoded
        # webp to use for img2img" — swagger GenerationInputStable.source_image)
        return await asyncio.to_thread(_encode_image_sync, image_bytes, max_size)
        
    except Exception as e:
        logger.warning("Failed to download/encode image %s: %s", url, e)
        return None


def _encode_image_sync(image_bytes: bytes, max_size: int) -> Optional[str]:
    """Synchronous image processing (runs in thread pool)."""
    import base64
    import io
    from PIL import Image
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Resize if too large
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))
        
        # Convert to RGB (remove alpha for JPEG)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        buffer = io.BytesIO()
        img.save(buffer, format="WEBP", quality=95)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
    except Exception as e:
        logger.error("Error encoding image: %s", e)
        return None


def _download_and_encode_image_sync(url: str, max_size: int = MAX_SOURCE_IMAGE_SIZE) -> Optional[str]:
    """Synchronous wrapper for _download_and_encode_image (runs in thread pool)."""
    return asyncio.run(_download_and_encode_image(url, max_size))

# ---------------------------------------------------------------------------
# Model catalog  — curated subset of popular Horde models
# ---------------------------------------------------------------------------

MODELS: Dict[str, Dict[str, Any]] = {
    # === Flux (only Schnell fp8 Compact sometimes has workers) ===
    "flux": {
        "display": "Flux.1-Schnell (fp8)",
        "speed": "~15s (demand-limited, 403 frecuente)",
        "strengths": "Solo Flux con workers, pero limitado — probar antes",
        "horde_name": "Flux.1-Schnell fp8 (Compact)",
        "category": "rare",
    },
    # === SDXL (más estables) ===
    "sdxl:1.0": {
        "display": "SDXL 1.0",
        "speed": "~8s",
        "strengths": "7 workers, queue 0, el más confiable",
        "horde_name": "SDXL 1.0",
        "category": "recommended",
    },
    "sdxl:albedo": {
        "display": "AlbedoBase XL (SDXL)",
        "speed": "~22s",
        "strengths": "12 workers, popular, cola ~22s",
        "horde_name": "AlbedoBase XL (SDXL)",
        "category": "recommended",
    },
    "sdxl:anime": {
        "display": "Nova Anime XL",
        "speed": "~6s",
        "strengths": "7 workers, anime, cola ~6s",
        "horde_name": "Nova Anime XL",
        "category": "recommended",
    },
    # === Pony ===
    "pony:realistic": {
        "display": "CyberRealistic Pony",
        "speed": "~55s",
        "strengths": "8 workers, photorealistic, cola ~55s",
        "horde_name": "CyberRealistic Pony",
        "category": "realistic",
    },
    "pony:anime": {
        "display": "WAI-ANI-NSFW-PONYXL",
        "speed": "~1s",
        "strengths": "5 workers, anime, cola mínima",
        "horde_name": "WAI-ANI-NSFW-PONYXL",
        "category": "anime",
    },
    "pony:amp": {
        "display": "AMPonyXL",
        "speed": "~variable",
        "strengths": "5 workers, cola 0 status pero cola real grande",
        "horde_name": "AMPonyXL",
        "category": "realistic",
    },
    # === SD 1.5 ===
    "sd15": {
        "display": "Stable Diffusion 1.5",
        "speed": "~6s",
        "strengths": "11 workers, cola ~6s",
        "horde_name": "stable_diffusion",
        "category": "recommended",
    },
    # === Especializados ===
    "icbinp": {
        "display": "ICBINP",
        "speed": "~variable",
        "strengths": "7 workers, photorealistic, cola variable",
        "horde_name": "ICBINP - I Can't Believe It's Not Photography",
        "category": "realistic",
    },
    "deliberate": {
        "display": "Deliberate",
        "speed": "~47s",
        "strengths": "7 workers, versátil, cola ~47s",
        "horde_name": "Deliberate",
        "category": "recommended",
    },
    "dreamshaper": {
        "display": "DreamShaper",
        "speed": "~12s",
        "strengths": "5 workers, artístico, cola ~12s",
        "horde_name": "Dreamshaper",
        "category": "artistic",
    },
    "juggernaut-xl": {
        "display": "Juggernaut XL",
        "speed": "~81s",
        "strengths": "5 workers, SDXL finetune, cola ~81s",
        "horde_name": "Juggernaut XL",
        "category": "realistic",
    },
}

DEFAULT_MODEL = "sdxl:albedo"

# Parameter metadata for the Hermes tools UI
PARAMS_METADATA: Dict[str, Dict[str, Any]] = {
    "cfg_scale": {
        "type": "float",
        "default": 6.5,
        "min": 1.0,
        "max": 30.0,
        "description": "Classifier-free guidance scale. Higher = prompt adherence but less creative.",
        "category": "advanced",
    },
    "steps": {
        "type": "int",
        "default": 25,
        "min": 1,
        "max": 100,
        "description": "Number of sampling steps. Higher = more detail, slower.",
        "category": "basic",
    },
    "sampler": {
        "type": "str",
        "default": "k_euler_a",
        "options": [
            "k_lms", "k_heun", "k_euler", "k_euler_a",
            "k_dpm_2", "k_dpm_2_a", "k_dpmpp_2s_a", "k_dpmpp_2m",
            "k_dpmpp_sde", "lcm", "DDIM",
        ],
        "description": "Sampling method.",
        "category": "advanced",
    },
    "base_resolution": {
        "type": "int",
        "default": 768,
        "min": 64,
        "max": 2048,
        "description": "Base resolution (long side, en ancho). Final size = base x (height from aspect ratio), redondeada hacia abajo a múltiplo de 64.",
        "category": "basic",
    },
    "seed": {
        "type": "int",
        "default": None,
        "description": "Random seed for reproducibility. None = random.",
        "category": "advanced",
    },
    "post_processing": {
        "type": "str",
        "default": [],
        "options": ["GFPGAN", "RealESRGAN_x4plus", "CodeFormers", "strip_background"],
        "description": "Post-processing filters to apply.",
        "category": "advanced",
    },
    "denoising_strength": {
        "type": "float",
        "default": None,
        "min": 0.0,
        "max": 1.0,
        "description": "Img2img denoising strength. 0 = no change, 1 = full change.",
        "category": "advanced",
    },
    "negative_prompt": {
        "type": "str",
        "default": "",
        "description": "Things to avoid in the image.",
        "category": "basic",
    },
    "source_processing": {
        "type": "str",
        "default": "img2img",
        "options": ["img2img", "inpainting", "outpainting", "remix", "txt2img"],
        "description": "How to process the source image (img2img, inpainting, outpainting, remix).",
        "category": "advanced",
    },
}

ASPECT_RATIOS = {
    "landscape": (16, 9),   # 16:9 (ancho dominante)
    "square": (1, 1),        # 1:1
    "portrait": (2, 3),      # 2:3 (alto dominante, portrait vertical)
}


def _nearest_multiple(value: int, multiple: int = 64) -> int:
    """Round to nearest multiple of 64."""
    return max(64, round(value / multiple) * multiple)


def _resolve_size(base_resolution: Optional[int], aspect: str) -> Tuple[int, int]:
    """Resolve final dimensions from a base resolution (long side) and aspect ratio.

    base_resolution is the LONGEST side the user wants. The short side is derived
    from the aspect ratio and rounded DOWN to the nearest multiple of 64.
    """
    base = int(base_resolution) if base_resolution else 768
    base = _nearest_multiple(max(64, base))
    ratio_w, ratio_h = ASPECT_RATIOS.get(aspect, ASPECT_RATIOS["square"])
    if ratio_w >= ratio_h:
        # landscape/square: width is the long side (base), height is short
        short = int(base * ratio_h / ratio_w)
        short = (short // 64) * 64
        if short < 64:
            short = 64
        return base, short
    # portrait: height is the long side (base), width is short
    short = int(base * ratio_w / ratio_h)
    short = (short // 64) * 64
    if short < 64:
        short = 64
    return short, base

# ---------------------------------------------------------------------------
# Horde API client — per-request sessions (thread-safe)
# ---------------------------------------------------------------------------

def _sanitize_key(text: str, api_key: str) -> str:
    """Replace API key with asterisks for safe logging."""
    if api_key and api_key != "0000000000" and isinstance(text, str):
        return text.replace(api_key, "********")
    return text


async def _horde_generate(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    model_horde_name: str,
    params: Dict[str, Any],
    root_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Submit a generation request to Horde. Returns the response."""
    payload = {
        "prompt": prompt,
        "params": params,
        **root_params,
    }
    if model_horde_name:
        payload["models"] = [model_horde_name]

    headers = {
        "apikey": api_key,
        "Client-Agent": "HermesAgent:HordePlugin:v1.0.0:hermeona",
        "Content-Type": "application/json",
    }

    logger.debug(
        "Horde submit: model=%s params=%s",
        model_horde_name,
        _sanitize_key(json.dumps({k: v for k, v in params.items() if k != "prompt"}), api_key),
    )

    async with session.post(
        f"{HORDE_API_URL}/generate/async",
        json=payload,
        headers=headers,
    ) as resp:
        if resp.status != 202:
            text = await resp.text()
            msg = _sanitize_key(f"Generation failed ({resp.status}): {text}", api_key)
            logger.error(msg)
            return {"error": msg}
        return await resp.json()


async def _horde_check(session: aiohttp.ClientSession, api_key: str, generation_id: str) -> Dict[str, Any]:
    """Check generation status."""
    headers = {"apikey": api_key}
    async with session.get(
        f"{HORDE_API_URL}/generate/check/{generation_id}",
        headers=headers,
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            return {"error": f"Check failed ({resp.status}): {text}"}
        return await resp.json()


async def _horde_status(session: aiohttp.ClientSession, api_key: str, generation_id: str) -> Dict[str, Any]:
    """Get generation results."""
    headers = {"apikey": api_key}
    async with session.get(
        f"{HORDE_API_URL}/generate/status/{generation_id}",
        headers=headers,
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            return {"error": f"Status failed ({resp.status}): {text}"}
        return await resp.json()


async def _horde_cancel(session: aiohttp.ClientSession, api_key: str, generation_id: str) -> Dict[str, Any]:
    """Cancel a generation."""
    headers = {"apikey": api_key}
    async with session.delete(
        f"{HORDE_API_URL}/generate/status/{generation_id}",
        headers=headers,
    ) as resp:
        if resp.status not in (200, 404):
            return {"error": f"Cancel failed ({resp.status})"}
        return await resp.json()


async def _download_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """Download image bytes from a URL."""
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.error("Download failed: %s -> %d", url, resp.status)
                return None
            return await resp.read()
    except Exception as e:
        logger.error("Download error: %s", e)
        return None


def _is_valid_image_bytes(data: Optional[bytes]) -> bool:
    """Validate worker output is a real decodable image.

    Workers return PNG, JPEG or WebP (sometimes as CDN URLs). Rejects
    tiny garbage blobs (e.g. 92-byte corrupt responses) AND solid-color
    placeholder images (some broken workers emit a flat gray canvas).
    """
    if not data or len(data) < MIN_IMAGE_BYTES:
        return False
    try:
        import io
        from PIL import Image, ImageStat
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        # Re-open for stats (verify() invalidates the object)
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            # Downscale before stats for speed
            im.thumbnail((64, 64))
            stat = ImageStat.Stat(im)
            # stddev across channels: a flat/solid image has ~0 stddev
            if max(stat.stddev) < 5.0:
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class HordeImageGenProvider(ImageGenProvider):
    """Image generation via AI Horde distributed inference."""

    @property
    def name(self) -> str:
        return "horde"

    @property
    def display_name(self) -> str:
        return "AI Horde"

    def is_available(self) -> bool:
        # Horde works with or without an API key — anonymous mode uses "0000000000"
        return True

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "AI Horde",
            "badge": "community",
            "tag": "Decentralized inference — 50+ models (Flux, SDXL, Pony)",
            "env_vars": [
                {
                    "key": "HORDE_API_KEY",
                    "prompt": "AI Horde API key (leave blank for anonymous)",
                    "url": "https://stablehorde.net/register",
                    "optional": True,
                },
            ],
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": mid,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "category": meta.get("category", "general"),
            }
            for mid, meta in MODELS.items()
        ]

    def list_parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": key,
                "type": meta.get("type", "str"),
                "default": meta.get("default"),
                "description": meta.get("description", ""),
                "category": meta.get("category", "advanced"),
                "min": meta.get("min"),
                "max": meta.get("max"),
                "options": meta.get("options"),
            }
            for key, meta in PARAMS_METADATA.items()
        ]

    def capabilities(self) -> Dict[str, Any]:
        """Horde supports text-to-image and image-to-image (img2img/inpainting/outpainting/remix)."""
        return {
            "modalities": ["text", "image"],
            "max_reference_images": 4,
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image via Horde API.
        
        This method is synchronous from the agent's perspective —
        we do the async dance internally.

        Accepts ``aspect_ratio`` and ``model`` as direct kwargs (tool dispatch
        contract) and/or ``params`` dict (backward-compat). Any unknown kwargs
        are silently ignored.
        """
        # Merge all parameter sources: 'params' dict (old style) + direct kwargs
        p = dict(kwargs.pop("params", {}))
        # Only take aspect_ratio from the framework's default when the agent
        # (or params dict) actually specified one. The framework default
        # "landscape" must NOT silently override our square default.
        if "aspect_ratio" not in p:
            # kwargs.aspect_ratio arrives with the framework's own default
            # (landscape) — indistinguishable from an explicit agent choice
            # here, so treat anything equal to the framework default as unset.
            if aspect_ratio is not None and aspect_ratio != DEFAULT_ASPECT_RATIO:
                p["aspect_ratio"] = aspect_ratio
            # else: leave unset -> resolved to "square" below
        if "model" in kwargs:
            p.setdefault("model", kwargs.pop("model"))
        # Silently absorb any remaining kwargs (forward-compat)

        # Apply config-driven defaults (lowest priority)
        cfg_defaults = self._load_config_defaults()
        p.setdefault("steps", cfg_defaults.get("default_steps", 20))
        p.setdefault("cfg_scale", cfg_defaults.get("default_cfg_scale", 7.5))
        p.setdefault("sampler", cfg_defaults.get("default_sampler", "k_dpmpp_2m"))
        p.setdefault("base_resolution", cfg_defaults.get("default_base_resolution", 1024))
        p.setdefault("negative_prompt", cfg_defaults.get("negative_prompt", ""))
        # denoise/strength via config, NOT via tool argument.
        # 0.6 proven-working (debug payload gen_5b76116c).
        p.setdefault("denoising_strength", cfg_defaults.get("denoising_strength", 0.6))

        model_id = p.get("model", DEFAULT_MODEL)
        # Framework's resolve_aspect_ratio(None) defaults to LANDSCAPE — but
        # our tool schema defaults to square. Force square when unspecified.
        aspect = p.get("aspect_ratio") or "square"
        if aspect not in ("landscape", "square", "portrait"):
            aspect = "square"
        base_resolution = p.get("base_resolution")
        steps = p.get("steps", 25)
        cfg = p.get("cfg_scale", 7.0)
        sampler = p.get("sampler", "k_euler_a")
        seed = p.get("seed")
        negative = p.get("negative_prompt", "")

        # Compat: if user passed "prompt ### negative" inline, split it
        # (Civitai / WebUI convention) — only if no explicit negative_prompt given
        if not negative and " ### " in prompt:
            prompt, negative = prompt.split(" ### ", 1)
            prompt = prompt.strip()
            negative = negative.strip()

        # Image-to-image / img2img: download and encode source image (sync, runs in thread pool)
        source_image_b64 = None
        if image_url:
            source_image_b64 = _download_and_encode_image_sync(image_url)
        elif reference_image_urls:
            # Use first reference image as source if no explicit image_url
            source_image_b64 = _download_and_encode_image_sync(reference_image_urls[0])

        # Resolve model name for Horde API
        model_meta = MODELS.get(model_id)
        if not model_meta:
            horde_model = model_id
            model_display = model_id
        else:
            horde_model = model_meta["horde_name"]
            model_display = model_meta["display"]

        final_w, final_h = _resolve_size(base_resolution, aspect)

        # Build Horde params
        horde_params: Dict[str, Any] = {
            "cfg_scale": cfg,
            "steps": steps,
            "width": final_w,
            "height": final_h,
            "sampler_name": sampler,
            # Proven-working img2img params (from debug payload gen_5b76116c)
            "hires_fix": True,
            "hires_fix_denoising_strength": 0.75,
            "karras": True,
            "clip_skip": 2,
            "facefixer_strength": 0.5,
            "image_is_control": False,
            "return_control_map": False,
            "tiling": False,
            "transparent": False,
            "n": 1,
            "post_processing": [],
        }

        if seed is not None:
            horde_params["seed"] = str(seed)
        if negative:
            horde_params["negative_prompt"] = negative

        # Post-processing
        pp = p.get("post_processing", [])
        if pp:
            if isinstance(pp, str):
                pp = [pp]
            horde_params["post_processing"] = pp

        # Image-to-image: source_image/source_processing go at ROOT level
        # (GenerationInputStable), NOT inside params. Sending them in params
        # makes the API ignore them -> worker does txt2img from scratch.
        source_processing = p.get("source_processing", "img2img")
        root_source = {}
        if source_image_b64:
            root_source["source_image"] = source_image_b64
            root_source["source_processing"] = source_processing
            # denoising_strength goes in params (ModelGenerationInputStable)
            horde_params["denoising_strength"] = p.get("denoising_strength", 0.6)

        # Root-level params per the AI Horde v2 API spec (swagger.json).
        # Note: an unrelated "use_nsfw_censor" flag appears in the worker's
        # internal source, but it is NOT part of the public client-side
        # payload — sending it is a no-op. The actual control fields are:
        #   - nsfw: bool   — accept NSFW workers/content
        #   - censor_nsfw: bool — request the worker NOT censor (worker chooses)
        root: Dict[str, Any] = {
            "nsfw": p.get("nsfw", True),
            "censor_nsfw": p.get("censor_nsfw", False),
            # Worker-selection flags from the proven-working payload
            "trusted_workers": False,
            "validated_backends": False,
            "slow_workers": True,
            "extra_slow_workers": False,
            "workers": [],
            "worker_blacklist": False,
            "shared": False,
            "replacement_filter": True,
            "allow_downgrade": False,
            "disable_batching": False,
            "r2": True,
            **root_source,
        }

        # Get API key
        api_key = self._load_api_key()
        logger.info(
            "Horde generate: model=%s prompt=%.80s size=%dx%d steps=%d cfg=%.1f",
            model_display, prompt, final_w, final_h, steps, cfg,
        )

        # Run async generation in a dedicated thread with its own event loop
        # to avoid "Event loop is closed" errors when the gateway already has
        # a running loop in the main thread.
        import concurrent.futures

        def _run_async() -> Dict[str, Any]:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self._generate_async(api_key, prompt, horde_model, horde_params, root)
                )
            finally:
                new_loop.close()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(_run_async).result()
            return result
        except Exception as exc:
            logger.error("Horde generation failed", exc_info=True)
            return error_response(
                error=f"Horde generation failed: {exc}",
                error_type="api_error",
                provider="horde",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

    # ------------------------------------------------------------------
    # Internal async orchestration
    # ------------------------------------------------------------------

    async def _generate_async(
        self,
        api_key: str,
        prompt: str,
        horde_model: str,
        horde_params: Dict[str, Any],
        root_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Full async generation lifecycle: submit → poll → download → save.
        
        Creates its own aiohttp session to ensure thread-safety — each
        generation runs in its own thread with its own event loop, so the
        session must not be shared across threads.
        """
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:

            # 1. Submit
            gen_resp = await _horde_generate(session, api_key, prompt, horde_model, horde_params, root_params)
            if "error" in gen_resp:
                return error_response(
                    error=f"Horde API error: {gen_resp['error']}",
                    error_type="api_error",
                    provider="horde",
                    model=horde_model,
                    prompt=prompt,
                )

            generation_id = gen_resp.get("id")
            if not generation_id:
                return error_response(
                    error="Horde returned no generation ID",
                    error_type="empty_response",
                    provider="horde",
                    model=horde_model,
                    prompt=prompt,
                )

            # 2. Poll until done
            deadline = time.monotonic() + DEFAULT_TIMEOUT
            while time.monotonic() < deadline:
                check = await _horde_check(session, api_key, generation_id)
                if "error" in check:
                    return error_response(
                        error=f"Status check error: {check['error']}",
                        error_type="api_error",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                if check.get("done"):
                    break
                await asyncio.sleep(POLL_INTERVAL)
            else:
                # Timeout — try to cancel
                await _horde_cancel(session, api_key, generation_id)
                return error_response(
                    error=f"Generation timed out after {DEFAULT_TIMEOUT}s",
                    error_type="timeout",
                    provider="horde",
                    model=horde_model,
                    prompt=prompt,
                )

            # 3. Get result
            result = await _horde_status(session, api_key, generation_id)
            if "error" in result:
                return error_response(
                    error=f"Result fetch error: {result['error']}",
                    error_type="api_error",
                    provider="horde",
                    model=horde_model,
                    prompt=prompt,
                )

            generations = result.get("generations", [])
            if not generations:
                return error_response(
                    error="Horde returned no images",
                    error_type="empty_response",
                    provider="horde",
                    model=horde_model,
                    prompt=prompt,
                )

            # 4. Detect worker-side censorship BEFORE downloading the (censored) image.
            # Per AI Horde v2 API spec (swagger.json), a generation object can signal
            # censorship via three fields (priority order):
            #   1. gen_metadata[] with type="censorship", value="csam" -> child-safety, NEVER retry
            #   2. gen_metadata[] with type="censorship", value in {"nsfw","censorlist",...} -> NSFW filter, retry
            #   3. censored: bool (in GenerationStable) -> true means worker replaced image, retry
            #   4. state: "ok" | "censored" (in Generation, OBSOLETE) -> "censored" means retry
            #
            # We check in order 1->2->3->4. gen_metadata is authoritative.
            first_gen = generations[0]

            # Helper: extract censorship info from a generation dict
            def _censorship_info(gen: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
                """
                Returns (censorship_type, worker_name) where censorship_type is:
                  "csam"       -> child-safety, non-retryable
                  "nsfw"       -> NSFW filter (nsfw, censorlist, baseline_mismatch, etc.)
                  "unknown"    -> censored=true or state="censored" but no gen_metadata
                  None         -> clean
                """
                # 1. gen_metadata is authoritative
                for meta in gen.get("gen_metadata", []):
                    if meta.get("type") == "censorship":
                        val = meta.get("value")
                        if val == "csam":
                            return "csam", gen.get("worker_name")
                        if val in ("nsfw", "censorlist", "baseline_mismatch", "see_ref"):
                            return "nsfw", gen.get("worker_name")
                        # Any other censorship value -> treat as nsfw (retryable)
                        return "nsfw", gen.get("worker_name")
                # 2. censored bool (GenerationStable extension)
                if gen.get("censored") is True:
                    return "unknown", gen.get("worker_name")
                # 3. state field (Generation, OBSOLETE)
                if gen.get("state") == "censored":
                    return "unknown", gen.get("worker_name")
                return None, None

            # Check initial generation
            ctype, cworker = _censorship_info(first_gen)
            if ctype == "csam":
                logger.warning(
                    "Horde generation flagged CSAM by worker %s; aborting (no retry)",
                    cworker or "?",
                )
                return error_response(
                    error=(
                        "Generation aborted by worker child-safety filter (gen_metadata=csam). "
                        "This is non-retryable."
                    ),
                    error_type="csam_rejected",
                    provider="horde",
                    model=horde_model,
                    prompt=prompt,
                )

            censored_attempts: List[str] = []
            while ctype in ("nsfw", "unknown"):
                censored_attempts.append(cworker or "?")
                logger.warning(
                    "Horde censored by worker %s on model %s (type=%s, attempt %d/%d)",
                    cworker or "?",
                    horde_model,
                    ctype,
                    len(censored_attempts),
                    MAX_CENSORED_RETRIES,
                )
                if len(censored_attempts) >= MAX_CENSORED_RETRIES:
                    return error_response(
                        error=(
                            f"Generation censored by {len(censored_attempts)} consecutive "
                            f"workers on model {horde_model}: {censored_attempts}. "
                            f"No more retries."
                        ),
                        error_type="censored_after_retry",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                # Re-submit with the same model — different worker pool
                retry_resp = await _horde_generate(
                    session, api_key, prompt, horde_model, horde_params, root_params,
                )
                if "error" in retry_resp:
                    return error_response(
                        error=f"Horde retry submit failed: {retry_resp['error']}",
                        error_type="api_error",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                retry_id = retry_resp.get("id")
                if not retry_id:
                    return error_response(
                        error="Horde retry returned no generation ID",
                        error_type="empty_response",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                # Poll
                deadline_r = time.monotonic() + DEFAULT_TIMEOUT
                timed_out = False
                while time.monotonic() < deadline_r:
                    chk = await _horde_check(session, api_key, retry_id)
                    if "error" in chk:
                        return error_response(
                            error=f"Horde retry status check error: {chk['error']}",
                            error_type="api_error",
                            provider="horde",
                            model=horde_model,
                            prompt=prompt,
                        )
                    if chk.get("done"):
                        break
                    await asyncio.sleep(POLL_INTERVAL)
                else:
                    timed_out = True
                if timed_out:
                    await _horde_cancel(session, api_key, retry_id)
                    return error_response(
                        error=f"Horde retry timed out after {DEFAULT_TIMEOUT}s",
                        error_type="timeout",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                # Status
                st = await _horde_status(session, api_key, retry_id)
                if "error" in st:
                    return error_response(
                        error=f"Horde retry result fetch error: {st['error']}",
                        error_type="api_error",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                rgens = st.get("generations", [])
                if not rgens:
                    return error_response(
                        error="Horde retry returned no images",
                        error_type="empty_response",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                rfirst = rgens[0]
                # Re-evaluate censorship on the retry
                ctype, cworker = _censorship_info(rfirst)
                if ctype == "csam":
                    return error_response(
                        error="Retry aborted by worker child-safety filter (gen_metadata=csam)",
                        error_type="csam_rejected",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                # If still censored (nsfw/unknown), loop again. Otherwise accept.
                if ctype in ("nsfw", "unknown"):
                    first_gen = rfirst
                    continue
                # Success on retry
                first_gen = rfirst
                logger.info(
                    "Horde retry succeeded after %d censored attempt(s); new worker=%s",
                    len(censored_attempts),
                    rfirst.get("worker_name", "?"),
                )
                break

            # 4b. Download the first image
            img_url = first_gen.get("img")
            if not img_url:
                return error_response(
                    error="Generation missing image URL",
                    error_type="empty_response",
                    provider="horde",
                    model=horde_model,
                    prompt=prompt,
                )

            image_bytes = await _download_image(session, img_url)
            if image_bytes is None:
                return error_response(
                    error="Failed to download generated image",
                    error_type="io_error",
                    provider="horde",
                    model=horde_model,
                    prompt=prompt,
                )

            # 4c. Validate downloaded bytes — a worker may return a "done" job
            # with garbage (not an image). Re-submit to a different worker up to
            # MAX_BROKEN_RETRIES times before giving up.
            broken_attempts: List[str] = []
            while not _is_valid_image_bytes(image_bytes):
                broken_attempts.append(first_gen.get("worker_name", "?"))
                logger.warning(
                    "Horde worker %s returned invalid image bytes (%d B, attempt %d/%d); re-submitting",
                    broken_attempts[-1],
                    len(image_bytes or b""),
                    len(broken_attempts),
                    MAX_BROKEN_RETRIES,
                )
                if len(broken_attempts) >= MAX_BROKEN_RETRIES:
                    return error_response(
                        error=(
                            f"Generation returned invalid image data {len(broken_attempts)} "
                            f"consecutive times ({broken_attempts}); giving up."
                        ),
                        error_type="bad_worker_output",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                # Re-submit same request, Horde routes to a different worker.
                # worker_blacklist is a ROOT bool ("avoid bad/known workers"),
                # NOT a list — sending a list yields 400 validation errors.
                retry_root = {**root_params, "worker_blacklist": True}
                retry_resp = await _horde_generate(
                    session, api_key, prompt, horde_model, horde_params, retry_root
                )
                if "error" in retry_resp or "id" not in retry_resp:
                    return error_response(
                        error=f"Horde re-submit after bad worker failed: {retry_resp.get('error', retry_resp)}",
                        error_type="api_error",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                retry_id = retry_resp["id"]
                # Poll until done
                deadline_r = time.monotonic() + DEFAULT_TIMEOUT
                done_r = False
                while time.monotonic() < deadline_r:
                    chk = await _horde_check(session, api_key, retry_id)
                    if "error" in chk:
                        break
                    if chk.get("done"):
                        done_r = True
                        break
                    await asyncio.sleep(POLL_INTERVAL)
                if not done_r:
                    await _horde_cancel(session, api_key, retry_id)
                    return error_response(
                        error="Horde re-submit after bad worker timed out",
                        error_type="timeout",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                st_r = await _horde_status(session, api_key, retry_id)
                rgens_r = st_r.get("generations", [])
                if not rgens_r:
                    return error_response(
                        error="Horde re-submit returned no generations",
                        error_type="empty_response",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                first_gen = rgens_r[0]
                img_url = first_gen.get("img")
                if not img_url:
                    return error_response(
                        error="Re-submit generation missing image URL",
                        error_type="empty_response",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
                image_bytes = await _download_image(session, img_url)
                if image_bytes is None:
                    return error_response(
                        error="Failed to download re-submitted image",
                        error_type="io_error",
                        provider="horde",
                        model=horde_model,
                        prompt=prompt,
                    )
            if broken_attempts:
                logger.info(
                    "Horde recovered after %d bad-worker attempt(s); new worker=%s",
                    len(broken_attempts),
                    first_gen.get("worker_name", "?"),
                )

            # 5. Save to cache
            try:
                from agent.image_gen_provider import _images_cache_dir
                cache_dir = _images_cache_dir()
                ts = time.strftime("%Y%m%d_%H%M%S")
                short = uuid.uuid4().hex[:8]
                seed_used = first_gen.get("seed", "?")
                ext = "png"
                safe_model = horde_model.replace(" ", "_").replace("/", "_")
                filename = f"horde_{safe_model}_{ts}_{short}.{ext}"
                path = cache_dir / filename
                path.write_bytes(image_bytes)
                logger.info("Image saved: %s (%d bytes, seed=%s)", path, len(image_bytes), seed_used)
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="horde",
                    model=horde_model,
                    prompt=prompt,
                )

            # 6. Build response
            extra: Dict[str, Any] = {
                "size": f"{first_gen.get('width', '?')}x{first_gen.get('height', '?')}",
                "seed": seed_used,
            }
            if first_gen.get("model"):
                extra["worker_model"] = first_gen["model"]

            # Report aspect from the REAL saved image (workers can ignore the
            # requested size; don't lie in the report like None==None would)
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(path) as _im:
                    _w, _h = _im.size
                _real_aspect = "square" if _w == _h else ("landscape" if _w > _h else "portrait")
            except Exception:
                _real_aspect = "square"
            return success_response(
                image=str(path),
                model=horde_model,
                prompt=prompt,
                aspect_ratio=_real_aspect,
                provider="horde",
                extra=extra,
            )

    # ------------------------------------------------------------------
    # Config defaults from config.yaml
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config_defaults() -> Dict[str, Any]:
        """Read user-configured defaults from ``image_gen.horde`` in config.yaml.
        
        Supported keys (all optional):
          default_steps, default_cfg_scale, default_sampler, negative_prompt,
          denoising_strength
        """
        defaults: Dict[str, Any] = {}
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            section = (cfg or {}).get("image_gen", {})
            if isinstance(section, dict):
                horde_cfg = section.get("horde", {})
                if isinstance(horde_cfg, dict):
                    for key in ("default_steps", "default_cfg_scale",
                                "default_sampler", "negative_prompt",
                                "denoising_strength"):
                        if key in horde_cfg and horde_cfg[key] is not None:
                            defaults[key] = horde_cfg[key]
        except Exception:
            pass
        return defaults

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_api_key() -> str:
        """Load Horde API key from env / config.

        Priority:
        1. ``HORDE_API_KEY`` env var (also supports ``AI_HORDE_API_KEY``)
        2. ``image_gen.horde.api_key`` in config.yaml
        """
        env_key = os.environ.get("HORDE_API_KEY") or os.environ.get("AI_HORDE_API_KEY")
        if env_key:
            return env_key

        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            section = (cfg or {}).get("image_gen", {})
            if isinstance(section, dict):
                horde_cfg = section.get("horde", {})
                if isinstance(horde_cfg, dict):
                    key = horde_cfg.get("api_key")
                    if key:
                        return key
        except Exception:
            pass

        logger.info("No Horde API key found — using anonymous mode")
        return "0000000000"


# ---------------------------------------------------------------------------
# Plugin entry point + custom tool replacement
# ---------------------------------------------------------------------------

# Schema for our replacement image_generate tool
CUSTOM_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "Generate an image using AI Horde distributed inference. "
        "⚠️ CRITICAL: You MUST write prompts in ENGLISH, be highly detailed, "
        "and rich in contextual depth. Include: subject appearance, setting, "
        "lighting, mood, colors, composition, style, camera angle, and any "
        "specific visual elements. POOR EXAMPLE: 'a cat'. GOOD EXAMPLE: "
        "'A majestic orange tabby cat with emerald-green eyes lounging on a "
        "velvet crimson chaise lounge in a dimly lit Victorian library, "
        "warm golden sunlight streaming through a arched stained-glass window, "
        "casting colorful patterns across the room, digital painting by "
        "James Gurney, exquisite detail, volumetric lighting, rich textures, "
        "8k resolution, cinematic composition'."
    ),
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "A HIGHLY DETAILED image description in ENGLISH. Include "
                "subject, setting, lighting, mood, colors, composition, style, "
                "and visual details. Longer, richer prompts produce better results."
            ),
        },
        "model": {
            "type": "string",
            "description": (
                "Horde model ID. Best default: sdxl:1.0 (fast, reliable). "
                "Options: sdxl:1.0, sdxl:albedo, sdxl:anime, sd15, deliberate, "
                "icbinp, dreamshaper, pony:realistic, pony:anime, pony:amp, "
                "juggernaut-xl, flux"
            ),
            "default": DEFAULT_MODEL,
        },
        "aspect_ratio": {
            "type": "string",
            "enum": ["square", "landscape", "portrait"],
            "default": "square",
            "description": "Image aspect ratio. square=1024x1024, landscape=1344x768, portrait=768x1344",
        },
        "steps": {
            "type": "integer",
            "default": 25,
            "minimum": 1,
            "maximum": 100,
            "description": "Sampling steps. Higher = more detail, slower.",
        },
        "cfg_scale": {
            "type": "number",
            "default": 7.0,
            "minimum": 1.0,
            "maximum": 30.0,
            "description": "Guidance scale. Higher = prompt adherence, lower = creativity.",
        },
        "sampler": {
            "type": "string",
            "enum": [
                "k_lms", "k_heun", "k_euler", "k_euler_a",
                "k_dpm_2", "k_dpm_2_a", "k_dpmpp_2s_a", "k_dpmpp_2m",
                "k_dpmpp_sde", "lcm", "DDIM",
            ],
            "default": "k_euler_a",
            "description": "Sampling method for denoising.",
        },
        "seed": {
            "type": "integer",
            "description": "Random seed for reproducibility. Omit for random.",
        },
        "negative_prompt": {
            "type": "string",
            "default": "",
            "description": "Things to avoid in the image. Comma-separated.",
        },
    },
    "required": ["prompt"],
}


def _handle_custom_generate(args: Dict[str, Any], **kw: Any) -> str:
    """Handler for our custom image_generate tool."""
    import json
    try:
        from agent.image_gen_registry import get_active_provider

        provider = get_active_provider()
        if provider is None:
            return json.dumps({
                "success": False,
                "image": None,
                "error": "No image generation provider is active.",
                "error_type": "no_provider",
            })

        prompt = args.get("prompt", "")
        if not prompt:
            return json.dumps({
                "success": False, "image": None,
                "error": "prompt is required", "error_type": "missing_param",
            })

        # Merge all args into kwargs for the provider
        kwargs = {"prompt": prompt}
        for key in ("model", "aspect_ratio", "steps", "cfg_scale",
                     "sampler", "seed", "negative_prompt",
                     "image_url", "reference_image_urls", "denoising_strength", "source_processing"):
            if key in args and args[key] is not None:
                kwargs[key] = args[key]

        result = provider.generate(**kwargs)
        if not isinstance(result, dict):
            return json.dumps({
                "success": False, "image": None,
                "error": "Provider returned non-dict", "error_type": "contract",
            })
        return json.dumps(result)
    except Exception as exc:
        logger.error("Custom image_generate handler failed", exc_info=True)
        return json.dumps({
            "success": False, "image": None,
            "error": f"Generation failed: {exc}", "error_type": "handler_error",
        })


def register(ctx) -> None:
    """Register provider and replace stock image_generate with our enhanced tool."""
    # 1. Register the provider
    ctx.register_image_gen_provider(HordeImageGenProvider())
    logger.info("Horde image gen provider registered")

    # 2. Deregister the stock image_generate tool
    try:
        from tools.registry import registry
        registry.deregister("image_generate")
        logger.info("Stock image_generate tool deregistered")
    except Exception as exc:
        logger.warning("Could not deregister stock tool: %s", exc)

    # 3. Register our enhanced replacement
    ctx.register_tool(
        name="image_generate",
        toolset="image_gen",
        schema=CUSTOM_TOOL_SCHEMA,
        handler=_handle_custom_generate,
        emoji="🎨",
        description=CUSTOM_TOOL_SCHEMA["description"],
    )
    logger.info("Custom image_generate tool registered")


def reload_plugin() -> str:
    """Hot-reload the Horde plugin module and re-register the provider.
    
    Call this after editing the plugin source to apply changes without
    restarting the gateway:
    
        python3 -c "import sys; sys.path.insert(0, '~/.hermes/plugins/image_gen/horde'); from horde_plugin import reload_plugin; print(reload_plugin())"
    
    Or from Python:
    
        from hermes_cli.plugins import get_plugin_manager
        pm = get_plugin_manager()
        pm.discover_and_load(force=True)
        from agent.image_gen_registry import register_provider, get_provider
        import importlib
        import sys
        for name, mod in list(sys.modules.items()):
            if mod and hasattr(mod, '__file__') and 'image_gen/horde/__init__' in str(mod.__file__):
                importlib.reload(mod)
                if hasattr(mod, 'HordeImageGenProvider'):
                    register_provider(mod.HordeImageGenProvider())
                    return f"✅ Horde plugin reloaded: {get_provider('horde')}"
        return "❌ Horde plugin module not found in sys.modules"
    """
    import importlib
    import sys

    from agent.image_gen_registry import register_provider

    target_marker = "image_gen" + "/" + "horde" + "/" + "__init__"
    for name, mod in list(sys.modules.items()):
        if mod and hasattr(mod, "__file__") and mod.__file__ and target_marker in mod.__file__:
            importlib.reload(mod)
            if hasattr(mod, "HordeImageGenProvider"):
                register_provider(mod.HordeImageGenProvider())
                from agent.image_gen_registry import get_provider

                p = get_provider("horde")
                logger.info("Horde plugin hot-reloaded: %s", p)
                return f"✅ Horde plugin reloaded: {p}"
            return "❌ Reloaded module has no HordeImageGenProvider"
    return "❌ Horde plugin module not found in sys.modules. Restart the gateway."
