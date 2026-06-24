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

DEFAULT_MODEL = "sdxl:1.0"

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
    "width": {
        "type": "int",
        "default": 768,
        "min": 64,
        "max": 2048,
        "description": "Image width. Must be multiple of 64. If not specified, aspect ratio defaults apply (landscape=1152x768, square=768x768, portrait=768x1152).",
        "category": "basic",
    },
    "height": {
        "type": "int",
        "default": 768,
        "min": 64,
        "max": 2048,
        "description": "Image height. Must be multiple of 64.",
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
}

ASPECT_SIZES = {
    "landscape": (1152, 768),
    "square": (768, 768),
    "portrait": (768, 1152),
}


def _nearest_multiple(value: int, multiple: int = 64) -> int:
    """Round to nearest multiple of 64."""
    return max(64, round(value / multiple) * multiple)


def _resolve_size(width: Optional[int], height: Optional[int], aspect: str) -> Tuple[int, int]:
    """Resolve final dimensions from explicit values or aspect ratio."""
    if width and height:
        return _nearest_multiple(width), _nearest_multiple(height)
    default_w, default_h = ASPECT_SIZES.get(aspect, ASPECT_SIZES["square"])
    return default_w, default_h


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

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
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
        p.setdefault("aspect_ratio", aspect_ratio)
        if "model" in kwargs:
            p.setdefault("model", kwargs.pop("model"))
        # Silently absorb any remaining kwargs (forward-compat)

        # Apply config-driven defaults (lowest priority)
        cfg_defaults = self._load_config_defaults()
        p.setdefault("steps", cfg_defaults.get("default_steps", 25))
        p.setdefault("cfg_scale", cfg_defaults.get("default_cfg_scale", 7.0))
        p.setdefault("sampler", cfg_defaults.get("default_sampler", "k_euler_a"))
        p.setdefault("negative_prompt", cfg_defaults.get("negative_prompt", ""))

        model_id = p.get("model", DEFAULT_MODEL)
        aspect = resolve_aspect_ratio(p.get("aspect_ratio"))
        width = p.get("width")
        height = p.get("height")
        steps = p.get("steps", 25)
        cfg = p.get("cfg_scale", 7.0)
        sampler = p.get("sampler", "k_euler_a")
        seed = p.get("seed")
        negative = p.get("negative_prompt", "")

        # Resolve model name for Horde API
        model_meta = MODELS.get(model_id)
        if not model_meta:
            # Try as direct horde model name
            horde_model = model_id
            model_display = model_id
        else:
            horde_model = model_meta["horde_name"]
            model_display = model_meta["display"]

        # Resolve dimensions
        final_w, final_h = _resolve_size(width, height, aspect)

        # Build Horde params
        horde_params: Dict[str, Any] = {
            "cfg_scale": cfg,
            "steps": steps,
            "width": final_w,
            "height": final_h,
            "sampler_name": sampler,
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

        # Root-level params per the AI Horde v2 API spec (swagger.json).
        # Note: an unrelated "use_nsfw_censor" flag appears in the worker's
        # internal source, but it is NOT part of the public client-side
        # payload — sending it is a no-op. The actual control fields are:
        #   - nsfw: bool   — accept NSFW workers/content
        #   - censor_nsfw: bool — request the worker NOT censor (worker chooses)
        root: Dict[str, Any] = {
            "nsfw": p.get("nsfw", True),
            "censor_nsfw": p.get("censor_nsfw", False),
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

            return success_response(
                image=str(path),
                model=horde_model,
                prompt=prompt,
                aspect_ratio=resolve_aspect_ratio(
                    "square" if first_gen.get("width") == first_gen.get("height") else
                    "landscape" if first_gen.get("width", 0) > first_gen.get("height", 0) else
                    "portrait"
                ),
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
          default_steps, default_cfg_scale, default_sampler, negative_prompt
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
                                "default_sampler", "negative_prompt"):
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
                     "sampler", "seed", "negative_prompt"):
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
