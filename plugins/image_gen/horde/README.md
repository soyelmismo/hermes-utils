# Horde — Hermes Image Generation Plugin

An [AI Horde](https://aihorde.net) backend for Hermes Agent's `image_generate` tool. Generates images via a **distributed inference cluster** — 50+ community-hosted models including Flux, SDXL, Pony, SD15, and many more.

No GPU required on your machine. Your request joins a global queue of workers running on volunteer GPUs.

## How it works

1. Agent calls `image_generate(prompt="...", aspect_ratio="...")`
2. Plugin POSTs to Horde API → gets a job ID
3. Polls Horde status until the image is ready
4. Downloads the image and returns it to the agent

## Install

```bash
# Symlink into Hermes plugins
ln -sf /path/to/hermes-utils/plugins/image_gen/horde ~/.hermes/plugins/image_gen/horde

# Enable in ~/.hermes/config.yaml
plugins:
  enabled:
    - image_gen/horde
```

**Optional:** Set your Horde API key. The plugin checks in this order:

1. `HORDE_API_KEY` env var in `~/.hermes/.env`
2. `AI_HORDE_API_KEY` env var (alternative name)
3. `image_gen.horde.api_key` in `~/.hermes/config.yaml`

```bash
# Option A: env var in .env
echo "HORDE_API_KEY=your_api_key_here" >> ~/.hermes/.env

# Option B: config.yaml
hermes config set image_gen.horde.api_key "your_api_key_here"
```

Without an API key, the Horde assigns an anonymous key (`0000000000`). This works but may have longer queue times.

Set your preferred model (default: `sdxl:1.0`):

```bash
hermes config set image_gen.horde.model "sdxl:1.0"
```

## Configuration

| Setting | Default | Description |
|---|---|---|
| `image_gen.horde.model` | `sdxl:1.0` | Model name (see [Horde models list](https://aihorde.net/api/v2/status/models)) |
| `image_gen.horde.api_key` | — | API key (alternative to `HORDE_API_KEY` env var) |
| `HORDE_API_KEY` | (anonymous) | Env var — checked first, overrides config |

## Censorship handling

AI Horde workers can refuse to return a generated image when their post-generation NSFW detector trips. The worker signals this in the response by setting `state: "censored"` on the generation object (and replaces the image with a censored PNG).

This plugin detects that and **retries automatically**:

- On `state: "censored"` → the same request is re-submitted to the Horde. A different worker in the pool will likely not have the same filter active.
- On `state: "csam"` (child-safety) → the request is **aborted immediately**. This is non-retryable.
- After **3 consecutive censored responses** with no clean one, the plugin returns an error with the list of workers that censored.

The user's `model` choice is **always preserved** — retries use the same model, just land on a different worker.

Tune `MAX_CENSORED_RETRIES` in `__init__.py` if you need a different ceiling.

## Available models (small sample)

Flux.1, SDXL, Pony Diffusion, SD 1.5, Anything V5, Realistic Vision, DreamShaper, and 40+ more. Full list at [aihorde.net/api/v2/status/models](https://aihorde.net/api/v2/status/models).

## Notes

- Generation time depends on queue depth (usually 10–60 seconds)
- The plugin uses `ThreadPoolExecutor` for non-blocking polling
- Each image is cached to `~/.hermes/cache/image_gen/`

## Give back: run a worker

The AI Horde is not just a service you consume — it's a **community-powered network**. Every image you generate runs on someone else's GPU. If you have a modest GPU sitting idle, you can return the favour.

### Why run a worker

- **You help keep the Horde free.** Every worker added means shorter queues for everyone, including yourself.
- **You earn Kudos.** The priority currency of the Horde. Run a worker and your own future requests jump the queue — you generate faster because you contributed compute.
- **It's set-and-forget.** The worker software idles until the Horde sends a job, processes it, and goes back to idle. No incoming ports, no security exposure — only outbound connections.
- **You already have the hardware.** If you have an NVIDIA GPU with 6 GB+ VRAM (or an AMD GPU with ROCm support), you can run an image worker today.

### Minimum requirements (image worker)

| Requirement | Minimum | Recommended |
|---|---|---|
| VRAM | 6 GB (SD 1.5) | 12 GB (SDXL) / 16 GB (Flux) |
| GPU | NVIDIA CUDA or AMD ROCm | NVIDIA RTX 3060+ |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 50 GB+ (multiple models) |
| Software | [AI-Horde-Worker](https://github.com/Haidra-Org/AI-Horde-Worker) | Same + GPU drivers |
| Internet | Broadband (outbound only) | Broadband |

### Quick start

```bash
# 1. Register at https://aihorde.net to get your API key

# 2. Install the worker
git clone https://github.com/Haidra-Org/AI-Horde-Worker.git
cd AI-Horde-Worker
pip install -r requirements.txt

# 3. Set your API key
export AIWORKER_API_KEY=your_api_key_here

# 4. Run the worker
python run.py
```

The worker polls for jobs, downloads models on demand, generates images, and goes back to idle. Stop it anytime — there are no commitments.

### Worker types

| Type | What it does | Model examples |
|---|---|---|
| **Dreamer** (image) | Generates images from prompts | Flux.1, SDXL, SD 1.5, Pony |
| **Scribe** (text) | Runs LLM inference | Llama, Mistral, etc. |
| **Alchemist** (post) | Upscaling, face fixing, captioning | ESRGAN, GFPGAN |

Image workers (Dreamers) are in highest demand and earn Kudos fastest.

### The philosophy

The Horde exists because a lot of people with spare GPU cycles decided to share them. It's a gift economy — you give compute when you can, you consume compute when you need. No contracts, no billing, no gatekeeping. If your GPU is idle for 8 hours while you sleep or work, those are 8 hours it could be helping someone else create.

> *"Volunteers share spare computer power so anyone can generate images and text."* — AI Horde

**Not able to run a worker?** That's fine. You can support the Horde financially at [aihorde.net/contribute/donate](https://aihorde.net/contribute/donate), or simply by spreading the word.

## License

MIT — part of [hermes-utils](https://github.com/soyelmismo/hermes-utils)
