# incoming-video

Auto-processes videos that arrive via Telegram (or any platform) **before the LLM sees them**, so the agent doesn't have to transcribe or extract frames manually — saving tokens and keeping the turn focused on the user's intent.

## What it does

When the user sends a video, the gateway caches it and injects a note like:

```
[The user sent a video attachment: 'video_abc123'. It is saved at: /root/.hermes/cache/videos/video_abc123.mp4 ...]
```

This plugin's `pre_llm_call` hook intercepts that note and, in one pass:

1. **Extracts the video path** from the attachment note (regex + path validation).
2. **Transcribes the audio** with the configured STT provider — `transcribe_audio()` from `tools.transcription_tools`, which honors the `stt.provider` setting in `config.yaml` (openproxy/whisper, local, etc.). No hardcoded local whisper.
3. **Extracts frames** with `ffmpeg` (count scales by duration: 1 frame for short clips, up to 8 for long ones), persists them as `<video>_frame_NN.png` next to the source in `~/.hermes/cache/videos/`, and builds a contact sheet `grid_<video>.jpg` for quick visual scanning.
4. **Injects the result** into the turn context via `{"context": "[Transcripción del vídeo]\n...\n[Frames extraídos del vídeo] ..."}`.

The agent then sees the transcription and frame paths as normal context — no manual transcriptioning, no token waste.

## Install

```bash
# from the repo root
ln -s "$PWD/plugins/incoming-video" ~/.hermes/plugins/incoming-video
# enable in config.yaml
#   plugins:
#     enabled: [..., incoming-video]
# restart the gateway
sudo systemctl restart hermes-gateway
```

## Requirements

- `ffmpeg` / `ffprobe` on PATH (gateway's systemd unit already includes `/usr/bin`).
- A configured STT provider (`stt.provider` in `config.yaml`) — e.g. openproxy whisper.
- The plugin runs in the gateway process, so it has access to `tools.transcription_tools`.

## For AI agents configuring this plugin

If you (an agent) are asked to install this plugin on a Hermes installation, follow this protocol exactly:

1. **Check the target layout.** Hermes discovers plugins in `~/.hermes/plugins/<name>/`. The canonical source is this repo (`plugins/incoming-video/`). The standard install is a **symlink**, not a copy:

   ```bash
   ln -s "$PWD/plugins/incoming-video" ~/.hermes/plugins/incoming-video
   ```

2. **Enable it** in `~/.hermes/config.yaml`:

   ```yaml
   plugins:
     enabled:
       - incoming-video
   ```

3. **Verify prerequisites** before restarting:
   - `ffmpeg` and `ffprobe` must be on the gateway's PATH (systemd unit `Environment=PATH=...` — `/usr/bin` is enough).
   - `stt.provider` must be set in `config.yaml` (e.g. `openproxy`). If it isn't, the transcription step fails and the hook degrades to frames-only.

4. **Restart the gateway** so the plugin loads:

   ```bash
   sudo systemctl restart hermes-gateway
   ```

5. **Verify it loaded**:

   ```bash
   hermes plugins list | grep incoming-video
   # expect: incoming-video │ enabled │ 1.0.0
   ```

6. **Smoke test** (optional but recommended): send a short video via the platform and check `~/.hermes/cache/videos/` for `<video>_frame_01.png` and the injected `[Transcripción del vídeo]` block in the agent's context.

**If the user is on a systemd gateway, never restart it yourself from inside the agent** — it kills the session. Tell the user to run the restart.

## Known limitations

- **Synchronous hook**: `pre_llm_call` is sync, so the gateway blocks while transcribing (30–60s for a 2-min clip). Acceptable in DMs since the gateway serializes messages anyway.
- **Cache growth**: frames persist in `~/.hermes/cache/videos/` — a future disk-cleanup pass should prune old `frame_*`/`grid_*`/`audio.wav` artifacts.

## Files

- `plugin.yaml` — manifest (`kind: standalone`, `provides_hooks: [pre_llm_call]`)
- `__init__.py` — the hook logic (pure stdlib + ffmpeg + `transcribe_audio`)
