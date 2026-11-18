---
title: Flan T5 Asl Gloss2pose
emoji: 🤟
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: English to ASL gloss to signing skeleton (API)
---

# flan-t5-asl-gloss2pose (backend)

Gradio app on ZeroGPU for the Never Silent demo. It translates English into ASL gloss with a fine-tuned
`google/flan-t5-base`, then joins pre-recorded MediaPipe `.pose` clips from
[SignBridge](https://github.com/SignBridgeApp) (MIT licence) into one signing animation, returned as JSON.

Only `model.generate` uses the GPU (`@spaces.GPU(duration=15)`); if no GPU is available it falls back to CPU.
Model: `MODEL_ID` variable, default `noxy28/flan-t5-base-aslg`.

API (see "Use via API" at the bottom of the Space page):

| Endpoint     | Input   | Output        |
|--------------|---------|---------------|
| `/translate` | `text`  | response JSON |
| `/sign`      | `gloss` | response JSON |
