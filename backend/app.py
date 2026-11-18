"""Never Silent backend: English -> ASL gloss (fine-tuned FLAN-T5-base) -> signing skeleton as JSON.

Gradio Space on ZeroGPU. Only model.generate runs on the GPU; sign planning and pose building stay on CPU.
"""
import os
import re

import spaces  # must be imported before torch touches CUDA on ZeroGPU
import torch
import gradio as gr
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import signing

MAX_CHARS = 300
MODEL_ID = os.environ.get("MODEL_ID", "").strip() or "noxy28/flan-t5-base-aslg"

# Decoding settings copied from the notebook's CFG (Part 0.3): identical for every model.
CFG = {
    "prefix": "translate English to ASL gloss: ",
    "max_len": 128,
    "num_beams": 4,
    "max_new_tokens": 128,
    "gen_batch_size": 32,
}


# Copied from the notebook (Part 0.5): the only function used to generate gloss.
def translate(model, tokenizer, texts, batch_size=None):
    """Translate a list of English sentences into ASL gloss with the shared decoding settings."""
    batch_size = batch_size or CFG["gen_batch_size"]
    model.eval()
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch = [CFG["prefix"] + t for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=CFG["max_len"]).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=CFG["num_beams"], max_new_tokens=CFG["max_new_tokens"])
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return [p.strip() for p in preds]


def normalise(sentence):
    """Match the corpus format: lowercase, with punctuation split off ("agreement ." not "agreement.")."""
    return re.sub(r"\s*([.,!?;:])", r" \1", sentence.lower()).strip()


# Load once at startup. On ZeroGPU the GPU copy is placed on "cuda" at module level, as the docs require;
# the CPU copy is the fallback when no GPU can be had (quota used up, queue timeout, ...).
ON_ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
cpu_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).eval()
gpu_model = None
if ON_ZERO_GPU or torch.cuda.is_available():
    gpu_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to("cuda").eval()
print(f"Ready: model={MODEL_ID} gpu={'zerogpu' if ON_ZERO_GPU else gpu_model is not None} "
      f"signs={len(signing.SIGN_WORDS)}", flush=True)


@spaces.GPU(duration=15)
def _generate_gpu(model_input):
    return translate(gpu_model, tokenizer, [model_input])[0]


def generate(model_input):
    if gpu_model is not None:
        try:
            return _generate_gpu(model_input)
        except Exception as e:
            print(f"GPU generation failed, using CPU instead: {e!r}", flush=True)
    return translate(cpu_model, tokenizer, [model_input])[0]


def _check(value, field):
    value = (value or "").strip()
    if not value:
        raise gr.Error(f"'{field}' is empty.")
    if len(value) > MAX_CHARS:
        raise gr.Error(f"'{field}' is longer than {MAX_CHARS} characters.")
    return value


def translate_api(text: str) -> dict:
    """English sentence -> normalisation + model + sign planning -> gloss and signing frames."""
    english = _check(text, "text")
    gloss = generate(normalise(english))
    return {"english": english, **signing.sign_gloss(gloss)}


def sign_api(gloss: str) -> dict:
    """ASL gloss -> signing frames (skips the model)."""
    gloss = _check(gloss, "gloss")
    return {"english": None, **signing.sign_gloss(gloss)}


def _preview(text):
    """Space page test UI: the response with the big arrays replaced by their sizes."""
    r = translate_api(text)
    for key in ("frames", "limbs", "point_colors"):
        r[key] = f"<{len(r[key])} items>"
    r["groups"] = {k: f"<{len(v)} points>" for k, v in r["groups"].items()}
    return r


with gr.Blocks(title="flan-t5-asl-gloss2pose") as demo:
    gr.Markdown("# flan-t5-asl-gloss2pose\nEnglish → ASL gloss → signing skeleton. API for the Never Silent demo: "
                "`/translate` (text) and `/sign` (gloss).")
    text = gr.Textbox(label="English", placeholder="Type a sentence…", max_length=MAX_CHARS)
    button = gr.Button("Translate", variant="primary")
    output = gr.JSON(label="Response (frames, limbs and colours shortened)")
    gr.Examples(["Hello, how are you?", "I hope we can trust the agreement.", "Thank you for your help."], text)
    button.click(_preview, text, output, api_name=False)
    text.submit(_preview, text, output, api_name=False)

    gr.api(translate_api, api_name="translate")
    gr.api(sign_api, api_name="sign")

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()
