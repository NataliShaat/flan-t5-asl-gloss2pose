# flan-t5-asl-gloss2pose

English → ASL gloss → animated signing skeleton. The web demo is called **Never Silent**.

NLP 434 course project, Princess Nora bint Abdulrahman University
(Natali Shaat, Ghaida Alharbi, Lena Aldokhayel, Alhanouf Alsaiary).
`google/flan-t5-base` is fine-tuned on ASLG-PC12 to translate English into ASL gloss. The gloss is then
turned into a signing figure by joining pre-recorded MediaPipe `.pose` clips from
[SignBridge](https://github.com/SignBridgeApp) (MIT licence).

```
flan-t5-asl-gloss2pose/
  backend/     Gradio app, runs as a Hugging Face Space on ZeroGPU
  frontend/    static HTML/CSS/JS, deployed on Vercel (no framework, no build step)
  notebook/    NLP_Project.ipynb (training, evaluation and the original demo)
```

The model (~1 GB with PyTorch) and the 280 MB sign dictionary are too big for Vercel, so the work is split:
the browser calls the Space through the official [`@gradio/client`](https://www.gradio.app/docs/js-client),
the Space returns the gloss plus the skeleton frames as JSON, and the browser draws them on a `<canvas>`.

On the Space, only `model.generate` runs on the GPU (`@spaces.GPU(duration=15)`). Sign planning and pose
building run on CPU and use no GPU quota. If no GPU can be had (daily quota used up, queue timeout), generation
falls back to CPU, so the demo keeps working, just a few seconds slower.

The backend reuses the notebook code unchanged: `translate()` and `CFG` from Part 0, and `clean_gloss`,
`plan_signs`, `_read_pose`, `_trim`, `_best_join`, `glosses_to_pose`, `sign_gloss` from cell 6.2
(see `backend/signing.py` for the small additions: dictionary download, per-word frame ranges, and JSON export
instead of a GIF).

---

## 1. Push the fine-tuned model to the Hugging Face Hub (from Colab)

Already done: [`noxy28/flan-t5-base-aslg`](https://huggingface.co/noxy28/flan-t5-base-aslg). To push a new version:

1. Create a **write** token at <https://huggingface.co/settings/tokens>.
2. In the notebook, after Part 0 (so `CFG` exists), run:

   ```python
   from huggingface_hub import login
   login()  # paste the write token

   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
   REPO = "noxy28/flan-t5-base-aslg"

   AutoTokenizer.from_pretrained(CFG["model_full"]).push_to_hub(REPO)
   AutoModelForSeq2SeqLM.from_pretrained(CFG["model_full"]).push_to_hub(REPO)
   ```

Keep the model repo **public**: the Space reads it without a token.

## 2. Create the Gradio Space on ZeroGPU from `backend/`

Free accounts can host up to 2 ZeroGPU Spaces (verified email, account older than 30 days).

1. Go to <https://huggingface.co/new-space> (already done: `noxy28/flan-t5-asl-gloss2pose`).
   - Space name: `flan-t5-asl-gloss2pose`
   - SDK: **Gradio** → Blank
   - Hardware: **ZeroGPU**
2. Open the Space's **Files** tab → **Add file → Upload files**, and upload these four files from `backend/`:
   `app.py`, `signing.py`, `requirements.txt`, `README.md`.
   Replace the Space's existing `README.md` when asked: its header sets `sdk: gradio`, `sdk_version: 5.50.0`,
   `python_version: "3.12"` and `app_file: app.py`.
   Commit the upload.
3. Optional: **Settings → Variables and secrets → New variable** `MODEL_ID`, only if you want a model other
   than the default `noxy28/flan-t5-base-aslg`.
4. The Space builds (a few minutes) and then starts. At its first start it downloads the sign dictionary and the
   model, which takes another minute or two. Watch **Logs** until you see `Ready: model=...`.
5. Test it on the Space page: type a sentence and press **Translate**; the JSON shows the gloss, the words and the
   number of frames. The **Use via API** link at the bottom lists the two endpoints, `/translate` and `/sign`.

Spaces go to sleep after a while without visitors. The first request after that wakes the Space up, which can
take a minute; the website shows "Waking up the signer…" while it waits.

**GPU quota.** ZeroGPU time is counted per visitor (about 2 minutes a day without a Hugging Face login).
One translation uses well under a second of GPU, and when a visitor runs out the Space quietly uses CPU instead.

## 3. Push the repo to GitHub

```bash
cd flan-t5-asl-gloss2pose
git init
git add .
git commit -m "Never Silent: English to ASL demo"
git branch -M main
git remote add origin https://github.com/YOUR-GITHUB-USERNAME/flan-t5-asl-gloss2pose.git
git push -u origin main
```

Create the empty repo `flan-t5-asl-gloss2pose` on GitHub first (no README, no .gitignore). The `.gitignore`
keeps model weights, `gloss2pose/` and `__pycache__/` out of the repo.

## 4. Import it in Vercel

1. <https://vercel.com/new> → **Import** the `flan-t5-asl-gloss2pose` GitHub repo.
2. **Root Directory**: click *Edit* and choose `frontend`.
3. **Framework Preset**: *Other*. Leave **Build Command** and **Output Directory** empty
   (turn the overrides on and leave them blank if Vercel fills something in). No install command is needed.
4. **Deploy**.

## 5. Put the Space id into `frontend/config.js`

```js
window.SPACE_ID = "noxy28/flan-t5-asl-gloss2pose";
```

It is already set to this Space. If the Space is renamed or moved, change it here, commit and push;
Vercel redeploys on its own.

---

## Running locally

Backend (Python 3.12), on CPU:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt gradio==5.50.0
python app.py                                           # http://127.0.0.1:7860
```

The first start downloads the sign dictionary into `backend/gloss2pose/` (set `SIGN_DIR` to keep it elsewhere)
and the model into the Hugging Face cache. `@spaces.GPU` does nothing outside ZeroGPU, so everything runs on CPU.

Frontend: point `frontend/config.js` at the local app with `window.SPACE_ID = "http://127.0.0.1:7860/";`, then

```bash
cd frontend
python -m http.server 8000      # open http://127.0.0.1:8000
```

## API

From JavaScript (this is what `frontend/app.js` does):

```js
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client@1.19.1/dist/index.min.js";
const app = await Client.connect("noxy28/flan-t5-asl-gloss2pose");
const res = await app.predict("/translate", { text: "Hello, how are you?" });   // or "/sign", { gloss: "hello" }
const data = res.data[0];
```

From Python: `gradio_client.Client("noxy28/flan-t5-asl-gloss2pose").predict(text="Hello", api_name="/translate")`.

Both return

```json
{
  "english": "Hello, how are you?",
  "gloss": "HELLO , HOW BE YOU ?",
  "fps": 30,
  "width": 384, "height": 384,
  "frames": [[[x, y] or null, ...], ...],
  "limbs": [[i, j, "#rrggbb"], ...],
  "point_colors": ["#rrggbb", ...],
  "groups": {"body": [...], "face": [...], "left_hand": [...], "right_hand": [...]},
  "words": [{"word": "hello", "how": "sign", "start": 0, "end": 31}, ...]
}
```

`frames[t][i]` is point `i` at frame `t` in a 384×384 pixel space (`null` = not detected).
`fps` is the frame rate of the first clip in the sentence (23–30 for SignBridge clips), as in the notebook.
`words[].start/end` is the frame range of each word (end exclusive); a word with no sign clip has `null`.
`english` is `null` for `/sign`. Input is limited to 300 characters; empty input returns an error.

## Credits

Sign clips and the gloss-to-pose method: [SignBridge](https://github.com/SignBridgeApp) (MIT licence).
Dataset: ASLG-PC12 (`achrafothman/aslg_pc12`). Base model: `google/flan-t5-base`.
