# Signing figure, copied from the notebook (cell 6.2).
# The gloss-to-pose method is adapted from the SignBridge server (github.com/SignBridgeApp/server, MIT licence).
# Each dictionary word is a pre-recorded MediaPipe skeleton (.pose file); clips are trimmed, joined and smoothed.
# Changes from the notebook:
#   - The 280 MB dictionary is downloaded and unzipped next to this file at startup if missing
#     (SIGN_DIR can point elsewhere).
#   - glosses_to_pose also returns the frame range of each token, so the page can highlight the current word.
#   - sign_gloss returns the pose as JSON (pose_to_json) instead of saving a GIF.
import os, re, zipfile, tempfile, urllib.request
import numpy as np
import scipy.signal
from scipy.spatial.distance import cdist
from num2words import num2words
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic, correct_wrists, pose_normalization_info

SIGN_DIR = os.environ.get("SIGN_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gloss2pose"))
SIGN_ZIP_URL = "https://github.com/SignBridgeApp/models/releases/download/v0.0/gloss2pose.zip"

if not os.path.exists(f"{SIGN_DIR}/words.txt"):
    print("Downloading the sign dictionary...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "gloss2pose.zip")
        urllib.request.urlretrieve(SIGN_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:   # the zip holds a top-level gloss2pose/ folder
            z.extractall(os.path.dirname(os.path.abspath(SIGN_DIR)))

with open(f"{SIGN_DIR}/words.txt", encoding="utf-8") as f:
    SIGN_WORDS = {w.strip() for w in f if w.strip()}
MAX_PHRASE = max(len(w.split()) for w in SIGN_WORDS)   # some entries are phrases, e.g. "thank you"
print(f"Sign dictionary: {len(SIGN_WORDS)} entries")


def clean_gloss(gloss):
    """ASLG gloss -> lowercase words the dictionary can match (drops X-/DESC- markers and punctuation)."""
    words = re.findall(r"[a-z0-9']+", re.sub(r"\b(?:X|DESC)-", "", gloss).lower())
    out = []
    for w in words:
        out.extend(num2words(int(w)).replace("-", " ").split() if w.isdigit() else [w])
    return out


def plan_signs(words):
    """Match the longest dictionary phrase first; otherwise fingerspell. Returns (tokens, report)."""
    tokens, report, i = [], [], 0
    while i < len(words):
        for n in range(min(MAX_PHRASE, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            if phrase in SIGN_WORDS:
                tokens.append(phrase); report.append((phrase, "sign")); i += n
                break
        else:
            letters = [c for c in words[i] if c in SIGN_WORDS]
            missing = sorted({c for c in words[i] if c not in SIGN_WORDS})
            tokens.extend(letters)
            report.append((words[i], "fingerspelled" + (f" (no clip for: {', '.join(missing)})" if missing else "")))
            i += 1
    return tokens, report


def _read_pose(token):
    with open(f"{SIGN_DIR}/asl/{token}.pose", "rb") as f:
        return Pose.read(f.read())


def _trim(pose, start, end):
    """Keep only the frames where a hand is visible."""
    h = pose.header
    wrists = [h._get_point_index("LEFT_HAND_LANDMARKS", "WRIST"), h._get_point_index("RIGHT_HAND_LANDMARKS", "WRIST")]
    hands = pose.body.confidence[:, 0, wrists].sum(axis=1) > 0
    if not hands.any():
        return pose
    a = int(np.argmax(hands)) if start else 0
    b = len(hands) - int(np.argmax(hands[::-1])) if end else len(hands)
    pose.body = pose.body[a:b]
    return pose


def _best_join(p1, p2, window=0.3):
    """Frames where the end of one clip is closest to the start of the next."""
    n1, n2 = max(1, int(len(p1.body.data) * window)), max(1, int(len(p2.body.data) * window))
    last = np.asarray(p1.body.data[-n1:]).reshape(n1, -1)
    first = np.asarray(p2.body.data[:n2]).reshape(n2, -1)
    i, j = np.unravel_index(np.argmin(cdist(last, first)), (n1, n2))
    return len(p1.body.data) - n1 + i, j


def glosses_to_pose(tokens, pause=0.2):
    """Returns (pose, spans): spans[k] = (start, end) frames of token k in the joined clip, end exclusive."""
    poses = [_read_pose(t) for t in tokens]
    poses = [reduce_holistic(p) for p in poses]
    poses = [p.normalize(pose_normalization_info(p.header)) for p in poses]
    poses = [_trim(p, i > 0, i < len(poses) - 1) for i, p in enumerate(poses)]

    # cut each clip at its best join point, then add a short blank gap that is filled by interpolation
    start, cut = 0, []
    for i, p in enumerate(poses):
        end, nxt = _best_join(p, poses[i + 1]) if i < len(poses) - 1 else (len(p.body.data), None)
        cut.append(p.body[start:end]); start = nxt or 0
    fps = poses[0].body.fps
    shape = cut[0].data.shape
    gap_data = np.zeros((int(pause * fps), shape[1], shape[2], shape[3]))
    gap_conf = np.zeros((int(pause * fps), shape[1], shape[2]))
    data, conf, spans, t = [], [], [], 0
    for k, b in enumerate(cut):
        data.append(np.asarray(b.data.filled(0)) if np.ma.isMaskedArray(b.data) else np.asarray(b.data))
        conf.append(np.asarray(b.confidence))
        spans.append((t, t + len(data[-1]))); t += len(data[-1])
        if k < len(cut) - 1:
            data.append(gap_data); conf.append(gap_conf); t += len(gap_data)
    body = NumPyPoseBody(fps=fps, data=np.concatenate(data), confidence=np.concatenate(conf)).interpolate(kind="linear")
    pose = Pose(header=poses[0].header, body=body)

    # light smoothing (skipping the face) and wrist correction
    face = [c for c in pose.header.components if c.name == "FACE_LANDMARKS"][0]
    face_idx = set(range(pose.header._get_point_index("FACE_LANDMARKS", face.points[0]),
                         pose.header._get_point_index("FACE_LANDMARKS", face.points[-1]) + 1))
    arr = pose.body.data
    if len(arr) >= 3:
        for p in range(arr.shape[2]):
            if p not in face_idx:
                arr[:, 0, p, :] = scipy.signal.savgol_filter(arr[:, 0, p, :], 3, 1, axis=0)
    pose = correct_wrists(pose)

    # move into pixel space for drawing
    size = 384
    pose.body.data = (pose.body.data + 1.25) * (size / 2.5)
    pose.header.dimensions.width = pose.header.dimensions.height = size
    return pose, spans


GROUPS = {"POSE_LANDMARKS": "body", "FACE_LANDMARKS": "face",
          "LEFT_HAND_LANDMARKS": "left_hand", "RIGHT_HAND_LANDMARKS": "right_hand"}


def _hex(rgb):
    # OpenCV saturates colour values to 0..255 when drawing (some header colours are above 255)
    return "#" + "".join(f"{int(np.clip(round(v), 0, 255)):02x}" for v in rgb)


def header_json(header):
    """Limbs, per-point colours and point groups, coloured the way pose_format's PoseVisualizer does it.

    The header stores RGB; PoseVisualizer only reverses it to BGR for OpenCV, so the header values are the
    browser colours. Point i of a component gets colors[i % len(colors)]; a limb gets the mean of its two
    points' colours.
    """
    limbs, point_colors, groups, idx = [], [], {}, 0
    for c in header.components:
        palette = np.asarray(c.colors, dtype=float).reshape(-1, 3)
        colors = palette[np.arange(len(c.points)) % len(palette)]
        point_colors.extend(_hex(col) for col in colors)
        limbs.extend([idx + a, idx + b, _hex((colors[a] + colors[b]) / 2)] for a, b in c.limbs)
        groups[GROUPS.get(c.name, c.name.lower())] = list(range(idx, idx + len(c.points)))
        idx += len(c.points)
    return {"limbs": limbs, "point_colors": point_colors, "groups": groups}


def pose_to_json(pose):
    """Frames as [[x, y] | None per point], rounded to 1 decimal; zero-confidence points become None."""
    data = pose.body.data
    xy = np.asarray(data.filled(0) if np.ma.isMaskedArray(data) else data)[:, 0, :, :2]
    visible = (np.asarray(pose.body.confidence)[:, 0, :] > 0) & np.isfinite(xy).all(axis=-1)
    xy = np.round(xy.astype(float), 1).tolist()
    frames = [[p if v else None for p, v in zip(f, vis)] for f, vis in zip(xy, visible.tolist())]
    return {"fps": float(pose.body.fps), "width": pose.header.dimensions.width,
            "height": pose.header.dimensions.height, "frames": frames, **header_json(pose.header)}


def _word_spans(report, spans):
    """Frame range per report entry: a sign is one token, a fingerspelled word spans all its letter clips."""
    words, k = [], 0
    for word, how in report:
        n = 1 if how == "sign" else sum(c in SIGN_WORDS for c in word)
        entry = {"word": word, "how": how.split(" (")[0],
                 "start": spans[k][0] if n else None, "end": spans[k + n - 1][1] if n else None}
        if "no clip for: " in how:
            entry["missing"] = how.split("no clip for: ")[1].rstrip(")").split(", ")
        words.append(entry); k += n
    return words


def sign_gloss(gloss):
    """Gloss string -> signing figure as JSON (gloss, frames, limbs, groups, words)."""
    tokens, report = plan_signs(clean_gloss(gloss))
    if not tokens:
        return {"gloss": gloss, "fps": None, "width": None, "height": None, "frames": [], "limbs": [],
                "point_colors": [], "groups": {}, "words": _word_spans(report, [])}
    pose, spans = glosses_to_pose(tokens)
    return {"gloss": gloss, **pose_to_json(pose), "words": _word_spans(report, spans)}
