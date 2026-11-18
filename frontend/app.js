import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client@1.19.1/dist/index.min.js";

(() => {
  "use strict";

  const SPACE = window.SPACE_ID;
  const LOOP_PAUSE_MS = 600;
  const PAD = 0.06; // empty margin around the figure, as a share of the canvas
  const WIDTH = { body: 4, left_hand: 3, right_hand: 3, face: 1.5 };
  const DOT = { body: 2.6, left_hand: 1.8, right_hand: 1.8, face: 0.9 };
  const DRAW_ORDER = ["body", "face", "left_hand", "right_hand"];

  const MSG = {
    waking: "Waking up the signer… this can take a minute the first time.",
    translating: "Translating…",
    error: "Sorry, something went wrong. Please try again in a moment.",
    noSigns: "We couldn't find signs for this sentence yet.",
  };

  const $ = (id) => document.getElementById(id);
  const canvas = $("figure");
  const ctx = canvas.getContext("2d");
  const glossEl = $("gloss");
  const statusEl = $("status");
  const form = $("form");
  const input = $("text");
  const submit = $("submit");
  const playBtn = $("play");
  const loopBtn = $("loop");
  const speedBtns = [...document.querySelectorAll("[data-speed]")];

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const state = {
    clip: null,      // prepared animation (see prepare())
    chips: [],       // chip elements matching clip.words
    frame: 0,
    elapsed: 0,      // ms of clip time played so far
    holdUntil: 0,    // timestamp until which we sit on the last frame between loops
    lastTs: 0,
    playing: !reducedMotion,
    loop: true,
    speed: 1,
    activeWord: -1,
  };

  // ---------- status line ----------
  function setStatus(text, pulse = false) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("pulse", Boolean(text) && pulse);
  }

  // ---------- network (Gradio client) ----------
  function onSpaceStatus(s) {
    // Passing a status callback is also what makes the client wake a sleeping Space.
    if ((s.status === "sleeping" || s.status === "building") && !busy) setStatus(MSG.waking, true);
  }

  let clientPromise = null;
  function client() {
    if (!clientPromise) {
      clientPromise = Client.connect(SPACE, { status_callback: onSpaceStatus, events: ["data", "status"] })
        .catch((e) => { clientPromise = null; throw e; });
    }
    return clientPromise;
  }

  async function call(endpoint, payload) {
    const app = await client();
    const result = await app.predict(endpoint, payload);
    return result.data[0];
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---------- preparing a clip ----------
  function prepare(data) {
    const frames = data.frames || [];
    const groupOf = [];
    for (const [name, idx] of Object.entries(data.groups || {})) {
      for (const i of idx) groupOf[i] = name;
    }
    const rank = (g) => {
      const r = DRAW_ORDER.indexOf(g);
      return r < 0 ? DRAW_ORDER.length : r;
    };
    const limbs = (data.limbs || [])
      .map(([a, b, color]) => ({ a, b, color, group: groupOf[a] || "body" }))
      .sort((x, y) => rank(x.group) - rank(y.group));

    const dots = [];
    const pointColors = data.point_colors || [];
    for (const g of DRAW_ORDER) {
      for (const i of (data.groups || {})[g] || []) dots.push({ i, group: g, color: pointColors[i] || "#1E2A3A" });
    }

    // One bounding box for the whole clip, so the figure does not jump around between frames.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const f of frames) {
      for (const p of f) {
        if (!p) continue;
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
      }
    }
    const box = Number.isFinite(minX)
      ? { minX, minY, w: Math.max(1, maxX - minX), h: Math.max(1, maxY - minY) }
      : { minX: 0, minY: 0, w: data.width || 384, h: data.height || 384 };

    return {
      frames,
      limbs,
      dots,
      box,
      fps: data.fps || 25,
      words: data.words || [],
      gloss: data.gloss || "",
    };
  }

  // ---------- canvas ----------
  let cssSize = 0;
  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    cssSize = rect.width;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function draw() {
    const clip = state.clip;
    ctx.clearRect(0, 0, cssSize, cssSize);
    if (!clip || !clip.frames.length) return;
    const pts = clip.frames[Math.min(state.frame, clip.frames.length - 1)];

    const inner = cssSize * (1 - 2 * PAD);
    const scale = inner / Math.max(clip.box.w, clip.box.h);
    const ox = (cssSize - clip.box.w * scale) / 2 - clip.box.minX * scale;
    const oy = (cssSize - clip.box.h * scale) / 2 - clip.box.minY * scale;
    const X = (p) => ox + p[0] * scale;
    const Y = (p) => oy + p[1] * scale;
    const k = Math.min(1.2, Math.max(0.7, cssSize / 380)); // keep strokes in proportion on small screens

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const l of clip.limbs) {
      const p = pts[l.a], q = pts[l.b];
      if (!p || !q) continue;
      ctx.strokeStyle = l.color;
      ctx.lineWidth = WIDTH[l.group] * k;
      ctx.beginPath();
      ctx.moveTo(X(p), Y(p));
      ctx.lineTo(X(q), Y(q));
      ctx.stroke();
    }
    for (const d of clip.dots) {
      const p = pts[d.i];
      if (!p) continue;
      ctx.fillStyle = d.color;
      ctx.beginPath();
      ctx.arc(X(p), Y(p), DOT[d.group] * k, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ---------- gloss chips ----------
  function renderGloss(words) {
    glossEl.replaceChildren();
    state.chips = words.map((w) => {
      const li = document.createElement("li");
      li.className = "chip";
      li.textContent = w.word;
      if (w.start == null) {
        li.classList.add("none");
        li.title = "No sign clip for this word";
      } else if (w.how === "fingerspelled") {
        const s = document.createElement("small");
        s.textContent = "spelled";
        li.append(s);
      }
      glossEl.append(li);
      return li;
    });
    state.activeWord = -1;
  }

  function highlight() {
    const clip = state.clip;
    if (!clip) return;
    // The word being signed is the last one that has started; it stays lit through the short gap after it.
    let idx = -1;
    clip.words.forEach((w, i) => {
      if (w.start != null && state.frame >= w.start) idx = i;
    });
    if (idx === state.activeWord) return;
    state.chips[state.activeWord]?.classList.remove("active");
    state.chips[idx]?.classList.add("active");
    state.activeWord = idx;
  }

  function describe(clip) {
    const words = clip.words.map((w) => w.word.toUpperCase()).join(" ");
    canvas.setAttribute("aria-label",
      words ? `A signing figure performing the ASL gloss: ${words}.` : "A signing figure.");
  }

  // ---------- playback ----------
  function show(data) {
    const clip = prepare(data);
    renderGloss(clip.words);
    if (!clip.frames.length) {
      setStatus(MSG.noSigns);
      return; // keep the previous animation on screen
    }
    state.clip = clip;
    state.chips.forEach((c) => c.classList.remove("active"));
    state.activeWord = -1;
    state.frame = 0;
    state.elapsed = 0;
    state.holdUntil = 0;
    describe(clip);
    draw();
    highlight();
  }

  function tick(ts) {
    requestAnimationFrame(tick);
    const dt = state.lastTs ? Math.min(ts - state.lastTs, 100) : 0;
    state.lastTs = ts;
    const clip = state.clip;
    if (!clip || !state.playing || !clip.frames.length) return;

    if (state.holdUntil) {
      if (ts < state.holdUntil) return;
      state.holdUntil = 0;
      state.elapsed = 0;
      state.frame = 0;
      draw();
      highlight();
      return;
    }

    state.elapsed += dt * state.speed;
    const f = Math.floor((state.elapsed * clip.fps) / 1000);
    if (f >= clip.frames.length) {
      state.frame = clip.frames.length - 1;
      if (state.loop) {
        state.holdUntil = ts + LOOP_PAUSE_MS;
      } else {
        setPlaying(false);
        state.elapsed = 0; // pressing play again starts from the top
      }
    } else if (f !== state.frame) {
      state.frame = f;
    } else {
      return;
    }
    draw();
    highlight();
  }

  function setPlaying(on) {
    state.playing = on;
    playBtn.classList.toggle("paused", !on);
    playBtn.setAttribute("aria-label", on ? "Pause" : "Play");
    if (on && state.clip && state.elapsed === 0) {
      state.frame = 0;
      draw();
      highlight();
    }
  }

  playBtn.addEventListener("click", () => setPlaying(!state.playing));

  loopBtn.addEventListener("click", () => {
    state.loop = !state.loop;
    loopBtn.setAttribute("aria-pressed", String(state.loop));
  });

  speedBtns.forEach((b) => b.addEventListener("click", () => {
    state.speed = Number(b.dataset.speed);
    speedBtns.forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
  }));

  // ---------- translate ----------
  let busy = false;
  async function translate(text) {
    text = text.trim();
    if (!text || busy) return;
    busy = true;
    submit.disabled = true;
    setStatus(MSG.translating, true);
    try {
      const data = await call("/translate", { text });
      setStatus("");
      show(data);
    } catch (e) {
      setStatus(MSG.error);
    } finally {
      busy = false;
      submit.disabled = false;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    translate(input.value);
  });

  document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => {
    input.value = b.textContent;
    translate(b.textContent);
  }));

  // ---------- greeting on load (retries while the Space wakes up) ----------
  async function greet() {
    const wakingTimer = setTimeout(() => { if (!busy) setStatus(MSG.waking, true); }, 1200);
    const deadline = Date.now() + 4 * 60 * 1000;
    let wait = 2000;
    try {
      while (true) {
        try {
          const data = await call("/sign", { gloss: "hello" });
          if (!state.clip) show(data); // a translation may have arrived first
          if (!busy) setStatus("");
          return;
        } catch (e) {
          if (Date.now() > deadline) throw e;
          await sleep(wait);
          wait = Math.min(wait * 1.5, 10000);
        }
      }
    } catch (e) {
      if (!busy) setStatus(MSG.error);
    } finally {
      clearTimeout(wakingTimer);
    }
  }

  setPlaying(state.playing);
  new ResizeObserver(resize).observe(canvas);
  requestAnimationFrame(tick);
  greet();
})();
