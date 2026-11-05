/* speakrail live UI: the same ws protocol as web/app.js (16 kHz int16 up, 24 kHz int16 + JSON events down), presented as
 * a blob that moves with the voices, subtitles for the reply, and a card per web search. Settings live in the top-left
 * dropdown and are sent as query params on connect (they only apply on the next start). */
const $ = (id) => document.getElementById(id);
const CAPTURE_HZ = 16000;
const SETTINGS = ["barge", "search", "showlog"];

let ws, micCtx, playCtx, worklet, stream, analyser, anaBuf, outGain;
let sources = [], playCursor = 0, uttStart = 0, uttId = 0;
const ijUtts = new Set();   // utts of v3 interjections (live tasks): never cut by the next utt
let running = false;

// ---------------------------------------------------------------- settings dropdown
const panel = $("panel"), gear = $("gear");
gear.onclick = (e) => { e.stopPropagation(); const open = panel.hidden; panel.hidden = !open; gear.setAttribute("aria-expanded", open); };
panel.onclick = (e) => e.stopPropagation();
document.addEventListener("click", () => { panel.hidden = true; gear.setAttribute("aria-expanded", "false"); });
for (const id of SETTINGS) {
  const el = $(id), saved = localStorage.getItem("speakrail." + id);
  if (saved != null) { if (el.type === "checkbox") el.checked = saved === "1"; else el.value = saved; }
  el.onchange = () => { localStorage.setItem("speakrail." + id, el.type === "checkbox" ? (el.checked ? "1" : "0") : el.value); if (id === "showlog") $("log").hidden = !el.checked; };
}
$("log").hidden = !$("showlog").checked;
function lockSettings(lock) { for (const id of SETTINGS) if (id !== "showlog") $(id).disabled = lock; }

// ---------------------------------------------------------------- blob
const canvas = $("blob"), ctx = canvas.getContext("2d");
const lvl = { mic: 0, micTarget: 0, head: 0, bot: 0, botTarget: 0 };   // 0..1, smoothed in the draw loop
let botSpeaking = false;
function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = canvas.clientWidth * dpr; canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize); resize();

function drawBlob(t) {
  const w = canvas.clientWidth, h = canvas.clientHeight, cx = w / 2, cy = h / 2;
  ctx.clearRect(0, 0, w, h);
  // smoothing: fast attack, slow release, so a syllable shows and the shape settles gently
  lvl.mic += (lvl.micTarget - lvl.mic) * (lvl.micTarget > lvl.mic ? 0.35 : 0.06);
  lvl.bot += (lvl.botTarget - lvl.bot) * (lvl.botTarget > lvl.bot ? 0.4 : 0.08);
  const user = Math.min(1, lvl.mic * 1.6 + lvl.head * 0.4);       // the mic level, lifted by the turn head's P(speaking)
  const bot = Math.min(1, lvl.bot * 1.8);
  const base = Math.min(w, h) * 0.17;
  const breathe = 1 + 0.02 * Math.sin(t * 0.0007);
  const R = base * breathe * (1 + user * 0.28 + bot * 0.14);
  const amp = 0.035 + user * 0.22 + bot * 0.08;                    // how far the surface wanders from a circle
  const s = t * 0.001;
  const N = 180, pts = [];
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const n = 0.5 * Math.sin(3 * a + s * 1.1) + 0.3 * Math.sin(5 * a - s * 0.8 + 1.0) + 0.2 * Math.sin(7 * a + s * 1.7 + 2.0)
            + user * 0.35 * Math.sin(11 * a - s * 6.0);           // a fine ripple only while the user talks
    const r = R * (1 + amp * n);
    pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  const path = new Path2D();
  for (let i = 0; i < N; i++) {
    const p0 = pts[i], p1 = pts[(i + 1) % N], mx = (p0[0] + p1[0]) / 2, my = (p0[1] + p1[1]) / 2;
    if (i === 0) path.moveTo(mx, my); else path.quadraticCurveTo(p0[0], p0[1], mx, my);
  }
  const p0 = pts[0], p1 = pts[1]; path.quadraticCurveTo(p0[0], p0[1], (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2);
  path.closePath();

  // halo: wide soft pink glow (the top bar's #f4b6c9), stronger while the assistant speaks
  const glow = ctx.createRadialGradient(cx, cy, R * 0.6, cx, cy, R * 2.2);
  const halo = bot > 0.02 ? 0.12 + bot * 0.20 : 0.08 + user * 0.12;
  glow.addColorStop(0, `rgba(244,182,201,${halo})`); glow.addColorStop(1, "rgba(244,182,201,0)");
  ctx.fillStyle = glow; ctx.fillRect(0, 0, w, h);

  // body: semi-transparent white, brighter core
  const fill = ctx.createRadialGradient(cx - R * 0.15, cy - R * 0.2, R * 0.1, cx, cy, R * 1.15);
  fill.addColorStop(0, `rgba(250,250,250,${0.78 + user * 0.15})`);
  fill.addColorStop(0.7, `rgba(250,250,250,${0.42 + user * 0.15})`);
  fill.addColorStop(1, "rgba(250,250,250,0.06)");
  ctx.shadowColor = "rgba(250,250,250,0.35)"; ctx.shadowBlur = 40 + user * 40;
  ctx.fillStyle = fill; ctx.fill(path);
  ctx.shadowBlur = 0;
  ctx.strokeStyle = `rgba(250,250,250,${0.18 + user * 0.2})`; ctx.lineWidth = 1; ctx.stroke(path);

  // the assistant's voice: a second, thinner membrane pulsing outside the body
  if (bot > 0.02) {
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const a = (i / N) * Math.PI * 2, r = R * (1.12 + bot * 0.22 + 0.04 * Math.sin(9 * a + s * 5));
      i ? ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a)) : ctx.moveTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
    }
    ctx.strokeStyle = `rgba(250,250,250,${0.12 + bot * 0.25})`; ctx.lineWidth = 1.5; ctx.stroke();
  }
  // output level from the playback analyser (drives `bot`)
  if (analyser && botSpeaking) {
    analyser.getFloatTimeDomainData(anaBuf);
    let sum = 0; for (let i = 0; i < anaBuf.length; i++) sum += anaBuf[i] * anaBuf[i];
    lvl.botTarget = Math.min(1, Math.sqrt(sum / anaBuf.length) * 6);
  } else lvl.botTarget = 0;
  requestAnimationFrame(drawBlob);
}
requestAnimationFrame(drawBlob);

// ---------------------------------------------------------------- subtitles
const subUser = $("user"), subBot = $("bot");
let botRaw = "", userRaw = "", fadeTimer = null;
const clean = (s) => s.replace(/<\|[a-z_]+:[a-z_]+\|>/g, "").replace(/<\|[^>]*$/, "").replace(/\((laugh|sigh|cough|clears throat)\)/g, "").replace(/\s+/g, " ").trim();
function showUser(text) { userRaw = text; subUser.textContent = text; subUser.classList.toggle("show", !!text); }
function showBot(text, cut) {
  botRaw = text; subBot.textContent = clean(text); subBot.classList.add("show"); subBot.classList.toggle("cut", !!cut);
  clearTimeout(fadeTimer);
}
function fadeBotLater(ms) { clearTimeout(fadeTimer); fadeTimer = setTimeout(() => { subBot.classList.remove("show"); subUser.classList.remove("show"); }, ms); }

// ---------------------------------------------------------------- search cards
const cards = $("cards"); const cardByTurn = {};
function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function cardSearch(m) {
  for (const t in cardByTurn) if (+t !== m.turn) dropCard(+t);
  const c = document.createElement("div"); c.className = "card";
  c.innerHTML = `<div class="head"><span class="spin"></span>Searching the web<span class="meta">…</span></div><div class="q">${esc(m.query)}</div>`;
  cards.appendChild(c); cardByTurn[m.turn] = c;
  clearTimeout(c.ttl); c.ttl = setTimeout(() => dropCard(m.turn), 40000);
}
function cardDone(m) {
  const c = cardByTurn[m.turn]; if (!c) return;
  const head = c.querySelector(".head");
  head.innerHTML = (m.n < 0 ? "Search failed" : "Web search") + `<span class="meta">${m.n < 0 ? "" : m.n + " results · "}${m.ms} ms</span>`;
  const res = (m.results || []).slice(0, 4);
  if (!res.length) c.insertAdjacentHTML("beforeend", `<div class="none">${m.n < 0 ? "The search did not answer in time." : "Nothing useful came back."}</div>`);
  for (const r of res) {
    let host = ""; try { host = new URL(r.url).host.replace(/^www\./, ""); } catch (e) {}
    c.insertAdjacentHTML("beforeend", `<div class="res"><div class="t">${esc(r.title)}</div><div class="s">${esc(r.snippet)}</div><div class="u">${esc(host)}</div></div>`);
  }
}
// tool cards: one per call (drops after 20 s) + a pinned "Memory" card with the session's notes / lists / todos / counters
const TOOL_TITLE = {claude_code: "Claude", task_status: "Task status", cancel_task: "Task cancelled", reset_chat: "Reset chat", record_note: "Note saved", list_add: "Added to list", todo_add: "To-do added", counter: "Counter",
  stopwatch: "Stopwatch", get_weather: "Weather", unit_convert: "Convert", calculator: "Calculator", dice_roll: "Dice", get_time: "Time"};
let toolN = 0;
function cardTool(m) {
  const c = document.createElement("div"); c.className = "card tool"; const key = "tool" + (++toolN);
  const args = Object.entries(m.args || {}).map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`).join(" · ");
  const r = m.result || {};
  const res = r.error ? `<div class="none">${esc(r.error)}</div>`
    : `<div class="kv">${Object.entries(r).map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v">${esc(typeof v === "string" ? v : JSON.stringify(v))}</span>`).join("")}</div>`;
  c.innerHTML = `<div class="head">${esc(TOOL_TITLE[m.name] || m.name)}<span class="meta">${esc(m.name)} · ${m.ms} ms</span></div>` +
    (args ? `<div class="args">${esc(args)}</div>` : "") + res;
  cards.appendChild(c); cardByTurn[key] = c; setTimeout(() => dropCard(key), 20000);
  if (["record_note", "list_add", "todo_add", "counter"].includes(m.name)) cardMemory(m.store || {});
}
function cardMemory(st) {
  let c = $("memcard");
  if (!c) { c = document.createElement("div"); c.className = "card mem"; c.id = "memcard"; cards.prepend(c); }
  const li = (xs) => xs.map((x) => `<li>${esc(x)}</li>`).join("");
  let h = `<div class="head">Memory<span class="meta">this session</span></div>`;
  if ((st.notes || []).length) h += `<div class="sec">Notes</div><ul>${li(st.notes)}</ul>`;
  for (const [name, items] of Object.entries(st.lists || {})) h += `<div class="sec">${esc(name)}</div><ul>${li(items)}</ul>`;
  if ((st.todos || []).length) h += `<div class="sec">To-do</div><ul>${li(st.todos.map((t) => t.text + (t.due ? " (" + t.due + ")" : "")))}</ul>`;
  const cs = Object.entries(st.counters || {});
  if (cs.length) h += `<div class="sec">Counters</div><div class="kv">${cs.map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v">${v}</span>`).join("")}</div>`;
  c.innerHTML = h;
}
function dropCard(turn) { const c = cardByTurn[turn]; if (!c) return; delete cardByTurn[turn]; c.classList.add("gone"); setTimeout(() => c.remove(), 450); }

// ---------------------------------------------------------------- timings log (optional)
function logTurn(m) {
  const f = (v) => (v == null ? "–" : v + " ms");
  const line = document.createElement("div");
  line.innerHTML = `<b>#${m.turn}</b> word end→ear ${f(m.word_end_to_ear)} · ttft ${f(m.llm_ttft)} · tts ${f(m.tts_ttfa)}` +
    (m.search ? ` · search ${f(m.search_ms)} (${m.search_n} res)` : "") + (m.cut ? ` · cut: ${m.cut}` : "");
  const log = $("log"); log.appendChild(line); while (log.children.length > 6) log.firstChild.remove();
}

// ---------------------------------------------------------------- state & events
function setState(s) { const e = $("state"); e.className = "pill " + s; e.textContent = s; }

function onMessage(ev) {
  if (ev.data instanceof ArrayBuffer) { const dv = new DataView(ev.data); playChunk(dv.getUint16(0, true), new Int16Array(ev.data, 4)); return; }
  const m = JSON.parse(ev.data);
  switch (m.type) {
    case "ready": break;
    case "head": lvl.head = m.p[0]; break;
    case "word": showUser(userRaw + m.raw); break;
    case "endpoint": showUser(m.text); break;
    case "backchannel": showUser(""); break;
    case "interject": ijUtts.add(m.utt); showBot(m.text, false); fadeBotLater(3000); break;
    case "reply_start": setState("thinking"); botRaw = ""; break;
    case "reply_delta": if (!botSpeaking) { botSpeaking = true; setState("speaking"); } showBot(botRaw + m.text, false); break;
    case "search": cardSearch(m); break;
    case "search_done": cardDone(m); break;
    case "tool": cardTool(m); break;
    case "reply_end": botSpeaking = false; setState("listening"); userRaw = ""; fadeBotLater(6000); break;
    case "stop_audio": stopPlayback(); botSpeaking = false; setState("listening"); if (botRaw) showBot(botRaw, true); userRaw = ""; fadeBotLater(2500); break;
    case "turn": logTurn(m); break;
    case "duck": if (outGain) outGain.gain.setTargetAtTime(m.gain, playCtx.currentTime, 0.03); break;
    case "error": setState("error"); showBot("error in " + m.where + ": " + m.detail, true); break;
    case "reset": showBot("(new chat" + (m.instructions ? ": " + m.instructions : "") + ")", false); fadeBotLater(4000); break;   // reset_chat
  }
}

// ---------------------------------------------------------------- audio (as in web/app.js)
const WORKLET = `class Cap extends AudioWorkletProcessor { process(inputs) { const ch = inputs[0][0]; if (ch) this.port.postMessage(new Float32Array(ch)); return true; } } registerProcessor('cap', Cap);`;

async function start() {
  $("go").disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1, sampleRate: CAPTURE_HZ } });
  } catch (e) { setState("error"); showBot("Microphone denied: " + e.message, true); $("go").disabled = false; return; }
  playCtx = new AudioContext({ sampleRate: 24000 }); await playCtx.resume();
  analyser = playCtx.createAnalyser(); analyser.fftSize = 512; analyser.smoothingTimeConstant = 0.5; anaBuf = new Float32Array(analyser.fftSize);
  analyser.connect(playCtx.destination);
  outGain = playCtx.createGain(); outGain.connect(analyser);      // ducking: lower our volume while the user talks over a reply
  micCtx = new AudioContext({ sampleRate: CAPTURE_HZ });
  await micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));

  const qs = `?barge=${$("barge").value}&search=${$("search").value}`;
  const ck = new URLSearchParams(location.search).get("ck");     // the claude_code access key, if the page has one
  const wsUrl = new URL("ws" + qs + (ck ? "&ck=" + encodeURIComponent(ck) : ""), location.href); wsUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(wsUrl); ws.binaryType = "arraybuffer";
  ws.onmessage = onMessage;
  ws.onclose = () => { if (running) stop(true); };
  ws.onerror = () => setState("error");
  await new Promise((r) => (ws.onopen = r));

  worklet = new AudioWorkletNode(micCtx, "cap");
  worklet.port.onmessage = (ev) => {
    const f = ev.data, src = micCtx.sampleRate === CAPTURE_HZ ? f : downsample(f, micCtx.sampleRate, CAPTURE_HZ);
    const pcm = new Int16Array(src.length); let peak = 0;
    for (let i = 0; i < src.length; i++) { pcm[i] = Math.max(-1, Math.min(1, src[i])) * 32767; peak = Math.max(peak, Math.abs(src[i])); }
    lvl.micTarget = Math.min(1, peak * 2.5);
    if (ws.readyState === 1) ws.send(pcm.buffer);
  };
  micCtx.createMediaStreamSource(stream).connect(worklet);
  worklet.connect(micCtx.destination);
  running = true; lockSettings(true);
  $("go").hidden = true; $("controls").hidden = false;
  setState("listening");
}

function stop(fromClose) {
  running = false;
  if (ws && !fromClose && ws.readyState === 1) ws.send(JSON.stringify({ cmd: "stop" }));
  stopPlayback(); botSpeaking = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (micCtx) micCtx.close();
  lvl.micTarget = 0; lvl.head = 0;
  $("controls").hidden = true; $("go").hidden = false; $("go").disabled = false; lockSettings(false);
  setState("idle");
}

function downsample(buf, from, to) {
  const ratio = from / to, out = new Float32Array(Math.floor(buf.length / ratio));
  for (let i = 0; i < out.length; i++) out[i] = buf[Math.floor(i * ratio)];
  return out;
}
function stopPlayback() { for (const s of sources) { try { s.stop(); } catch (e) {} } sources = []; playCursor = 0; uttStart = 0; }
function playChunk(utt, pcm) {
  if (utt !== uttId) {                 // a new utt cuts the old one, unless the old one was an interjection: queue after it
    if (ijUtts.has(uttId)) uttStart = 0; else stopPlayback();
    uttId = utt;
  }
  const buf = playCtx.createBuffer(1, pcm.length, playCtx.sampleRate), ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = playCtx.createBufferSource(); src.buffer = buf; src.connect(outGain || analyser);
  const now = playCtx.currentTime;
  if (playCursor < now) playCursor = now + 0.008;
  if (uttStart === 0) { uttStart = playCursor; ws.send(JSON.stringify({ cmd: "play_start", utt: utt, delay_ms: (playCursor - now) * 1000 })); }
  src.start(playCursor); playCursor += pcm.length / playCtx.sampleRate;
  sources.push(src); src.onended = () => { sources = sources.filter((s) => s !== src); };
}

$("go").onclick = start;
$("stop").onclick = () => stop(false);
$("interrupt").onclick = () => ws && ws.readyState === 1 && ws.send(JSON.stringify({ cmd: "interrupt" }));
$("reconnect").onclick = () => ws && ws.readyState === 1 && ws.send(JSON.stringify({ cmd: "reconnect_asr" }));
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && running && !["SELECT", "INPUT", "BUTTON"].includes(document.activeElement.tagName)) { e.preventDefault(); $("interrupt").click(); }
});

// ?demo=1: a static preview of the running state (subtitles + a search card + a moving blob), no microphone needed
if (new URLSearchParams(location.search).get("demo") === "1") {
  if (new URLSearchParams(location.search).get("settings") === "1") panel.hidden = false;
  $("go").hidden = true; $("controls").hidden = false; setState("speaking");
  showUser("What's the weather like in Berlin right now?");
  cardSearch({ turn: 1, query: "current weather in Berlin" });
  cardDone({ turn: 1, n: 5, ms: 379, results: [
    { title: "Berlin, Berlin, Germany Weather Forecast - AccuWeather", snippet: "Today. 9/19. 70° 59°. Breezy this morning. Night: Partly to mostly cloudy.", url: "https://www.accuweather.com/en/de/berlin/10178/weather-forecast/178087" },
    { title: "Berlin - BBC Weather", snippet: "Light rain and a gentle breeze. Sunny intervals and a moderate breeze later in the week.", url: "https://www.bbc.com/weather/2950159" },
    { title: "Weather Forecast and Conditions for Berlin, Germany - The Weather Channel", snippet: "Today's Outlook · 1 pm 65° · 2 pm 66° · Partly cloudy with a light breeze.", url: "https://weather.com/weather/today/l/Berlin" }] });
  showBot("One sec, looking that up, Right now, it is around sixty-five degrees and partly cloudy in Berlin. It feels like about sixty-six degrees with a light breeze.", false);
  let ph = 0; setInterval(() => { ph += 0.09; lvl.micTarget = Math.max(0, 0.45 * Math.sin(ph) + 0.2 * Math.sin(ph * 3.1)); lvl.head = 0.8; }, 40);
}
