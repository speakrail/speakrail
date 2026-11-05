/* speakrail debug client.
 * Capture: 16 kHz AudioContext + worklet -> int16 -> ws (echoCancellation on: the
 * assistant's voice reaching the mic would become user words).
 * Playback: 24 kHz AudioContext (the TTS rate). Every scheduled buffer is tracked so a
 * stop_audio can actually stop it. play_start is reported with its scheduling delay.
 */
const $ = (id) => document.getElementById(id);
const CAPTURE_HZ = 16000;

let ws, micCtx, playCtx, worklet, stream;
let sources = [], playCursor = 0, uttStart = 0, uttId = 0;
const ijUtts = new Set();   // utts of v3 interjections (live tasks): never cut by the next utt
let userEl = null, botEl = null, botRaw = "";
const stat = { lag: [], ttft: [], clause: [], ttfa: [], play: [], m2e: [], turns: 0, replaced: 0, prem: 0 };
const pct = (a, q) => { if (!a.length) return "-"; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(s.length * q))]; };
const two = (a) => a.length ? `${pct(a, .5)} / ${pct(a, .95)}` : "-";
function renderStats() {
  $("s_lag").textContent = two(stat.lag);
  $("s_ttft").textContent = pct(stat.ttft, .5);
  $("s_clause").textContent = pct(stat.clause, .5);
  $("s_ttfa").textContent = pct(stat.ttfa, .5);
  $("s_play").textContent = pct(stat.play, .5);
  $("s_m2e").textContent = two(stat.m2e);
  $("s_turns").textContent = `${stat.turns} (${stat.replaced})`;
  $("s_prem").textContent = stat.prem;
}

const WORKLET = `
class Cap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0][0]; if (ch) this.port.postMessage(new Float32Array(ch)); return true; }
}
registerProcessor('cap', Cap);`;

function setState(s) { const e = $("state"); e.className = "pill " + s; e.textContent = s; }

async function start() {
  $("go").disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1, sampleRate: CAPTURE_HZ } });
  } catch (e) {
    setState("error"); addMsg("bot", "Microphone denied: " + e.message); $("go").disabled = false; return;
  }
  playCtx = new AudioContext({ sampleRate: 24000 });
  await playCtx.resume();
  micCtx = new AudioContext({ sampleRate: CAPTURE_HZ });
  await micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));

  const qs = `?barge=${$("barge").value}`;
  // relative to the page so the UI also works behind a path prefix (a reverse proxy)
  const wsUrl = new URL("ws" + qs, location.href); wsUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";
  ws.onmessage = onMessage;
  ws.onclose = () => { setState("idle"); $("interrupt").disabled = true; $("reconnect").disabled = true; };
  ws.onerror = () => setState("error");
  await new Promise((r) => (ws.onopen = r));

  worklet = new AudioWorkletNode(micCtx, "cap");
  worklet.port.onmessage = (ev) => {
    const f = ev.data;
    const src = micCtx.sampleRate === CAPTURE_HZ ? f : downsample(f, micCtx.sampleRate, CAPTURE_HZ);
    const pcm = new Int16Array(src.length);
    let peak = 0;
    for (let i = 0; i < src.length; i++) { pcm[i] = Math.max(-1, Math.min(1, src[i])) * 32767; peak = Math.max(peak, Math.abs(src[i])); }
    $("mic").style.width = Math.min(100, peak * 300) + "%";
    if (ws.readyState === 1) ws.send(pcm.buffer);
  };
  micCtx.createMediaStreamSource(stream).connect(worklet);
  worklet.connect(micCtx.destination);
  setState("listening");
  $("interrupt").disabled = false; $("stop").disabled = false; $("reconnect").disabled = false;
  $("barge").disabled = true;
}

function downsample(buf, from, to) {
  const ratio = from / to, out = new Float32Array(Math.floor(buf.length / ratio));
  for (let i = 0; i < out.length; i++) out[i] = buf[Math.floor(i * ratio)];
  return out;
}

function stopPlayback() {
  for (const s of sources) { try { s.stop(); } catch (e) {} }
  sources = []; playCursor = 0; uttStart = 0;
}

function playChunk(utt, pcm) {
  if (utt !== uttId) {                 // a new utt cuts the old one, unless the old one was an interjection: queue after it
    if (ijUtts.has(uttId)) uttStart = 0; else stopPlayback();
    uttId = utt;
  }
  const buf = playCtx.createBuffer(1, pcm.length, playCtx.sampleRate);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = playCtx.createBufferSource();
  src.buffer = buf; src.connect(playCtx.destination);
  const now = playCtx.currentTime;
  if (playCursor < now) playCursor = now + 0.008;     // 8 ms cushion on the first chunk
  if (uttStart === 0) {
    uttStart = playCursor;
    ws.send(JSON.stringify({ cmd: "play_start", utt: utt, delay_ms: (playCursor - now) * 1000 }));
  }
  src.start(playCursor);
  playCursor += pcm.length / playCtx.sampleRate;
  sources.push(src);
  src.onended = () => { sources = sources.filter((s) => s !== src); };
}

function addMsg(cls, text) {
  const d = document.createElement("div");
  d.className = "msg " + cls; d.textContent = text;
  $("convo").appendChild(d); $("convo").scrollTop = 1e9;
  return d;
}

function logRow(cells, cls) {
  const tb = $("log").tBodies[0];
  const tr = tb.insertRow();
  if (cls) tr.className = cls;
  cells.forEach((c) => { tr.insertCell().textContent = c; });
  while (tb.rows.length > 400) tb.deleteRow(0);
  tb.parentElement.scrollTop = 1e9;
}

function onMessage(ev) {
  if (ev.data instanceof ArrayBuffer) {
    const dv = new DataView(ev.data);
    playChunk(dv.getUint16(0, true), new Int16Array(ev.data, 4));
    return;
  }
  const m = JSON.parse(ev.data);
  if (m.type === "ready") {
    $("recinfo").textContent = `voxtral delay ${m.cfg.delay_ms} ms · peek + speculation on · barge-in ${m.cfg.barge_in}`;
    $("modeinfo").textContent = "turn head: P(complete) ≥ 0.9 for 3 frames" + (m.cfg.silence_ms ? `, or ${m.cfg.silence_ms} ms of silence` : ", no silence fallback") + "; the model decides every step";
  } else if (m.type === "head") {
    $("h_comp").style.width = (100 * m.p[1]).toFixed(0) + "%";
    $("h_bc").style.width = (100 * m.p[3]).toFixed(0) + "%";
    $("h_spk").style.width = (100 * m.p[0]).toFixed(0) + "%";
    $("headstate").textContent = (m.armed ? "armed " : "") + (m.bc ? "bc-latched" : "");
  } else if (m.type === "arm") {
    logRow([m.t.toFixed(2), "", "", "", "", `head armed endpoint (${m.reason}) at ${m.frame_ms} ms`], "ep");
  } else if (m.type === "peek") {
    logRow([m.t.toFixed(2), "", "", "", "", `peek: ${m.n_words} new word(s) in ${m.ms} ms GPU → LLM: ` + (m.text || "").slice(-40)], "ep");
  } else if (m.type === "barge") {
    logRow([m.t.toFixed(2), "", "", "", "", `head: user speaking at ${m.frame_ms} ms (P ${m.p[0].toFixed(2)}) → cutting reply`], "cut");
  } else if (m.type === "spec") {
    logRow([m.t.toFixed(2), "", "", "", "", `speculating at ${m.frame_ms} ms: peek + LLM + TTS, audio held`], "ep");
  } else if (m.type === "spec_discard") {
    logRow([m.t.toFixed(2), "", "", "", "", `speculation discarded (${m.why})`], "cut");
  } else if (m.type === "arm_cancel") {
    logRow([m.t.toFixed(2), "", "", "", "", "armed endpoint cancelled: user continued"], "cut");
  } else if (m.type === "backchannel") {
    if (userEl) { const t = document.createElement("span"); t.className = "tag"; t.textContent = "backchannel, ignored"; userEl.appendChild(t); }
    userEl = null;
    logRow([m.t.toFixed(2), "", "", "", "", "BACKCHANNEL ignored: " + m.text.slice(-40)], "cut");
  } else if (m.type === "word") {
    if (!userEl) userEl = addMsg("user", "");
    userEl.textContent += m.raw;
    if (m.lag_ms != null) stat.lag.push(m.lag_ms);
    logRow([m.t.toFixed(2), m.text, m.start, m.end, m.lag_ms == null ? "-" : m.lag_ms + " ms", ""]);
    if (stat.lag.length % 5 === 0) renderStats();
  } else if (m.type === "endpoint") {
    if (userEl) { const t = document.createElement("span"); t.className = "tag"; t.textContent = "→ LLM"; userEl.appendChild(t); }
    userEl = null; botEl = null;
    logRow([m.t.toFixed(2), "", "", "", "", `ENDPOINT (${m.reason || "head"}) → LLM: ` + m.text.slice(-40)], "ep");
  } else if (m.type === "interject") {
    ijUtts.add(m.utt);
    const el = addMsg("bot", m.text); el.classList.add("interject");
    userEl = null;                                  // the user's next words: a new bubble after it
    logRow([m.t.toFixed(2), "", "", "", "", "INTERJECT: " + m.text], "ep");
  } else if (m.type === "reply_start") {
    setState("thinking");
  } else if (m.type === "reply_delta") {
    if (!botEl) { botEl = addMsg("bot", ""); botRaw = ""; setState("speaking"); }
    botRaw += m.text;
    let txt = botEl.querySelector(".txt");      // the reply text lives in its own span so tags (search, cut) survive updates
    if (!txt) { txt = document.createElement("span"); txt.className = "txt"; botEl.prepend(txt); }
    txt.textContent = botRaw.replace(/<\|[a-z_]+:[a-z_]+\|>/g, "").replace(/<\|[^>]*$/, "").replace(/\s+/g, " ");   // TTS control tags
    $("convo").scrollTop = 1e9;
  } else if (m.type === "search") {
    if (!botEl) { botEl = addMsg("bot", ""); botRaw = ""; setState("speaking"); }
    const t = document.createElement("span"); t.className = "tag search"; t.textContent = "searching: " + m.query; botEl.appendChild(t);
    logRow([m.t.toFixed(2), "", "", "", "", "web_search: " + m.query], "ep");
  } else if (m.type === "search_done") {
    const t = botEl && botEl.querySelector(".tag.search");
    if (t) t.textContent += m.n < 0 ? ` (failed, ${m.ms} ms)` : ` (${m.n} results, ${m.ms} ms)`;
    logRow(["", "", "", "", "", `search done: ${m.n < 0 ? "FAILED" : m.n + " results"} in ${m.ms} ms`], m.n < 0 ? "cut" : "ep");
  } else if (m.type === "reply_end") {
    botEl = null; setState("listening");
  } else if (m.type === "stop_audio") {
    stopPlayback();
    if (botEl) { const t = document.createElement("span"); t.className = "tag"; t.textContent = "cut (" + m.reason + ")"; botEl.classList.add("cut"); botEl.appendChild(t); }
    botEl = null; setState("listening");
    logRow(["", "", "", "", "", "reply cut: " + m.reason], "cut");
  } else if (m.type === "gap") {
    if (m.premature) { stat.prem++; renderStats(); }
    logRow(["", "", "", "", "", `next word ${Math.round(m.gap_ms)} ms after turn ${m.turn}` + (m.premature ? " — PREMATURE endpoint" : "")], m.premature ? "cut" : "");
  } else if (m.type === "turn") {
    onTurn(m);
  } else if (m.type === "asr_reconnect") {
    logRow(["", "", "", "", "", `ASR reconnected (#${m.n}) at ${m.at_ms} ms`], "ep");
  } else if (m.type === "summary") {
    const { type, ...rest } = m;
    addMsg("bot", "session summary\n" + JSON.stringify(rest, null, 1));
  } else if (m.type === "error") {
    setState("error"); addMsg("bot", "error in " + m.where + ": " + JSON.stringify(m.detail));
  }
}

const rows = {};
function onTurn(m) {
  const tb = $("turns").tBodies[0];
  let tr = rows[m.turn];
  const fresh = !tr;
  if (fresh) { tr = tb.insertRow(); rows[m.turn] = tr; stat.turns++; }
  const f = (v) => (v == null ? "-" : v);
  const gap = m.next_gap_ms == null ? "-" : Math.round(m.next_gap_ms) + (m.premature ? " !" : "");
  const cells = [m.turn, f(m.word_end_to_asr), f(m.llm_ttft), f(m.first_token_to_tts_in), f(m.tts_ttfa), f(m.pcm_to_play),
                 f(m.word_end_to_ear), f(m.mic_offset_to_ear), gap, (m.cut ? "[" + m.cut + "] " : "") + (m.reply || "")];
  while (tr.cells.length) tr.deleteCell(0);
  cells.forEach((c, i) => { const td = tr.insertCell(); td.textContent = c; if (i === 6) td.className = "hi"; });
  if (m.cut) tr.classList.add("cutrow");
  if (fresh) {
    if (m.llm_ttft != null) stat.ttft.push(m.llm_ttft);
    if (m.first_token_to_tts_in != null) stat.clause.push(m.first_token_to_tts_in);
    if (m.tts_ttfa != null) stat.ttfa.push(m.tts_ttfa);
    if (m.pcm_to_play != null) stat.play.push(m.pcm_to_play);
    if (m.word_end_to_ear != null) stat.m2e.push(m.word_end_to_ear);
    if (m.cut && m.cut.startsWith("replaced")) stat.replaced++;
  } else if (m.word_end_to_ear != null && !tr.dataset.m2e) {
    stat.m2e.push(m.word_end_to_ear);
  }
  if (m.word_end_to_ear != null) tr.dataset.m2e = "1";
  tb.parentElement.scrollTop = 1e9;
  renderStats();
}

$("go").onclick = start;
$("interrupt").onclick = () => ws && ws.send(JSON.stringify({ cmd: "interrupt" }));
$("reconnect").onclick = () => ws && ws.send(JSON.stringify({ cmd: "reconnect_asr" }));
$("stop").onclick = () => {
  if (ws) ws.send(JSON.stringify({ cmd: "stop" }));
  stopPlayback();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (micCtx) micCtx.close();
  $("stop").disabled = true; $("interrupt").disabled = true; $("reconnect").disabled = true; $("go").disabled = false;
  $("barge").disabled = false;
  setState("idle");
};
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && ws && ws.readyState === 1 && document.activeElement.tagName !== "SELECT") { e.preventDefault(); $("interrupt").click(); }
});
