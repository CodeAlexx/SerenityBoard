#!/usr/bin/env node
// Headless-Chrome driver over the DevTools protocol (no npm deps; Node >= 22 WebSocket).
//   node scripts/ui_drive.mjs <url> <script.js> [outdir]
// The script file is evaluated in the page (async allowed) and may call:
//   await shot(name)          -> saves <outdir>/<name>.png
//   await sleep(ms)
//   log(...)                  -> forwarded to this process's stdout
// Its final expression / returned JSON is printed as RESULT.
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { setTimeout as delay } from "node:timers/promises";

const [url, scriptPath, outdir = "evidence/ui"] = process.argv.slice(2);
if (!url || !scriptPath) { console.error("usage: ui_drive.mjs <url> <script.js> [outdir]"); process.exit(64); }
mkdirSync(outdir, { recursive: true });
const port = 9333 + Math.floor(Math.random() * 500);
const chrome = spawn("google-chrome", [
  "--headless=new", `--remote-debugging-port=${port}`, "--no-first-run", "--no-default-browser-check",
  "--window-size=1500,950", "--disable-gpu", "--user-data-dir=/tmp/difc-ui-chrome-" + port, "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
let chromeErr = ""; chrome.stderr.on("data", d => { chromeErr += d; });
process.on("exit", () => chrome.kill("SIGKILL"));

async function getTargets() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(`http://127.0.0.1:${port}/json/list`); return await r.json(); } catch { await delay(100); }
  }
  throw new Error("chrome did not expose devtools: " + chromeErr.slice(-500));
}
const targets = await getTargets();
const page = targets.find(t => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let nextId = 1; const pending = new Map(); const events = [];
ws.onmessage = ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result); }
  else if (m.method) {
    events.push(m);
    if (m.method === "Runtime.consoleAPICalled") {
      const text = m.params.args.map(a => a.value ?? a.description ?? "").join(" ");
      if (text.startsWith("[ui]")) console.log(text);
      else consoleLog.push(`${m.params.type}: ${text}`);
    }
    if (m.method === "Runtime.exceptionThrown") consoleLog.push("EXCEPTION: " + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
  }
};
const consoleLog = [];
const send = (method, params = {}) => new Promise((res, rej) => { const id = nextId++; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });
await send("Page.enable"); await send("Runtime.enable"); await send("Log.enable");
await send("Page.navigate", { url });
await delay(1500);
async function shot(name) {
  const { data } = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(`${outdir}/${name}.png`, Buffer.from(data, "base64"));
  console.log(`[shot] ${outdir}/${name}.png`);
}
const script = readFileSync(scriptPath, "utf8");
// Expose shot/sleep via a binding-like polling loop: the page sets window.__difc_req, we serve it.
await send("Runtime.evaluate", { expression: `window.__difc = { queue: [], results: {} };
  window.shot = (name) => new Promise(r => { window.__difc.queue.push({kind:'shot', name}); const t=setInterval(()=>{ if(window.__difc.results[name]){clearInterval(t); r();} },50); });
  window.sleep = (ms) => new Promise(r => setTimeout(r, ms));
  window.log = (...a) => console.log('[ui]', ...a.map(x => typeof x === 'string' ? x : JSON.stringify(x)));` });
const run = send("Runtime.evaluate", { expression: `(async () => { ${script} })()`, awaitPromise: true, returnByValue: true, timeout: 600000 });
let finished = false; run.finally(() => { finished = true; });
while (!finished) {
  const { result } = await send("Runtime.evaluate", { expression: "JSON.stringify(window.__difc.queue.splice(0))", returnByValue: true });
  for (const req of JSON.parse(result.value || "[]")) {
    if (req.kind === "shot") { await shot(req.name); await send("Runtime.evaluate", { expression: `window.__difc.results[${JSON.stringify(req.name)}]=true` }); }
  }
  await delay(100);
}
try {
  const r = await run;
  console.log("RESULT " + JSON.stringify(r.result.value ?? null));
  if (r.exceptionDetails) console.log("SCRIPT EXCEPTION " + JSON.stringify(r.exceptionDetails).slice(0, 2000));
} catch (e) { console.log("SCRIPT ERROR " + e.message); }
if (consoleLog.length) console.log("CONSOLE:\n" + consoleLog.slice(0, 60).join("\n"));
ws.close(); chrome.kill("SIGKILL"); process.exit(0);
