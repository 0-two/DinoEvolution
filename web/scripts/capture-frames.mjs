// Capture frames of the live evolution page for the README animation.
//
//   npm run build && npx vite preview --port 4173
//   node scripts/capture-frames.mjs --url http://localhost:4173/ --out ../report/media/frames
//   uv run python -m sim.figures.gif          (from the repository root; assembles the GIF)
//
// A headless browser loads the live page at 4x speed and a screenshot is taken every
// `--interval` milliseconds of real time. The frames are cropped to the population view, the
// network drawing, the status line and the fitness chart.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, "")] = process.argv[i + 1];
const url = args.url ?? "http://localhost:4173/";
const out = resolve(args.out ?? "../report/media/frames");
const FRAMES = Number(args.frames ?? 190);
const INTERVAL = Number(args.interval ?? 120);
const SPEED = args.speed ?? "4";
const browser = [
  args.browser,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
  .filter(Boolean)
  .find((p) => existsSync(p));
if (!browser) throw new Error("no Chrome or Edge found; pass --browser <path>");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = 9100 + Math.floor(Math.random() * 300);
const profile = mkdtempSync(join(tmpdir(), "dino-gif-"));
const proc = spawn(
  browser,
  ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "--window-size=1200,1000", "about:blank"],
  { stdio: "ignore" },
);

try {
  let targets;
  for (let i = 0; i < 60; i++) {
    try {
      targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      if (targets.length) break;
    } catch {
      /* browser still starting */
    }
    await sleep(200);
  }
  const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let nextId = 0;
  const pending = new Map();
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) {
      pending.get(d.id)(d);
      pending.delete(d.id);
    }
  };
  const send = (method, params = {}) =>
    new Promise((resolveCall) => {
      const id = ++nextId;
      pending.set(id, resolveCall);
      ws.send(JSON.stringify({ id, method, params }));
    });
  await send("Page.enable");
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
  await send("Emulation.setDeviceMetricsOverride", { width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `${url}?speed=${SPEED}&seed=1#/live` });
  await sleep(600);

  mkdirSync(out, { recursive: true });
  const clip = { x: 56, y: 70, width: 1090, height: 900, scale: 0.7 };
  const started = Date.now();
  for (let i = 0; i < FRAMES; i++) {
    const wanted = started + i * INTERVAL;
    await sleep(Math.max(0, wanted - Date.now()));
    const shot = await send("Page.captureScreenshot", { format: "png", clip });
    writeFileSync(join(out, `frame_${String(i).padStart(4, "0")}.png`), Buffer.from(shot.result.data, "base64"));
  }
  console.log(`wrote ${FRAMES} frames to ${out} in ${((Date.now() - started) / 1000).toFixed(1)} s`);
  ws.close();
} finally {
  proc.kill();
  await sleep(500);
  try {
    rmSync(profile, { recursive: true, force: true });
  } catch {
    /* the OS temp cleaner will remove it */
  }
}
