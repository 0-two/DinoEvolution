// Frame-pacing benchmark for the live page.
//
//   npm run build
//   npx vite preview --port 4173
//   node scripts/perf.mjs --url http://localhost:4173/ [--browser <path to Chrome or Edge>]
//
// Each case starts a fresh headless browser, loads the page with a given population and
// speed, and records requestAnimationFrame intervals for a few seconds. Results go to
// ../results/benchmarks/web_perf.json. Headless runs use software rendering, which is slower
// than a normal desktop session, so the numbers are a pessimistic bound.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { cpus, platform, release, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, "")] = process.argv[i + 1];
const url = args.url ?? "http://localhost:4173/";
const SECONDS = Number(args.seconds ?? 6);
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
const withTimeout = (promise, ms, what) =>
  Promise.race([promise, new Promise((_, rej) => setTimeout(() => rej(new Error(`timeout: ${what}`)), ms))]);

async function runCase(pop, speed) {
  const port = 9700 + Math.floor(Math.random() * 200);
  const profile = mkdtempSync(join(tmpdir(), "dino-perf-"));
  const proc = spawn(
    browser,
    ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "--window-size=1200,900", "about:blank"],
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
    const info = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
    const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
    await new Promise((r) => (ws.onopen = r));
    let nextId = 0;
    const pending = new Map();
    const errors = [];
    ws.onmessage = (m) => {
      const d = JSON.parse(m.data);
      if (d.id && pending.has(d.id)) {
        pending.get(d.id)(d);
        pending.delete(d.id);
      }
      if (d.method === "Runtime.exceptionThrown") errors.push(d.params.exceptionDetails.text);
    };
    const send = (method, params = {}) =>
      new Promise((resolve) => {
        const id = ++nextId;
        pending.set(id, resolve);
        ws.send(JSON.stringify({ id, method, params }));
      });
    await send("Runtime.enable");
    await send("Page.enable");
    await send("Emulation.setDeviceMetricsOverride", { width: 1200, height: 900, deviceScaleFactor: 1, mobile: false });
    await send("Page.navigate", { url: `${url}?pop=${pop}&speed=${speed}` });
    await sleep(1500);
    const res = await withTimeout(
      send("Runtime.evaluate", {
        awaitPromise: true,
        returnByValue: true,
        expression: `new Promise((resolve) => {
          const deltas = []; let last = performance.now(); const end = last + ${SECONDS * 1000};
          function f(t) {
            deltas.push(t - last); last = t;
            if (t < end) return requestAnimationFrame(f);
            const s = deltas.slice(2).sort((a, b) => a - b);
            const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
            resolve({ frames: s.length, mean_ms: s.reduce((a, b) => a + b, 0) / s.length,
              p50_ms: q(0.5), p95_ms: q(0.95), p99_ms: q(0.99), max_ms: s[s.length - 1],
              rate: document.getElementById('n-rate').textContent });
          }
          requestAnimationFrame(f);
        })`,
      }),
      (SECONDS + 15) * 1000,
      `case ${pop}/${speed}`,
    );
    ws.close();
    return { value: res.result.result.value, errors, browserVersion: info.Browser };
  } finally {
    proc.kill();
    await sleep(500);
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {
      /* the browser may still hold files; the OS temp cleaner will remove them */
    }
  }
}

const cases = [
  { pop: 150, speed: "1" },
  { pop: 300, speed: "1" },
  { pop: 300, speed: "4" },
  { pop: 300, speed: "16" },
  { pop: 300, speed: "max" },
];
const results = [];
const pageErrors = [];
let browserVersion = "";
for (const c of cases) {
  const { value: v, errors, browserVersion: bv } = await runCase(c.pop, c.speed);
  browserVersion = bv;
  pageErrors.push(...errors);
  results.push({
    population: c.pop,
    speed: c.speed === "max" ? "max" : `${c.speed}x`,
    fps: 1000 / v.mean_ms,
    frame_ms: { p50: v.p50_ms, p95: v.p95_ms, p99: v.p99_ms, max: v.max_ms },
    simulation_rate: v.rate,
  });
  console.log(c.pop, c.speed, `${(1000 / v.mean_ms).toFixed(1)} fps`, `p99 ${v.p99_ms.toFixed(1)} ms`, v.rate);
}

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const out = join(root, "results", "benchmarks", "web_perf.json");
mkdirSync(dirname(out), { recursive: true });
writeFileSync(
  out,
  JSON.stringify(
    {
      browser: browserVersion,
      rendering: "headless, software (--disable-gpu)",
      machine: { cpu: cpus()[0]?.model, cores: cpus().length, os: `${platform()} ${release()}` },
      seconds_per_case: SECONDS,
      page_errors: pageErrors,
      results,
    },
    null,
    2,
  ) + "\n",
);
console.log(`wrote ${out}`);
