// Accessibility and privacy check, run against a served build.
//
//   npm run build && npx vite preview --port 4173
//   npm run a11y -- --url http://localhost:4173/
//
// For every page, in the light and the dark theme, this
//   1. runs axe-core with the WCAG 2.0/2.1 level A and AA rules and lists the violations,
//   2. presses Tab through the page and checks that every focused element shows a focus
//      indicator (a non-zero outline, or a visible box-shadow-free alternative),
//   3. records every network request and flags any that leaves the local server,
//   4. with `prefers-reduced-motion: reduce`, checks that the live page starts paused.
// Exit status is 1 if any of these fail.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, "")] = process.argv[i + 1];
const base = (args.url ?? "http://localhost:4173/").replace(/\/$/, "");
const require = createRequire(import.meta.url);
const axeSource = readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");
const browser = [
  args.browser,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
]
  .filter(Boolean)
  .find((p) => existsSync(p));
if (!browser) throw new Error("no Chrome or Edge found; pass --browser <path>");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = 9500 + Math.floor(Math.random() * 300);
const profile = mkdtempSync(join(tmpdir(), "dino-a11y-"));
const proc = spawn(
  browser,
  ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "--window-size=1200,900", "about:blank"],
  { stdio: "ignore" },
);

let failed = false;
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
  const requests = new Set();
  const pageErrors = [];
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) {
      pending.get(d.id)(d);
      pending.delete(d.id);
    }
    if (d.method === "Network.requestWillBeSent") requests.add(d.params.request.url);
    if (d.method === "Runtime.exceptionThrown") pageErrors.push(d.params.exceptionDetails.text);
  };
  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const id = ++nextId;
      pending.set(id, resolve);
      ws.send(JSON.stringify({ id, method, params }));
    });
  const evaluate = async (expression) =>
    (await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true })).result.result?.value;
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Network.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1200, height: 900, deviceScaleFactor: 1, mobile: false });

  const pages = ["live", "watch", "how", "results"];
  for (const scheme of ["light", "dark"]) {
    await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: scheme }] });
    for (const page of pages) {
      // A new query string each time forces a full load; a hash-only change would reuse the page.
      await send("Page.navigate", { url: `${base}/?pop=60&check=${scheme}-${page}#/${page}` });
      await sleep(3500);

      // 1. axe-core
      await evaluate(axeSource);
      const axe = await evaluate(`axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } })
        .then(r => ({ violations: r.violations.map(v => ({ id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.length,
          sample: v.nodes.slice(0, 3).map(n => n.target.join(" ") + " :: " + (n.any[0]?.message ?? n.all[0]?.message ?? "")) })), passes: r.passes.length, incomplete: r.incomplete.length }))`);
      console.log(`${scheme.padEnd(5)} ${page.padEnd(8)} axe: ${axe.violations.length} violations, ${axe.passes} rules passed, ${axe.incomplete.length ?? axe.incomplete} need review`);
      for (const v of axe.violations) {
        failed = true;
        console.log(`   [${v.impact}] ${v.id}: ${v.help} (${v.nodes} nodes)`);
        for (const s of v.sample) console.log(`       ${s}`);
      }

      // 2. keyboard focus indicators
      await evaluate(`document.activeElement && document.activeElement.blur(); window.scrollTo(0, 0); 0`);
      const focusProblems = [];
      let stops = 0;
      for (let i = 0; i < 45; i++) {
        await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
        await send("Input.dispatchKeyEvent", { type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
        const info = await evaluate(`(() => { const e = document.activeElement; if (!e || e === document.body) return null;
          const s = getComputedStyle(e); return { tag: e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") + (e.getAttribute("class") ? "." + e.getAttribute("class").split(" ")[0] : ""),
            outline: s.outlineStyle + " " + s.outlineWidth, label: (e.getAttribute("aria-label") || e.textContent || "").trim().slice(0, 30) }; })()`);
        if (info === null) continue;
        stops++;
        if (info.outline.startsWith("none") || info.outline.endsWith(" 0px")) focusProblems.push(`${info.tag} "${info.label}" outline ${info.outline}`);
      }
      console.log(`      keyboard: ${stops} focus stops, ${focusProblems.length} without a visible outline`);
      for (const p of focusProblems) {
        failed = true;
        console.log(`       ${p}`);
      }
    }
  }

  // 3. network requests
  const external = [...requests].filter((u) => !u.startsWith(base) && !u.startsWith("data:") && !u.startsWith("blob:"));
  console.log(`\nnetwork: ${requests.size} requests, ${external.length} outside ${base}`);
  for (const u of external) {
    failed = true;
    console.log(`   external: ${u}`);
  }
  if (pageErrors.length) {
    failed = true;
    console.log("page errors:", pageErrors);
  }

  // 4. reduced motion
  await send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-motion", value: "reduce" }, { name: "prefers-color-scheme", value: "light" }],
  });
  await send("Page.navigate", { url: `${base}/?pop=60&check=reduced-motion#/live` });
  await sleep(3000);
  await send("Page.bringToFront");
  await sleep(1500);
  const visibility = await evaluate("document.visibilityState");
  const rm = await evaluate(`({ button: document.getElementById("toggle").textContent, status: document.getElementById("status").textContent })`);
  console.log(`   (document.visibilityState: ${visibility})`);
  const paused = rm.button === "Run" && /gen 001 +round 1\/3/.test(rm.status) && /best +0/.test(rm.status);
  console.log(`reduced motion: button "${rm.button}", status "${rm.status.replace(/\s+/g, " ")}" -> ${paused ? "starts paused" : "NOT paused"}`);
  if (!paused) failed = true;
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
console.log(failed ? "\nFAILED" : "\nall accessibility and privacy checks passed");
process.exit(failed ? 1 : 0);
