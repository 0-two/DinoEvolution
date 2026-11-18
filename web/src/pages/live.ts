/**
 * Live evolution page: the whole population on one canvas, the followed dino's network next
 * to it, a fitness chart below, and the controls.
 *
 * Time. The game advances in fixed 1/60 s ticks. At 1x, 4x and 16x each animation frame adds
 * the elapsed time (times the multiplier) to an accumulator and runs as many ticks as it
 * holds, so the display's frame rate never changes how the game plays. At "max" the frame
 * spends a fixed compute budget on simulation and skips drawing the game altogether.
 */
import { Trainer } from "../evolve/trainer";
import { FitnessChart } from "../render/fitnessChart";
import { NetworkGraph } from "../render/networkGraph";
import { PopulationCanvas } from "../render/populationCanvas";
import { formatInt } from "../render/svg";
import { TICKS_PER_SECOND } from "../sim/game";

const TICK_MS = 1000 / TICKS_PER_SECOND;
/** Longest a frame may spend simulating before it hands control back to the browser. */
const FRAME_BUDGET_MS = 12;
/** At max speed, how many ticks to run between clock reads. */
const MAX_BATCH = 32;
/** At max speed the game canvas is redrawn this often, so the page does not look frozen. */
const MAX_DRAW_INTERVAL_MS = 250;

type Speed = 1 | 4 | 16 | "max";
const SPEEDS: Speed[] = [1, 4, 16, "max"];

const TEMPLATE = `
<div class="grid">
  <section class="col-8 stage" aria-labelledby="h-pop">
    <h2 id="h-pop" class="label">Population</h2>
    <canvas id="game" role="img" aria-label="The population of dinos running the same course"></canvas>
  </section>
  <section class="col-4 stage" aria-labelledby="h-net">
    <h2 id="h-net" class="label">Followed dino</h2>
    <div id="network"></div>
    <p class="legend">Line width is weight size. Solid lines are positive weights, dashed lines negative. Nodes fill with activation.</p>
  </section>

  <p id="status" class="col-12 status"></p>

  <div class="col-12 controls">
    <div class="control-row">
      <button id="toggle" type="button" class="primary">Pause</button>
      <button id="reset" type="button">Reset</button>
      <label class="field">seed <input id="seed" type="number" min="0" step="1" inputmode="numeric" /></label>
      <div class="segmented" role="group" aria-label="Speed" id="speeds"></div>
    </div>
    <div class="control-row sliders">
      <label class="slider">
        <span>population <output id="o-pop"></output></span>
        <input id="pop" type="range" min="20" max="300" step="10" />
      </label>
      <label class="slider">
        <span>mutation rate <output id="o-rate"></output></span>
        <input id="rate" type="range" min="0" max="0.5" step="0.01" />
      </label>
      <label class="slider">
        <span>mutation size <output id="o-sigma"></output></span>
        <input id="sigma" type="range" min="0.05" max="1" step="0.05" />
      </label>
    </div>
    <p class="hint">Population size applies from the next generation. Rate is the chance that a weight is changed; size is the spread of the change.</p>
  </div>

  <section class="col-8 stage" aria-labelledby="h-chart">
    <h2 id="h-chart" class="label">Fitness by generation</h2>
    <div id="chart"></div>
  </section>
  <section class="col-4 stage" aria-labelledby="h-nums">
    <h2 id="h-nums" class="label">Numbers</h2>
    <dl class="numbers">
      <div><dt id="t-best">best fitness so far</dt><dd class="big" id="n-best">&ndash;</dd></div>
      <div><dt>last generation, best</dt><dd id="n-last-best">&ndash;</dd></div>
      <div><dt>last generation, mean</dt><dd id="n-last-mean">&ndash;</dd></div>
      <div><dt>simulation rate</dt><dd id="n-rate">0 x real time</dd></div>
    </dl>
  </section>
</div>
<p id="announce" class="sr-only" aria-live="polite"></p>
`;

function byId<T extends HTMLElement>(root: HTMLElement, id: string): T {
  const el = root.querySelector<T>(`#${id}`);
  if (el === null) throw new Error(`missing #${id}`);
  return el;
}

function pad(value: number | string, width: number): string {
  return String(value).padStart(width, " ");
}

/** Start values can be overridden from the address bar, e.g. `?pop=300&speed=16&seed=4`. */
function readParams() {
  const q = new URLSearchParams(window.location.search);
  const num = (key: string, fallback: number) => {
    const v = Number(q.get(key));
    return q.has(key) && Number.isFinite(v) ? v : fallback;
  };
  const speedParam = q.get("speed");
  const speed: Speed = speedParam === "max" ? "max" : (SPEEDS.find((s) => String(s) === speedParam) ?? 1);
  return {
    seed: Math.max(0, Math.floor(num("seed", 1))),
    pop: Math.min(300, Math.max(20, Math.floor(num("pop", 150)))),
    rate: num("rate", 0.1),
    sigma: num("sigma", 0.3),
    cap: Math.max(100, Math.floor(num("cap", 36_000))),
    speed,
  };
}

export function mountLive(root: HTMLElement): () => void {
  root.innerHTML = TEMPLATE;
  const params = readParams();

  const canvas = new PopulationCanvas(byId<HTMLCanvasElement>(root, "game"));
  const network = new NetworkGraph(byId(root, "network"));
  const chart = new FitnessChart(byId(root, "chart"));
  const trainer = new Trainer(
    { popSize: params.pop, mutationRate: params.rate, mutationSigma: params.sigma },
    { maxTicks: params.cap },
    params.seed,
  );

  const status = byId(root, "status");
  const toggle = byId<HTMLButtonElement>(root, "toggle");
  const seedInput = byId<HTMLInputElement>(root, "seed");
  const popInput = byId<HTMLInputElement>(root, "pop");
  const rateInput = byId<HTMLInputElement>(root, "rate");
  const sigmaInput = byId<HTMLInputElement>(root, "sigma");
  const speedGroup = byId(root, "speeds");
  const announce = byId(root, "announce");

  let speed: Speed = params.speed;
  let running = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let accumulator = 0;
  let lastFrame = performance.now();
  let drawnHistory = -1;
  let rateTicks = 0;
  let rateSince = performance.now();
  let rateText = "0";
  let lastDraw = 0;

  // --- controls ----------------------------------------------------------------------
  seedInput.value = String(params.seed);
  popInput.value = String(params.pop);
  rateInput.value = String(params.rate);
  sigmaInput.value = String(params.sigma);

  const speedButtons = SPEEDS.map((s) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = s === "max" ? "max" : `${s}×`;
    b.addEventListener("click", () => {
      speed = s;
      accumulator = 0;
      syncSpeed();
    });
    speedGroup.appendChild(b);
    return b;
  });

  function syncSpeed(): void {
    SPEEDS.forEach((s, i) => speedButtons[i].setAttribute("aria-pressed", String(s === speed)));
  }

  function syncToggle(): void {
    toggle.textContent = running ? "Pause" : "Run";
  }

  function syncSliders(): void {
    trainer.ga.config.popSize = Number(popInput.value);
    trainer.ga.config.mutationRate = Number(rateInput.value);
    trainer.ga.config.mutationSigma = Number(sigmaInput.value);
    byId(root, "o-pop").textContent = popInput.value;
    byId(root, "o-rate").textContent = Number(rateInput.value).toFixed(2);
    byId(root, "o-sigma").textContent = Number(sigmaInput.value).toFixed(2);
  }

  function seedValue(): number {
    const n = Math.floor(Number(seedInput.value));
    return Number.isFinite(n) && n >= 0 ? n : 0;
  }

  function reset(): void {
    seedInput.value = String(seedValue());
    trainer.reset(seedValue());
    drawnHistory = -1;
    accumulator = 0;
    announce.textContent = "Reset.";
  }

  toggle.addEventListener("click", () => {
    running = !running;
    lastFrame = performance.now();
    syncToggle();
  });
  byId(root, "reset").addEventListener("click", reset);
  seedInput.addEventListener("change", reset);
  for (const input of [popInput, rateInput, sigmaInput]) input.addEventListener("input", syncSliders);
  syncSpeed();
  syncToggle();
  syncSliders();

  // --- frame loop --------------------------------------------------------------------
  function advance(now: number): void {
    const dt = Math.min(now - lastFrame, 100); // a long pause must not replay minutes of game
    if (!running) return;
    const started = performance.now();
    let ticks = 0;
    if (speed === "max") {
      while (performance.now() - started < FRAME_BUDGET_MS) {
        for (let i = 0; i < MAX_BATCH; i++) trainer.step();
        ticks += MAX_BATCH;
      }
    } else {
      accumulator += dt * speed;
      while (accumulator >= TICK_MS) {
        trainer.step();
        accumulator -= TICK_MS;
        ticks += 1;
        if (performance.now() - started > FRAME_BUDGET_MS) {
          accumulator = 0; // too slow to keep up: drop the backlog instead of spiralling
          break;
        }
      }
    }
    rateTicks += ticks;
  }

  function updateRate(now: number): void {
    const elapsed = now - rateSince;
    if (elapsed >= 500) {
      rateText = formatInt(rateTicks / (elapsed / 1000) / TICKS_PER_SECOND);
      rateTicks = 0;
      rateSince = now;
    }
  }

  function setText(el: HTMLElement, text: string): void {
    if (el.textContent !== text) el.textContent = text;
  }

  function render(): void {
    const lead = trainer.leaderGame;
    const now = performance.now();
    if (speed !== "max" || now - lastDraw >= MAX_DRAW_INTERVAL_MS) {
      canvas.draw(trainer);
      lastDraw = now;
    }
    const g = trainer.agents[trainer.leader];
    network.setGenome(g.genome);
    network.update(g.act, g.action);

    setText(
      status,
      `gen ${String(trainer.generation + 1).padStart(3, "0")}   round ${trainer.round + 1}/${trainer.config.rounds}   ` +
        `alive ${pad(trainer.aliveCount, 3)}/${trainer.popSize}   best ${pad(formatInt(trainer.roundBest), 6)}   ` +
        `speed ${lead.speed.toFixed(1)}${speed === "max" ? "   drawn 4/s" : ""}`,
    );

    if (trainer.history.length !== drawnHistory) {
      drawnHistory = trainer.history.length;
      chart.render(trainer.history);
      const last = trainer.history[trainer.history.length - 1];
      if (last !== undefined) {
        setText(byId(root, "n-last-best"), formatInt(last.best));
        setText(byId(root, "n-last-mean"), formatInt(last.mean));
        announce.textContent = `Generation ${last.generation} finished. Best ${formatInt(last.best)}, mean ${formatInt(last.mean)}.`;
      } else {
        setText(byId(root, "n-last-best"), "\u2013");
        setText(byId(root, "n-last-mean"), "\u2013");
      }
      const best = trainer.bestEver;
      setText(byId(root, "n-best"), best === null ? "\u2013" : formatInt(best.fitness));
      setText(byId(root, "t-best"), best === null ? "best fitness so far" : `best fitness so far, generation ${best.generation}`);
    }
    setText(byId(root, "n-rate"), `${rateText} × real time`);
  }

  let frameId = 0;
  function frame(now: number): void {
    advance(now);
    lastFrame = now;
    updateRate(now);
    render();
    frameId = requestAnimationFrame(frame);
  }
  frameId = requestAnimationFrame(frame);

  return () => cancelAnimationFrame(frameId);
}
