/**
 * Watch a trained agent: pick an algorithm and one of its runs, and play that run's best genome
 * on any seed. The genomes are the files the Python experiments exported to `results/`, so what
 * plays here is what was tested there. The default seed is the first of the 100 test seeds,
 * which no algorithm trained or was tuned on.
 */
import { genomeIndex as index, genomeLoader as loaderFor } from "../evolve/genomeSource";
import { loadPolicy, type Policy } from "../evolve/policies";
import { NetworkGraph } from "../render/networkGraph";
import { GameView } from "../render/gameView";
import { formatInt } from "../render/svg";
import { DinoGame, TICKS_PER_SECOND } from "../sim/game";

const TICK_MS = 1000 / TICKS_PER_SECOND;
const FRAME_BUDGET_MS = 12;
const RESTART_DELAY_TICKS = 90;
const FIRST_TEST_SEED = 1_000_000;

const LABELS: Record<string, string> = {
  ga: "GA",
  cmaes: "CMA-ES",
  neat: "NEAT",
  ppo: "PPO",
  rule_fixed: "rule, fixed distance",
  rule_scaled: "rule, speed-scaled",
};
const ORDER = ["ga", "cmaes", "neat", "ppo", "rule_fixed", "rule_scaled"];

const TEMPLATE = `
<div class="grid">
  <section class="col-8 stage" aria-labelledby="h-agent">
    <h2 id="h-agent" class="label">Agent</h2>
    <canvas id="game" role="img" aria-label="A trained agent playing the game"></canvas>
  </section>
  <section class="col-4 stage" aria-labelledby="h-info">
    <h2 id="h-info" class="label">Policy</h2>
    <div id="summary" class="summary"></div>
    <div id="network"></div>
    <p id="note" class="legend"></p>
  </section>

  <p id="status" class="col-12 status"></p>

  <div class="col-12 controls">
    <div class="control-row">
      <label class="field">algorithm <select id="algo"></select></label>
      <label class="field">run <select id="run"></select></label>
      <label class="field">game seed <input id="seed" type="number" min="0" step="1" inputmode="numeric" /></label>
    </div>
    <div class="control-row">
      <button id="toggle" type="button" class="primary">Pause</button>
      <button id="reset" type="button">Reset</button>
      <div class="segmented" role="group" aria-label="Speed" id="speeds"></div>
    </div>
    <p class="hint">Runs are listed by their score on the 100 unseen test seeds, best first. The default game seed is the first of those test seeds. After a crash the next seed starts.</p>
  </div>
</div>
`;

function byId<T extends HTMLElement>(root: HTMLElement, id: string): T {
  const el = root.querySelector<T>(`#${id}`);
  if (el === null) throw new Error(`missing #${id}`);
  return el;
}

export function mountWatch(root: HTMLElement): () => void {
  root.innerHTML = TEMPLATE;
  const view = new GameView(byId<HTMLCanvasElement>(root, "game"));
  const graphHost = byId(root, "network");
  const graph = new NetworkGraph(graphHost);
  const summary = byId(root, "summary");
  const note = byId(root, "note");
  const status = byId(root, "status");
  const algoSelect = byId<HTMLSelectElement>(root, "algo");
  const runSelect = byId<HTMLSelectElement>(root, "run");
  const seedInput = byId<HTMLInputElement>(root, "seed");
  const toggle = byId<HTMLButtonElement>(root, "toggle");
  const speedGroup = byId(root, "speeds");

  if (index.length === 0) {
    root.replaceChildren();
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent =
      "No trained genomes are bundled with this build. Run `make experiments` and `make figures`, then rebuild the page.";
    root.appendChild(p);
    return () => undefined;
  }

  const game = new DinoGame(FIRST_TEST_SEED);
  let policy: Policy | null = null;
  let running = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let speed = 1;
  let accumulator = 0;
  let last = performance.now();
  let deadTicks = 0;
  let disposed = false;
  let loadToken = 0;

  seedInput.value = String(FIRST_TEST_SEED);

  // --- selectors -----------------------------------------------------------------------
  const algos = ORDER.filter((a) => index.some((e) => e.run_id === a));
  for (const a of algos) algoSelect.add(new Option(LABELS[a] ?? a, a));

  function fillRuns(): void {
    const runs = index.filter((e) => e.run_id === algoSelect.value).sort((a, b) => b.test_mean - a.test_mean);
    runSelect.replaceChildren();
    runs.forEach((r, i) => {
      const text = `seed ${r.seed}, test ${formatInt(r.test_mean)}${i === 0 ? " (best)" : ""}`;
      runSelect.add(new Option(text, r.genome_file));
    });
  }

  async function loadSelected(): Promise<void> {
    const token = ++loadToken;
    const file = runSelect.value;
    const loader = loaderFor(file);
    if (loader === undefined) {
      status.textContent = `genome file ${file} is not bundled with this build`;
      return;
    }
    try {
      const payload = await loader();
      if (token !== loadToken || disposed) return;
      policy = loadPolicy(payload);
      summary.replaceChildren(
        ...policy.summary.map((line) => {
          const p = document.createElement("p");
          p.textContent = line;
          return p;
        }),
      );
      const drawable = policy.genome !== undefined;
      graphHost.hidden = !drawable;
      note.textContent = drawable
        ? "Line width is weight size. Solid lines are positive weights, dashed lines negative. Nodes fill with activation."
        : "The network drawing is available for the fixed 9-8-3 networks of the GA and CMA-ES.";
      if (policy.genome !== undefined) graph.setGenome(policy.genome);
      restart();
    } catch (err) {
      status.textContent = `could not load ${file}: ${(err as Error).message}`;
    }
  }

  function seedValue(): number {
    const n = Math.floor(Number(seedInput.value));
    return Number.isFinite(n) && n >= 0 ? n : FIRST_TEST_SEED;
  }

  function restart(): void {
    seedInput.value = String(seedValue());
    game.reset(seedValue());
    deadTicks = 0;
    accumulator = 0;
  }

  algoSelect.addEventListener("change", () => {
    fillRuns();
    void loadSelected();
  });
  runSelect.addEventListener("change", () => void loadSelected());
  seedInput.addEventListener("change", restart);
  byId(root, "reset").addEventListener("click", restart);
  toggle.addEventListener("click", () => {
    running = !running;
    last = performance.now();
    toggle.textContent = running ? "Pause" : "Run";
  });
  toggle.textContent = running ? "Pause" : "Run";

  const speeds = [1, 4, 16];
  const buttons = speeds.map((s) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = `${s}×`;
    b.addEventListener("click", () => {
      speed = s;
      accumulator = 0;
      sync();
    });
    speedGroup.appendChild(b);
    return b;
  });
  const sync = () => speeds.forEach((s, i) => buttons[i].setAttribute("aria-pressed", String(s === speed)));
  sync();

  fillRuns();
  void loadSelected();

  // --- loop ----------------------------------------------------------------------------
  function tick(): void {
    if (game.done) {
      deadTicks += 1;
      if (deadTicks >= RESTART_DELAY_TICKS) {
        seedInput.value = String(seedValue() + 1);
        restart();
      }
      return;
    }
    if (policy !== null) game.step(policy.act(game));
  }

  let frameId = 0;
  function frame(now: number): void {
    const dt = Math.min(now - last, 100);
    last = now;
    if (running) {
      accumulator += dt * speed;
      const started = performance.now();
      while (accumulator >= TICK_MS) {
        tick();
        accumulator -= TICK_MS;
        if (performance.now() - started > FRAME_BUDGET_MS) {
          accumulator = 0;
          break;
        }
      }
    }
    view.draw(game);
    if (policy?.activations !== undefined) graph.update(policy.activations, policy.lastAction);
    const state = game.dead ? "dead" : game.done ? "cap" : "run";
    const text =
      `seed ${String(game.seed).padStart(7, " ")}   tick ${String(game.tick).padStart(5, "0")}   ` +
      `score ${formatInt(game.score).padStart(6, " ")}   speed ${game.speed.toFixed(1)}   ${state}`;
    if (status.textContent !== text && policy !== null) status.textContent = text;
    frameId = requestAnimationFrame(frame);
  }
  frameId = requestAnimationFrame(frame);

  return () => {
    disposed = true;
    cancelAnimationFrame(frameId);
  };
}
