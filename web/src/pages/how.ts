/**
 * How it works: a short explanation of the loop, the network's inputs, why fitness is averaged
 * over seeds, and what mutation size does. The widgets run the real game and network.
 */
import { DEFAULT_GA } from "../evolve/ga";
import { bestRun, genomeLoader } from "../evolve/genomeSource";
import { INPUT_LABELS, GENOME_LENGTH } from "../evolve/network";
import { loadPolicy } from "../evolve/policies";
import { DEFAULT_TRAINER } from "../evolve/trainer";
import { mutate, relativeChildScore, ruleScore, winProbability } from "../evolve/toy";
import { GameView } from "../render/gameView";
import { MiniChart } from "../render/miniChart";
import { NetworkGraph } from "../render/networkGraph";
import { DINO_W, DINO_X, DinoGame, Mulberry32, START_SPEED, MAX_SPEED } from "../sim/game";

const loopFigures = import.meta.glob("../../../report/figures/loop.svg", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

/** A labelled range input with a live readout. */
function slider(
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  fmt: (v: number) => string,
  onInput: (v: number) => void,
): HTMLElement {
  const wrap = el("label", { class: "slider" });
  const out = el("output");
  const head = el("span");
  head.append(`${label} `, out);
  const input = el("input", { type: "range", min: String(min), max: String(max), step: String(step) });
  input.value = String(value);
  out.textContent = fmt(value);
  input.addEventListener("input", () => {
    const v = Number(input.value);
    out.textContent = fmt(v);
    onInput(v);
  });
  wrap.append(head, input);
  return wrap;
}

function section(title: string, text: string[], widget: HTMLElement | null): HTMLElement {
  const s = el("section", { class: "result" });
  s.appendChild(el("h2", { class: "result-title" }, title));
  const grid = el("div", { class: "grid result-grid" });
  const side = el("div", { class: "col-4 figure-note" });
  for (const t of text) side.appendChild(el("p", {}, t));
  const main = el("div", { class: "col-8 widget" });
  if (widget !== null) main.appendChild(widget);
  grid.append(side, main);
  s.appendChild(grid);
  return s;
}

// --- 1. the loop ---------------------------------------------------------------------------------
function loopSection(): HTMLElement {
  const holder = el("div", { class: "figure" });
  const svgText = Object.values(loopFigures)[0];
  if (svgText !== undefined) {
    holder.innerHTML = svgText;
    holder.querySelector("svg")?.removeAttribute("width");
    holder.querySelector("svg")?.removeAttribute("height");
  }
  return section(
    "The loop has four steps",
    [
      `A population of ${DEFAULT_GA.popSize} genomes starts with random weights. Each generation plays every genome, keeps the best ${DEFAULT_GA.eliteCount} unchanged, and fills the rest with children of tournament winners.`,
      `A child takes each weight from one parent or the other, then each weight is changed with probability ${DEFAULT_GA.mutationRate} by Gaussian noise. The children are the next generation. Nothing is learned from a single game: the population drifts toward genomes that survive longer.`,
    ],
    holder,
  );
}

// --- 2. the inputs -------------------------------------------------------------------------------
const OBSTACLES = {
  "small cactus": { y: 0, w: 17, h: 35 },
  "large cactus group": { y: 0, w: 75, h: 50 },
  "bird, middle height": { y: 30, w: 46, h: 30 },
} as const;

function inputsSection(): HTMLElement {
  const root = el("div", { class: "inputs-widget" });
  const canvas = el("canvas", { role: "img", "aria-label": "A dino and the next obstacle" });
  const tableHost = el("div");
  const controls = el("div", { class: "widget-controls" });
  root.append(canvas, controls, tableHost);
  const view = new GameView(canvas);
  const state = { distance: 140, speed: 8, height: 0, kind: "small cactus" as keyof typeof OBSTACLES };

  const game = new DinoGame(0, 1000, false);
  const rows: HTMLElement[] = [];
  const table = el("table", { class: "data compact" });
  table.appendChild(el("caption", {}, "Network inputs for this state"));
  const body = el("tbody");
  INPUT_LABELS.forEach((label) => {
    const tr = el("tr");
    const v = el("td", { class: "num" });
    tr.append(el("th", { scope: "row" }, label), v);
    body.appendChild(tr);
    rows.push(v);
  });
  table.appendChild(body);
  tableHost.appendChild(table);

  function refresh(): void {
    const o = OBSTACLES[state.kind];
    game.speed = state.speed;
    game.y = state.height;
    game.airborne = state.height > 0;
    game.obstacles = [
      [DINO_X + DINO_W + state.distance, o.y, o.w, o.h],
      [DINO_X + DINO_W + state.distance + 220, 0, 17, 35],
    ];
    view.draw(game);
    game.observe().forEach((v, i) => {
      rows[i].textContent = v.toFixed(3);
    });
  }

  const kind = el("select", { "aria-label": "Obstacle" });
  for (const k of Object.keys(OBSTACLES)) kind.add(new Option(k, k));
  kind.addEventListener("change", () => {
    state.kind = kind.value as keyof typeof OBSTACLES;
    refresh();
  });
  const kindLabel = el("label", { class: "field" });
  kindLabel.append("obstacle ", kind);
  controls.append(
    kindLabel,
    slider("distance", 0, 400, 5, state.distance, (v) => `${v} px`, (v) => { state.distance = v; refresh(); }),
    slider("speed", START_SPEED, MAX_SPEED, 0.5, state.speed, (v) => v.toFixed(1), (v) => { state.speed = v; refresh(); }),
    slider("dino height", 0, 90, 5, state.height, (v) => `${v} px`, (v) => { state.height = v; refresh(); }),
  );
  refresh();

  return section(
    "The network sees nine numbers",
    [
      "Each tick the network gets the distance to the next obstacle, its width, height and height above ground, the distance to the one after it, the game speed, the dino's height and vertical velocity, and a constant 1. All are scaled to roughly 0 to 1.",
      "The network picks the action with the largest output: nothing, jump or duck. Move the sliders to see which inputs change. The dino's vertical velocity is zero here because the state is set by hand.",
    ],
    root,
  );
}

// --- 3. seed averaging ---------------------------------------------------------------------------
const POOL = 40;
const SEED_CAP = 3000;
const CANDIDATES = [50, 70, 90, 110, 130, 150];

function seedsSection(): HTMLElement {
  const root = el("div", { class: "seeds-widget" });
  const status = el("p", { class: "readout" }, "Playing the agents on 40 seeds");
  const out = el("p", { class: "readout" });
  const chartHost = el("div");
  const controls = el("div", { class: "widget-controls" });
  root.append(status, controls, out, chartHost);

  const chart = new MiniChart(
    chartHost,
    {
      xLabel: "seeds per fitness, K",
      yLabel: "chance the worse agent ranks first",
      xTicks: [1, 2, 4, 6, 8, 10],
      yTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5],
      yFormat: (v) => `${Math.round(v * 100)}%`,
    },
    "Chance that the worse agent scores higher on the average of K seeds, for K from 1 to 10",
  );

  const scores = new Map<number, number[]>();
  const means = new Map<number, number>();
  let a = CANDIDATES[2];
  let b = CANDIDATES[3];
  let k = 3;

  function update(): void {
    if (scores.size === 0) return;
    const sa = scores.get(a) ?? [];
    const sb = scores.get(b) ?? [];
    const [better, worse] = (means.get(a) ?? 0) >= (means.get(b) ?? 0) ? [sa, sb] : [sb, sa];
    const curve: [number, number][] = [];
    for (let kk = 1; kk <= 10; kk++) curve.push([kk, winProbability(better, worse, kk, 3000, 11)]);
    chart.setPoints(curve);
    chart.setHighlight(k);
    const p = curve[k - 1][1];
    out.textContent =
      a === b
        ? "Pick two different agents."
        : `Agent A (distance ${a}) averages ${Math.round(means.get(a) ?? 0).toLocaleString("en-US")} over ${POOL} seeds, agent B (distance ${b}) ${Math.round(means.get(b) ?? 0).toLocaleString("en-US")}. Judged on ${k} random seed${k === 1 ? "" : "s"}, the worse of the two comes out ahead ${Math.round(p * 100)}% of the time.`;
  }

  const makeSelect = (label: string, value: number, onChange: (v: number) => void) => {
    const sel = el("select", { "aria-label": label });
    for (const d of CANDIDATES) sel.add(new Option(`distance ${d}`, String(d)));
    sel.value = String(value);
    sel.addEventListener("change", () => onChange(Number(sel.value)));
    const lab = el("label", { class: "field" });
    lab.append(`${label} `, sel);
    return [lab, sel] as const;
  };
  const [labelA, selectA] = makeSelect("agent A", a, (v) => { a = v; update(); });
  const [labelB, selectB] = makeSelect("agent B", b, (v) => { b = v; update(); });
  controls.append(
    labelA,
    labelB,
    slider("seeds per fitness, K", 1, 10, 1, k, (v) => String(v), (v) => { k = v; update(); }),
  );

  // Play every candidate on the same seeds, in small steps so the page stays responsive.
  let next = 0;
  const step = (): void => {
    if (next >= CANDIDATES.length) {
      status.textContent = `Each agent is the rule "jump when an obstacle is within d px", played on ${POOL} seeds for up to ${SEED_CAP} ticks. The seeds are the same for every agent.`;
      // Start with the two agents whose averages are closest, where single seeds mislead most.
      // Agents that die at the first obstacle are left out: two of those are close but dull.
      const best = Math.max(...CANDIDATES.map((d) => means.get(d) ?? 0));
      const playable = CANDIDATES.filter((d) => (means.get(d) ?? 0) >= 0.3 * best);
      let closest = Infinity;
      for (const x of playable) {
        for (const y of playable) {
          const mx = means.get(x) ?? 0;
          const my = means.get(y) ?? 0;
          const gap = Math.abs(mx - my) / Math.max(mx, my, 1);
          if (x < y && gap < closest) {
            closest = gap;
            a = x;
            b = y;
          }
        }
      }
      selectA.value = String(a);
      selectB.value = String(b);
      update();
      return;
    }
    const d = CANDIDATES[next++];
    const list = Array.from({ length: POOL }, (_, i) => ruleScore(d, i, SEED_CAP));
    scores.set(d, list);
    means.set(d, list.reduce((s, x) => s + x, 0) / list.length);
    setTimeout(step, 0);
  };
  setTimeout(step, 0);

  return section(
    "Fitness is averaged over seeds",
    [
      "The obstacle sequence is random. A genome that meets an easy sequence scores well whatever its quality, so a score from one seed is a noisy measure.",
      `The GA scores each genome on ${DEFAULT_TRAINER.rounds} seeds and uses the mean. Here two similar agents are judged on K seeds drawn from a pool of ${POOL}. With K = 1 the worse agent often looks better. As K grows, the ranking settles, at the price of K times as many game ticks per genome.`,
    ],
    root,
  );
}

// --- 4. mutation ---------------------------------------------------------------------------------
const SIGMAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1];
const CHILDREN = 12;
const MUT_SEEDS = [0, 1, 2];
const MUT_CAP = 2500;

function mutationSection(): HTMLElement {
  const root = el("div", { class: "mutation-widget" });
  const note = el("p", { class: "readout" }, "Loading a trained network");
  const controls = el("div", { class: "widget-controls" });
  const graphs = el("div", { class: "two-graphs" });
  const parentHost = el("figure");
  const childHost = el("figure");
  const childInfo = el("figcaption", { class: "readout" });
  parentHost.append(el("figcaption", { class: "readout" }, "parent"));
  childHost.appendChild(childInfo);
  graphs.append(parentHost, childHost);
  const chartHost = el("div");
  root.append(note, controls, graphs, chartHost);

  const parentGraph = new NetworkGraph(parentHost);
  const childGraph = new NetworkGraph(childHost);
  const chart = new MiniChart(
    chartHost,
    {
      xLabel: "mutation size (sd)",
      yLabel: "child score / parent score",
      xTicks: [0.02, 0.05, 0.1, 0.2, 0.5, 1],
      yTicks: [0, 0.25, 0.5, 0.75, 1],
      xFormat: (v) => String(v),
      yFormat: (v) => v.toFixed(2),
      logX: true,
      reference: { y: 1, label: "same as parent" },
    },
    "Average child score relative to its parent for mutation sizes from 0.02 to 1",
  );

  let parent: Float64Array | null = null;
  let sigma = DEFAULT_GA.mutationSigma;
  let childSeed = 1;
  let ready = false;

  function drawChild(): void {
    if (parent === null) return;
    const { child, changed, largest } = mutate(parent, sigma, DEFAULT_GA.mutationRate, new Mulberry32(childSeed));
    childGraph.setGenome(child);
    childInfo.textContent = `child: ${changed} of ${GENOME_LENGTH} weights changed, largest change ${largest.toFixed(2)}`;
    if (ready) chart.setHighlight(sigma);
  }

  controls.append(
    slider("mutation size", 0.02, 1, 0.01, sigma, (v) => v.toFixed(2), (v) => { sigma = v; drawChild(); }),
  );
  const again = el("button", { type: "button" }, "New child");
  again.addEventListener("click", () => {
    childSeed += 1;
    drawChild();
  });
  controls.appendChild(again);

  async function load(): Promise<void> {
    const entry = bestRun("ga");
    const loader = entry === undefined ? undefined : genomeLoader(entry.genome_file);
    if (loader !== undefined) {
      const policy = loadPolicy(await loader());
      parent = policy.genome ?? null;
      note.textContent = `The parent is the best genome of one trained GA run (seed ${entry?.seed}). The line shows the average score of ${CHILDREN} children at each mutation size, on ${MUT_SEEDS.length} seeds, relative to the parent's score on the same seeds.`;
    }
    if (parent === null) {
      const rng = new Mulberry32(5);
      parent = Float64Array.from({ length: GENOME_LENGTH }, () => rng.random() * 2 - 1);
      note.textContent = "No trained genome is bundled with this build, so the parent is a random network.";
    }
    parentGraph.setGenome(parent);
    drawChild();
    const points: [number, number][] = [];
    for (const s of SIGMAS) {
      await new Promise((r) => setTimeout(r, 0));
      points.push([s, relativeChildScore(parent, s, DEFAULT_GA.mutationRate, MUT_SEEDS, CHILDREN, MUT_CAP)]);
      chart.setPoints([...points]);
    }
    ready = true;
    chart.setHighlight(sigma);
  }
  void load();

  return section(
    "Mutation size sets how far a child moves from its parent",
    [
      "Mutation adds Gaussian noise to some of a parent's weights. Small noise gives children that behave like the parent. Large noise gives children that behave like random networks.",
      "Drag the slider to see one child's weights against its parent's, and where that mutation size falls on the curve. Line width is weight size, and dashed lines are negative weights.",
    ],
    root,
  );
}

export function mountHow(root: HTMLElement): () => void {
  root.replaceChildren();
  const intro = el("section", { class: "result" });
  intro.appendChild(
    el(
      "p",
      { class: "lede" },
      "Small networks learn to play the game by evolution. Nothing in this page is a recording: each widget below runs the same game and network code as the live page.",
    ),
  );
  root.appendChild(intro);
  root.append(loopSection(), inputsSection(), seedsSection(), mutationSection());
  return () => undefined;
}
