/**
 * Results page. It shows the report's figures and tables, read straight from `report/`, which
 * the Python figure script writes from `results/`. Nothing on this page is typed in by hand;
 * if a figure or table is missing it says so.
 *
 * The figures are inlined SVG, so they pick up the page's colours and fonts and can be made
 * interactive without a second chart renderer (see render/figureInteractions.ts).
 */
import {
  Tooltip,
  enhanceFigure,
  seriesOf,
  setSeriesVisible,
} from "../render/figureInteractions";

interface Table {
  headers: string[];
  align: string[];
  rows: string[][];
}

const figureFiles = import.meta.glob("../../../report/figures/*.svg", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const tableFiles = import.meta.glob("../../../report/tables/tables.json", {
  import: "default",
  eager: true,
}) as Record<string, Record<string, Table>>;

function figureSvg(name: string): string | undefined {
  const key = Object.keys(figureFiles).find((k) => k.endsWith(`/${name}.svg`));
  return key === undefined ? undefined : figureFiles[key];
}

const tables: Record<string, Table> = Object.values(tableFiles)[0] ?? {};

interface Section {
  figure: string;
  title: string;
  caption: string;
  table?: string;
  tableTitle?: string;
  toggles?: boolean;
}

/** Headings name what is shown. Reading the numbers is left to the report. */
const SECTIONS: Section[] = [
  {
    figure: "learning_curves",
    title: "Learning curves",
    caption:
      "Fitness against environment steps, one line per algorithm, mean over seeds with a 95% band. " +
      "Move the pointer over a plot to read values, or focus it and use the arrow keys. " +
      "For the evolutionary methods the best curve is the best genome of each generation on its three " +
      "seeds; for PPO it is the best episode in a window of about 50,000 steps. Compare shapes, and " +
      "use the unseen-seed scores for levels.",
    toggles: true,
  },
  {
    figure: "final_scores",
    title: "Final scores on unseen seeds",
    caption:
      "Each run's best genome, played on 100 seeds that no algorithm trained or was tuned on, " +
      "in the full-length game.",
  },
  {
    figure: "sample_efficiency",
    title: "Environment steps to reach 1,000 and 3,000 points",
    caption:
      "Steps until a generation's best fitness first reached the score. Runs that never got there " +
      "are not drawn and are counted on the right.",
  },
  {
    figure: "generalisation",
    title: "Training score against test score",
    caption:
      "Both scores use the 12,000-tick training cap, so the axes are comparable. Marks below the " +
      "dotted line did worse on unseen seeds than on the three seeds that selected them. " +
      "PPO has no selection step: its training value is the best window mean under its stochastic " +
      "policy, and the test plays the deterministic policy, so its gap is not like for like.",
    table: "generalisation",
    tableTitle: "Generalisation, per algorithm",
  },
  {
    figure: "ablations",
    title: "GA ablations",
    caption:
      "One factor changed at a time against the default settings, ten runs per setting, same step budget.",
    table: "ablations",
    tableTitle: "Ablation results",
  },
  {
    figure: "decision_maps",
    title: "Decision maps",
    caption:
      "The action each algorithm's best genome takes when one obstacle is a given distance away at a " +
      "given game speed. The state is set by hand, so a map shows the policy alone.",
  },
  {
    figure: "neat_growth",
    title: "NEAT topology growth",
    caption:
      "Size of the best genome and number of species by generation, mean over runs with a 95% band.",
  },
];

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

function renderTable(t: Table, title: string): HTMLElement {
  const wrap = el("div", { class: "table-wrap" });
  const table = el("table", { class: "data" }, undefined);
  table.appendChild(el("caption", {}, title));
  const head = el("thead");
  const hr = el("tr");
  t.headers.forEach((h, i) => hr.appendChild(el("th", { scope: "col", class: t.align[i] === "r" ? "num" : "" }, h)));
  head.appendChild(hr);
  table.appendChild(head);
  const body = el("tbody");
  for (const row of t.rows) {
    const tr = el("tr");
    row.forEach((cell, i) => {
      tr.appendChild(el(i === 0 ? "th" : "td", i === 0 ? { scope: "row" } : { class: t.align[i] === "r" ? "num" : "" }, cell));
    });
    body.appendChild(tr);
  }
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}

function renderSection(s: Section, tooltip: Tooltip): HTMLElement | null {
  const svgText = figureSvg(s.figure);
  if (svgText === undefined) return null;
  const section = el("section", { class: "result" });
  section.appendChild(el("h2", { class: "result-title" }, s.title));

  const grid = el("div", { class: "grid result-grid" });
  const figure = el("div", { class: "col-8 figure" });
  figure.innerHTML = svgText;
  const svg = figure.querySelector<SVGSVGElement>("svg");
  if (svg !== null) {
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    enhanceFigure(svg, tooltip);
  }
  const side = el("div", { class: "col-4 figure-note" });
  side.appendChild(el("p", {}, s.caption));

  if (s.toggles && svg !== null) {
    const fieldset = el("fieldset", { class: "toggles" });
    fieldset.appendChild(el("legend", {}, "Show"));
    for (const { id, label } of seriesOf(svg)) {
      const lab = el("label");
      const box = el("input", { type: "checkbox", checked: "" });
      box.addEventListener("change", () => setSeriesVisible(svg, id, box.checked));
      lab.append(box, ` ${label}`);
      fieldset.appendChild(lab);
    }
    side.appendChild(fieldset);
  }
  grid.append(figure, side);
  section.appendChild(grid);

  const table = s.table === undefined ? undefined : tables[s.table];
  if (table !== undefined && s.tableTitle !== undefined) {
    section.appendChild(renderTable(table, s.tableTitle));
  }
  return section;
}

export function mountResults(root: HTMLElement): () => void {
  root.replaceChildren();
  const tooltip = new Tooltip(root);

  const intro = el("section", { class: "result" });
  intro.appendChild(el("h2", { class: "result-title" }, "Test scores on unseen seeds"));
  const headline = tables.headline;
  if (headline === undefined) {
    intro.appendChild(
      el(
        "p",
        { class: "empty" },
        "No results are bundled with this build. Run `make experiments` and `make figures`, then rebuild the page.",
      ),
    );
    root.appendChild(intro);
    return () => tooltip.hide();
  }
  intro.appendChild(
    el(
      "p",
      { class: "lede" },
      "Each algorithm's best genome from every run, played on 100 seeds that were never used for training or tuning. " +
        "Steps to 1,000 points are environment steps, one per game tick, until a generation's best fitness first reached that score.",
    ),
  );
  intro.appendChild(renderTable(headline, "Headline results"));
  root.appendChild(intro);

  for (const s of SECTIONS) {
    const node = renderSection(s, tooltip);
    if (node !== null) root.appendChild(node);
  }
  return () => tooltip.hide();
}
