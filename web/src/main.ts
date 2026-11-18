import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "@fontsource/instrument-serif/latin-400.css";
import "./style.css";
import { mountLive } from "./pages/live";
import { startRouter, type Page } from "./router";

/** Load a page's code when it is first opened. The results page carries every figure, so it
 *  stays out of the bundle the live page needs. If the user leaves before the code arrives,
 *  the page is never mounted. */
function lazy(load: () => Promise<{ mount(c: HTMLElement): () => void }>): Page["mount"] {
  return (container) => {
    let stop: (() => void) | null = null;
    let cancelled = false;
    void load().then((page) => {
      if (!cancelled) stop = page.mount(container);
    });
    return () => {
      cancelled = true;
      stop?.();
    };
  };
}

const pages: Page[] = [
  { id: "live", label: "Live evolution", mount: mountLive },
  {
    id: "watch",
    label: "Watch an agent",
    mount: lazy(async () => ({ mount: (await import("./pages/watch")).mountWatch })),
  },
  {
    id: "how",
    label: "How it works",
    mount: lazy(async () => ({ mount: (await import("./pages/how")).mountHow })),
  },
  {
    id: "results",
    label: "Results",
    mount: lazy(async () => ({ mount: (await import("./pages/results")).mountResults })),
  },
];

const container = document.getElementById("page");
const nav = document.getElementById("nav");
if (container === null || nav === null) throw new Error("page shell is missing");
startRouter(pages, container, nav);
