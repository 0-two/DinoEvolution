/**
 * Hash routing, no framework. A page is a function that fills a container and returns a
 * function that stops whatever it started (animation frames, listeners).
 *
 * Only pages that exist are registered, and the navigation is built from the registry, so a
 * link never points at a page that is not there.
 */
export interface Page {
  id: string;
  label: string;
  mount(container: HTMLElement): () => void;
}

export function startRouter(pages: Page[], container: HTMLElement, nav: HTMLElement): void {
  let unmount: (() => void) | null = null;

  const current = (): Page => {
    const id = window.location.hash.replace(/^#\/?/, "");
    return pages.find((p) => p.id === id) ?? pages[0];
  };

  const show = (): void => {
    unmount?.();
    container.replaceChildren();
    const page = current();
    for (const link of nav.querySelectorAll("a")) {
      if (link.dataset.page === page.id) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    document.title = `${page.label} – Dino neuroevolution`;
    unmount = page.mount(container);
  };

  for (const page of pages) {
    const a = document.createElement("a");
    a.href = `#/${page.id}`;
    a.textContent = page.label;
    a.dataset.page = page.id;
    nav.appendChild(a);
  }
  window.addEventListener("hashchange", show);
  show();
}
