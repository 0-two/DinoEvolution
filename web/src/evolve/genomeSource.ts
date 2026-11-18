/**
 * Where the trained genomes come from. The Python experiments export one JSON file per run to
 * `results/`, and the figure script writes an index of them to `report/tables/genomes.json`.
 * Vite bundles the index, and each genome file is loaded only when asked for.
 */
import type { PolicyFile } from "./policies";

export interface IndexEntry {
  run_id: string;
  algorithm: string;
  seed: number;
  test_mean: number;
  /** Path relative to the repository root, e.g. `results/main/genomes/ga/seed_3.json`. */
  genome_file: string;
}

const indexFiles = import.meta.glob("../../../report/tables/genomes.json", {
  import: "default",
  eager: true,
}) as Record<string, IndexEntry[]>;

export const genomeIndex: IndexEntry[] = Object.values(indexFiles)[0] ?? [];

const loaders = import.meta.glob("../../../results/{main,quick}/genomes/*/seed_*.json", {
  import: "default",
}) as Record<string, () => Promise<PolicyFile>>;

export function genomeLoader(genomeFile: string): (() => Promise<PolicyFile>) | undefined {
  const key = Object.keys(loaders).find((k) => k.endsWith(`/${genomeFile}`));
  return key === undefined ? undefined : loaders[key];
}

/** The run of one algorithm with the highest test score, or undefined if there is none. */
export function bestRun(runId: string): IndexEntry | undefined {
  return genomeIndex.filter((e) => e.run_id === runId).sort((a, b) => b.test_mean - a.test_mean)[0];
}
