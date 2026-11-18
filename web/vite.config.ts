import { defineConfig } from "vitest/config";

// The sim fixtures and baseline results live in ../sim and ../results, outside this
// folder, so the dev server needs permission to read from the repo root.
export default defineConfig({
  base: "./",
  server: { fs: { allow: [".."] } },
  test: { include: ["src/**/*.test.ts"] },
});
