import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/", "node_modules/"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    // Plain Node scripts (benchmarks) use these runtime globals.
    files: ["scripts/**/*.mjs"],
    languageOptions: {
      globals: {
        process: "readonly",
        console: "readonly",
        fetch: "readonly",
        WebSocket: "readonly",
        setTimeout: "readonly",
        Buffer: "readonly",
      },
    },
  },
);
