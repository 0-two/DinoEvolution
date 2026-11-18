# Targets for the Python side, then the web app.
.PHONY: setup lint test test-docs parity bench baseline quick experiments figures docs dead-ends gif web-setup web-check web-parity web-dev web-perf

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

# Code tests, the same set CI runs. The documentation checks are separate: `make test-docs`.
test:
	uv run pytest -q --ignore=sim/tests/test_docs.py

# Style, link and placeholder checks on the README and report (not run by CI).
test-docs:
	uv run pytest -q sim/tests/test_docs.py

# Replays the stored golden traces against the Python game. The web test replays the same file.
parity:
	uv run python -m sim.parity check

bench:
	uv run python -m sim.bench

baseline:
	uv run python -m sim.algorithms.rule_based

# Every algorithm and three GA ablations at small budgets, 3 seeds. Under ten minutes on a
# laptop. Writes results/quick/. Interrupted runs resume; delete results/quick to start over.
quick: setup
	uv run python -m sim.experiments.run --suite quick

# The full comparison (10 seeds per algorithm) and the GA ablations. Takes hours on one machine.
experiments: setup
	uv run python -m sim.experiments.run --suite main
	uv run python -m sim.experiments.run --suite ablations

# Figures (SVG) and tables for the report and the Results page, all generated from results/.
# The first step replays every final genome on the test seeds at the training cap (a few
# minutes); the second writes report/figures and report/tables.
figures: setup
	uv run python -m sim.analysis.generalisation --suite main
	uv run python -m sim.figures.build

# README.md and report/REPORT.md from their templates, with every number read from results/.
docs: setup
	uv run python -m sim.figures.docs

# Assemble report/media/live-evolution.gif from frames captured by web/scripts/capture-frames.mjs
# (start `npx vite preview --port 4173` in web/ first, then run the capture script).
gif:
	uv run python -m sim.figures.gif

# Re-measure the settings that did not work (results/notes/dead_ends.json). A few minutes.
dead-ends: setup
	uv run python -m sim.experiments.dead_ends

web-setup:
	cd web && npm ci

# Lint, typecheck, run all web tests (including parity) and build.
web-check:
	cd web && npm run lint && npm run typecheck && npm test && npm run build

# The parity test on its own: Python golden traces replayed through the TypeScript port.
web-parity:
	cd web && npx vitest run src/sim/parity.test.ts

web-dev:
	cd web && npm run dev

# Frame pacing at 150 and 300 dinos. Needs Chrome or Edge. Writes results/benchmarks/web_perf.json.
web-perf:
	cd web && npm run build && (npx vite preview --port 4173 --strictPort & echo $$! > .preview.pid; sleep 3; 	  node scripts/perf.mjs --url http://localhost:4173/; kill `cat .preview.pid`; rm -f .preview.pid)
