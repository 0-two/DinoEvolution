/**
 * Readers for the genome files the Python side exports (see sim/genome_io.py).
 *
 * `dino-genome/1`  fixed 9-8-3 network, 96 weights (GA, CMA-ES)
 * `dino-neat/1`    ordered evaluation plan of a feed-forward graph (NEAT)
 * `dino-mlp/1`    dense layers with tanh between them (PPO)
 * `dino-rule/1`   the two numbers of the rule-based baseline
 *
 * Every reader turns a file into a `Policy`. The three network formats are checked against
 * Python in parity.test.ts: same file, same inputs, same action.
 */
import type { DinoGame } from "../sim/game";
import { ruleAction } from "../sim/rule";
import {
  OUTPUTS,
  forward,
  fromGenomeFile,
  makeActivations,
  type Activations,
  type GenomeFile,
} from "./network";

export interface NeatFile {
  format: "dino-neat/1";
  inputs: number[];
  outputs: number[];
  nodes: { id: number; bias: number; response: number; inputs: [number, number][] }[];
}

export interface MlpFile {
  format: "dino-mlp/1";
  layers: { weights: number[][]; bias: number[] }[];
}

export interface RuleFile {
  format: "dino-rule/1";
  d: number;
  gain: number;
}

export type PolicyFile = GenomeFile | NeatFile | MlpFile | RuleFile;

export interface Policy {
  kind: PolicyFile["format"];
  /** One line per fact worth showing next to the game. */
  summary: string[];
  /** Choose an action for the current game state. */
  act(game: DinoGame): number;
  /** The same decision from an observation vector. Absent for the rule, which reads the game. */
  actFromObservation?(obs: ArrayLike<number>): number;
  /** Present for the fixed network, so its activations can be drawn. */
  activations?: Activations;
  genome?: Float64Array;
  lastAction: number;
}

function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function fixedPolicy(file: GenomeFile): Policy {
  const genome = fromGenomeFile(file);
  const activations = makeActivations();
  const policy: Policy = {
    kind: file.format,
    summary: ["fixed network, 9 inputs, 8 tanh hidden units, 3 outputs", "96 weights"],
    genome,
    activations,
    lastAction: 0,
    actFromObservation(obs) {
      activations.inputs.set(Array.from(obs));
      return forward(genome, activations);
    },
    act(game) {
      game.observeInto(activations.inputs);
      policy.lastAction = forward(genome, activations);
      return policy.lastAction;
    },
  };
  return policy;
}

function neatPolicy(file: NeatFile): Policy {
  const values = new Map<number, number>();
  const run = (obs: ArrayLike<number>): number => {
    values.clear();
    // Outputs start at 0, as in neat-python, in case a plan never evaluates one.
    for (const o of file.outputs) values.set(o, 0);
    file.inputs.forEach((key, i) => values.set(key, obs[i]));
    for (const node of file.nodes) {
      let total = 0;
      for (const [src, w] of node.inputs) total += (values.get(src) ?? 0) * w;
      values.set(node.id, Math.tanh(node.bias + node.response * total));
    }
    return argmax(file.outputs.map((o) => values.get(o) ?? 0));
  };
  const hidden = file.nodes.filter((n) => !file.outputs.includes(n.id)).length;
  const links = file.nodes.reduce((sum, n) => sum + n.inputs.length, 0);
  const policy: Policy = {
    kind: file.format,
    summary: [`evolved graph, ${hidden} hidden nodes, ${links} connections`, "tanh at every node"],
    lastAction: 0,
    actFromObservation: run,
    act(game) {
      const obs = new Float64Array(file.inputs.length);
      game.observeInto(obs);
      policy.lastAction = run(obs);
      return policy.lastAction;
    },
  };
  return policy;
}

function mlpPolicy(file: MlpFile): Policy {
  const run = (obs: ArrayLike<number>): number => {
    let x: number[] = Array.from(obs);
    file.layers.forEach((layer, li) => {
      const next = layer.weights.map((row, o) => {
        let sum = layer.bias[o];
        for (let i = 0; i < row.length; i++) sum += row[i] * x[i];
        return sum;
      });
      x = li < file.layers.length - 1 ? next.map(Math.tanh) : next;
    });
    return argmax(x);
  };
  const widths = [file.layers[0].weights[0].length, ...file.layers.map((l) => l.bias.length)];
  const policy: Policy = {
    kind: file.format,
    summary: [`dense network ${widths.join("-")}`, "tanh between layers, trained with PPO"],
    lastAction: 0,
    actFromObservation: run,
    act(game) {
      const obs = new Float64Array(widths[0]);
      game.observeInto(obs);
      policy.lastAction = run(obs);
      return policy.lastAction;
    },
  };
  return policy;
}

function rulePolicy(file: RuleFile): Policy {
  const policy: Policy = {
    kind: file.format,
    summary: [
      `hand-written rule: jump or duck within ${file.d} px`,
      file.gain === 0 ? "same distance at every speed" : `plus ${file.gain} px per unit of speed`,
    ],
    lastAction: 0,
    act(game) {
      policy.lastAction = ruleAction(game, { d: file.d, gain: file.gain });
      return policy.lastAction;
    },
  };
  return policy;
}

export function loadPolicy(file: PolicyFile): Policy {
  switch (file.format) {
    case "dino-genome/1":
      return fixedPolicy(file);
    case "dino-neat/1":
      return neatPolicy(file);
    case "dino-mlp/1":
      return mlpPolicy(file);
    case "dino-rule/1":
      return rulePolicy(file);
    default:
      throw new Error(`unknown genome format ${(file as { format?: string }).format ?? "?"}`);
  }
}

export { OUTPUTS };
