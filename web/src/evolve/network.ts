/**
 * The policy network: 9 inputs -> 8 hidden (tanh) -> 3 outputs, no separate bias terms.
 * The ninth input is a constant 1, so its column of first-layer weights is the hidden bias.
 * The outputs have no bias, as in the project brief.
 *
 * A genome is a flat vector of 96 numbers: the first layer row by row (hidden unit 0's nine
 * weights, then unit 1's, ...) followed by the second layer row by row. The Python
 * implementation will use the same layout, so a genome trained there loads here unchanged.
 *
 * The action is the index of the largest output. Ties go to the lower index, which is
 * "nothing", so a network that has not learned anything stands still.
 */
import { OBSERVATION_SIZE } from "../sim/game";

export const INPUTS = OBSERVATION_SIZE;
export const HIDDEN = 8;
export const OUTPUTS = 3;
export const GENOME_LENGTH = HIDDEN * INPUTS + OUTPUTS * HIDDEN;

export const INPUT_LABELS = [
  "distance",
  "width",
  "height",
  "obstacle y",
  "next distance",
  "speed",
  "dino y",
  "dino vy",
  "bias",
] as const;
export const OUTPUT_LABELS = ["nothing", "jump", "duck"] as const;

/** Scratch buffers for one forward pass, kept so the best dino's activations can be drawn. */
export interface Activations {
  inputs: Float64Array;
  hidden: Float64Array;
  outputs: Float64Array;
}

export function makeActivations(): Activations {
  return {
    inputs: new Float64Array(INPUTS),
    hidden: new Float64Array(HIDDEN),
    outputs: new Float64Array(OUTPUTS),
  };
}

/** Run the network on `act.inputs`, fill `act.hidden` and `act.outputs`, return the action. */
export function forward(genome: Float64Array, act: Activations): number {
  const { inputs, hidden, outputs } = act;
  let w = 0;
  for (let h = 0; h < HIDDEN; h++) {
    let sum = 0;
    for (let i = 0; i < INPUTS; i++) sum += genome[w++] * inputs[i];
    hidden[h] = Math.tanh(sum);
  }
  let best = 0;
  for (let o = 0; o < OUTPUTS; o++) {
    let sum = 0;
    for (let h = 0; h < HIDDEN; h++) sum += genome[w++] * hidden[h];
    outputs[o] = sum;
    if (sum > outputs[best]) best = o;
  }
  return best;
}

/** First-layer weight from input i to hidden unit h. */
export function w1(genome: Float64Array, h: number, i: number): number {
  return genome[h * INPUTS + i];
}

/** Second-layer weight from hidden unit h to output o. */
export function w2(genome: Float64Array, o: number, h: number): number {
  return genome[HIDDEN * INPUTS + o * HIDDEN + h];
}

/** JSON shape shared with the Python side. */
export interface GenomeFile {
  format: "dino-genome/1";
  topology: { inputs: number; hidden: number; outputs: number; activation: "tanh" };
  weights: number[];
}

export function toGenomeFile(genome: Float64Array): GenomeFile {
  return {
    format: "dino-genome/1",
    topology: { inputs: INPUTS, hidden: HIDDEN, outputs: OUTPUTS, activation: "tanh" },
    weights: Array.from(genome),
  };
}

export function fromGenomeFile(file: GenomeFile): Float64Array {
  const t = file.topology;
  if (file.format !== "dino-genome/1" || t.inputs !== INPUTS || t.hidden !== HIDDEN || t.outputs !== OUTPUTS) {
    throw new Error("genome does not match the 9-8-3 topology");
  }
  if (file.weights.length !== GENOME_LENGTH) {
    throw new Error(`expected ${GENOME_LENGTH} weights, got ${file.weights.length}`);
  }
  return Float64Array.from(file.weights);
}
