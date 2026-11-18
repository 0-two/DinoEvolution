/**
 * Port of sim/algorithms/rule_based.py: jump or duck when the next obstacle is closer than
 * a trigger distance. In the browser it drives the minimal demo until evolved agents exist.
 */
import {
  DINO_H,
  DINO_W,
  DINO_X,
  DUCK,
  DUCK_H,
  JUMP,
  NOOP,
  START_SPEED,
  type DinoGame,
} from "./game";

export interface RuleParams {
  d: number;
  /** Extra trigger distance per unit of speed above START_SPEED. */
  gain: number;
}

export function ruleAction(game: DinoGame, params: RuleParams): number {
  if (game.airborne) return NOOP;
  const [first] = game.upcoming();
  if (first === null) return NOOP;
  const trigger = params.d + params.gain * (game.speed - START_SPEED);
  if (first[0] - (DINO_X + DINO_W) >= trigger) return NOOP;
  const bottom = first[1];
  if (bottom >= DINO_H) return NOOP;
  return bottom >= DUCK_H ? DUCK : JUMP;
}
