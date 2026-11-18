/**
 * Shared drawing code: the palette and the shapes. Everything is a rectangle, a line or a
 * triangle. There are no sprites and no images.
 *
 * Colours are read from the page's CSS variables so the light and dark themes both apply.
 */
import {
  BIRD_H,
  DINO_H,
  DINO_W,
  DINO_X,
  DUCK_H,
  DUCK_W,
  type DinoGame,
  type Obstacle,
} from "../sim/game";

export interface Palette {
  paper: string;
  ink: string;
  ink2: string;
  rule: string;
  accent: string;
}

let cached: Palette | null = null;

export function palette(): Palette {
  if (cached === null) {
    const css = getComputedStyle(document.documentElement);
    const get = (name: string) => css.getPropertyValue(name).trim();
    cached = {
      paper: get("--paper"),
      ink: get("--ink"),
      ink2: get("--ink-2"),
      rule: get("--rule"),
      accent: get("--accent"),
    };
  }
  return cached;
}

/** Call after the theme changes so the next draw reads fresh colours. */
export function invalidatePalette(): void {
  cached = null;
}

export const VIEW_HEIGHT = 190;
export const GROUND_Y = 160;

export function drawGround(ctx: CanvasRenderingContext2D, width: number, pixel: number): void {
  ctx.strokeStyle = palette().rule;
  ctx.lineWidth = pixel;
  ctx.beginPath();
  ctx.moveTo(0, GROUND_Y + 0.5);
  ctx.lineTo(width, GROUND_Y + 0.5);
  ctx.stroke();
}

/** Solid dino. Standing is a body with a head; ducking is a flat block. */
export function drawDino(ctx: CanvasRenderingContext2D, game: DinoGame, color: string): void {
  const ducking = game.ducking;
  const w = ducking ? DUCK_W : DINO_W;
  const h = ducking ? DUCK_H : DINO_H;
  const top = GROUND_Y - game.y - h;
  ctx.fillStyle = color;
  if (ducking) {
    ctx.fillRect(DINO_X, top, w, h);
    ctx.fillStyle = palette().paper;
    ctx.fillRect(DINO_X + w - 12, top + 5, 4, 4);
  } else {
    ctx.fillRect(DINO_X, top + 14, w - 10, h - 14);
    ctx.fillRect(DINO_X + w - 24, top, 24, 22);
    ctx.fillStyle = palette().paper;
    ctx.fillRect(DINO_X + w - 9, top + 5, 4, 4);
  }
}

/** Thin outline of a dino's bounding box, for the crowd. */
export function strokeDino(ctx: CanvasRenderingContext2D, game: DinoGame): void {
  const ducking = game.ducking;
  const h = ducking ? DUCK_H : DINO_H;
  ctx.strokeRect(DINO_X, GROUND_Y - game.y - h, ducking ? DUCK_W : DINO_W, h);
}

export function drawObstacle(ctx: CanvasRenderingContext2D, o: Obstacle): void {
  const [x, y, w, h] = o;
  const top = GROUND_Y - y - h;
  ctx.fillStyle = palette().ink2;
  if (h === BIRD_H) {
    // Bird: a body block, a wing triangle and a beak. It does not flap.
    ctx.fillRect(x + 6, top + 10, w - 12, 12);
    ctx.beginPath();
    ctx.moveTo(x + 14, top + 10);
    ctx.lineTo(x + 28, top);
    ctx.lineTo(x + 34, top + 10);
    ctx.closePath();
    ctx.fill();
    ctx.fillRect(x, top + 14, 8, 4);
    return;
  }
  // Cactus group: one trunk with two short arms per unit.
  const unit = h === 35 ? 17 : 25;
  const count = Math.round(w / unit);
  for (let i = 0; i < count; i++) {
    const ux = x + i * unit;
    const trunk = Math.max(6, unit * 0.4);
    const tx = ux + (unit - trunk) / 2;
    ctx.fillRect(tx, top, trunk, h);
    ctx.fillRect(ux, top + h * 0.3, tx - ux + 1, 4);
    ctx.fillRect(ux, top + h * 0.3 - 8, 4, 12);
    ctx.fillRect(tx + trunk - 1, top + h * 0.5, ux + unit - tx - trunk + 1, 4);
    ctx.fillRect(ux + unit - 4, top + h * 0.5 - 8, 4, 12);
  }
}
