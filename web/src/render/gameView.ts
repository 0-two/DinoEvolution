/** One game on a canvas: the road, the obstacles and a solid dino. Used by the Watch page. */
import { WORLD_WIDTH, type DinoGame } from "../sim/game";
import {
  VIEW_HEIGHT,
  drawDino,
  drawGround,
  drawObstacle,
  invalidatePalette,
  palette,
} from "./draw";

export class GameView {
  private readonly ctx: CanvasRenderingContext2D;
  private scale = 1;
  private last: DinoGame | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (ctx === null) throw new Error("2D canvas is not available");
    this.ctx = ctx;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", invalidatePalette);
  }

  resize(): void {
    const cssWidth = this.canvas.clientWidth || WORLD_WIDTH;
    const dpr = window.devicePixelRatio || 1;
    this.scale = cssWidth / WORLD_WIDTH;
    this.canvas.style.height = `${VIEW_HEIGHT * this.scale}px`;
    this.canvas.width = Math.round(cssWidth * dpr);
    this.canvas.height = Math.round(VIEW_HEIGHT * this.scale * dpr);
    this.ctx.setTransform(this.scale * dpr, 0, 0, this.scale * dpr, 0, 0);
    // Setting the canvas size clears it, so draw again if a game was already shown.
    if (this.last !== null) this.draw(this.last);
  }

  draw(game: DinoGame): void {
    this.last = game;
    const p = palette();
    this.ctx.fillStyle = p.paper;
    this.ctx.fillRect(0, 0, WORLD_WIDTH, VIEW_HEIGHT);
    drawGround(this.ctx, WORLD_WIDTH, 1 / this.scale);
    for (const o of game.obstacles) drawObstacle(this.ctx, o);
    drawDino(this.ctx, game, game.dead ? p.accent : p.ink);
  }
}
