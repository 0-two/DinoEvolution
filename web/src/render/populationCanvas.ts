/**
 * The population view. Every living dino is a thin outline at low opacity, so a crowd shows
 * up as a darker band where many of them overlap. The followed dino is drawn solid.
 *
 * All living dinos share one obstacle sequence and one speed, so the obstacles are drawn once,
 * from the followed dino's game.
 */
import { WORLD_WIDTH } from "../sim/game";
import type { Trainer } from "../evolve/trainer";
import {
  VIEW_HEIGHT,
  drawDino,
  drawGround,
  drawObstacle,
  invalidatePalette,
  palette,
  strokeDino,
} from "./draw";

const CROWD_ALPHA = 0.22;

export class PopulationCanvas {
  private readonly ctx: CanvasRenderingContext2D;
  private scale = 1;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (ctx === null) throw new Error("2D canvas is not available");
    this.ctx = ctx;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      invalidatePalette();
    });
  }

  /** Match the backing store to the CSS size so lines stay sharp on any display. */
  resize(): void {
    const cssWidth = this.canvas.clientWidth || WORLD_WIDTH;
    const dpr = window.devicePixelRatio || 1;
    this.scale = cssWidth / WORLD_WIDTH;
    this.canvas.style.height = `${VIEW_HEIGHT * this.scale}px`;
    this.canvas.width = Math.round(cssWidth * dpr);
    this.canvas.height = Math.round(VIEW_HEIGHT * this.scale * dpr);
    this.ctx.setTransform(this.scale * dpr, 0, 0, this.scale * dpr, 0, 0);
  }

  draw(trainer: Trainer): void {
    const { ctx } = this;
    const p = palette();
    const pixel = 1 / this.scale;
    ctx.fillStyle = p.paper;
    ctx.fillRect(0, 0, WORLD_WIDTH, VIEW_HEIGHT);
    drawGround(ctx, WORLD_WIDTH, pixel);

    const lead = trainer.leaderGame;
    for (const o of lead.obstacles) drawObstacle(ctx, o);

    ctx.lineWidth = pixel;
    ctx.strokeStyle = p.ink;
    ctx.globalAlpha = CROWD_ALPHA;
    const agents = trainer.agents;
    for (let i = 0; i < agents.length; i++) {
      if (i !== trainer.leader && !agents[i].game.done) strokeDino(ctx, agents[i].game);
    }
    ctx.globalAlpha = 1;
    drawDino(ctx, lead, p.ink);
  }
}
