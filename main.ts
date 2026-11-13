// polymarket-nvidia-ceo-forecast-01 entry (Node 22, standard library only).
//
// Model: persistence median with a per-market, regime-adaptive Laplace spread.
//
// These Polymarket prices are step functions: flat for long stretches, small
// ticks, and occasional jumps on news (the practice window holds the
// 2026-11-12 resignation of Jensen Huang and the Colette Kress spike to 0.59
// that reverted the next minute). The house baseline gives every market the
// same normal spread (sigma 0.0025); measured on the practice cases that is
// far too wide for quiet markets and too narrow for active ones, and its 90%
// coverage comes out at 0.95 instead of 0.90. This entry instead:
//
//   - estimates each market's own per-minute scale from the case's history:
//     the mean absolute 1-minute move, rescaled to a standard deviation
//     (x sqrt(pi/2)), averaged over the last 120, 240 and 1440 minutes, so
//     the estimate follows the market's current regime without chasing a
//     single hour. Moves are clipped at +/-0.05 before measuring: the feed
//     contains synchronised one-minute blips to ~0.485 across many markets
//     at once (e.g. around minute 27 after the 2026-11-11T23:26 listings),
//     which are book artefacts that revert instantly, not volatility.
//     The first 120 minutes of a market's history are dropped for the same
//     reason (listing transients). The scale is floored at 0.0002 (a dormant
//     market can still jump) and capped at 0.03.
//   - uses Laplace quantiles instead of normal ones: the move distribution
//     is sharply peaked at zero with heavy tails, so the inner quantiles sit
//     closer in and the outer ones further out than a normal's, at the same
//     scale. The scale multiplier 1.35 and the sqrt(h) widening were chosen
//     on the six practice cases plus five extra checkpoints cut from the
//     same history, where this program's shape scores a mean pinball loss
//     of about 0.311 against the baseline's 0.403 on the same minutes.
//   - keeps the median at the checkpoint price: drift, momentum and
//     mean-reversion terms were all tested on the practice and each made
//     the pinball loss worse, so the honest median is "stays where it is".
//
// The program is deterministic: no clock, no randomness, no network. It
// reads every <id>.in in the input directory and writes <id>.out beside
// the required shape: {"forecasts":[{"id","q"}]} with horizon_minutes rows
// of the case's five quantiles, non-decreasing, each in [0,1].

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Minutes of a market's early history ignored when measuring its scale. */
const BURN_IN = 120;
/** 1-minute moves are clipped to +/- this before their size is measured. */
const MOVE_CAP = 0.05;
/** Windows (minutes of 1-minute moves) whose scale estimates are averaged. */
const SCALE_WINDOWS = [120, 240, 1440];
/** A dormant market still gets this much scale; a wild one at most this. */
const SIGMA_FLOOR = 0.0002;
const SIGMA_CAP = 0.03;
/**
 * Scale for a market with no usable moves at all (its whole history is
 * inside the burn-in: a just-listed market, like Johny Srouji's 20 prices
 * in practice case p03). Falling back to the house baseline's 0.0025 beat
 * both the floor (far too narrow for a listing transient) and measuring
 * the transient itself on the practice cases.
 */
const SIGMA_DEFAULT = 0.0025;
/** Overall spread multiplier, tuned on the practice cases. */
const SPREAD_K = 1.35;
/** Mean absolute value of a N(0, s) variable is s * sqrt(2/pi); invert it. */
const MEAN_ABS_TO_SD = Math.sqrt(Math.PI / 2);

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** The Laplace quantile at level tau, in units of the distribution's SD. */
function laplaceZ(tau: number): number {
  if (tau < 0.5) return Math.log(2 * tau) / Math.SQRT2;
  if (tau > 0.5) return -Math.log(2 * (1 - tau)) / Math.SQRT2;
  return 0;
}

/**
 * A market's per-minute scale: the rescaled mean absolute 1-minute move,
 * averaged over SCALE_WINDOWS windows of its (burned-in, clipped) moves.
 */
function marketSigma(prices: number[]): number {
  const moves: number[] = [];
  for (let i = BURN_IN + 1; i < prices.length; i++) {
    const d = (prices[i] as number) - (prices[i - 1] as number);
    moves.push(Math.min(MOVE_CAP, Math.max(-MOVE_CAP, d)));
  }
  if (moves.length === 0) return SIGMA_DEFAULT;
  let total = 0;
  for (const w of SCALE_WINDOWS) {
    const tail = moves.slice(Math.max(0, moves.length - w));
    let sum = 0;
    for (const d of tail) sum += Math.abs(d);
    total += (sum / tail.length) * MEAN_ABS_TO_SD;
  }
  const sigma = total / SCALE_WINDOWS.length;
  return Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sigma));
}

/** One market's forecast: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number, zs: number[]): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;
  const spread = SPREAD_K * marketSigma(m.prices);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = spread * Math.sqrt(h);
    rows.push(zs.map((z) => price6(x + z * s)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const zs = c.quantiles.map(laplaceZ);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes, zs) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
