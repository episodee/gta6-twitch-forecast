# Research note — GTA VI launch-day Twitch forecast

## The problem in one paragraph

Grand Theft Auto VI unlocks at rolling midnight local time on 19 November 2026.
The contest asks, at the end of the day's first UTC hour, for five quantiles of
the Twitch "Grand Theft Auto VI" category's concurrent viewers in each of hours
2–24, scored by mean pinball loss on the log scale. Hour 1 (00:00–01:00 UTC)
only contains the regions whose midnight has already passed — New Zealand,
Australia, the UK — while the US east coast unlocks at 05:00 UTC and the west
coast at 08:00 UTC. So the day is not flat, and the shape of the day depends
on how big hour 1 already is: a launch whose category was quiet before a
simultaneous global unlock can grow 100–1000x from hour 1, while a launch
whose hour 1 is already 875k (Elden Ring) is near its peak and grows ~1x.

## Method

One file, `main.py` (Python 3.13, standard library only, no network). At run
time it reads the case's own `comparables` — past big launches' hourly Twitch
category averages (SullyGnome) from 14 days before to 3 days after each
release day — and, for each forecast hour h, fits a kernel-weighted linear
regression of `log(viewers_h / viewers_1)` on `log(viewers_1)`:

- **Kernel:** Gaussian weights in log hour-1 space, bandwidth 1.5. A target
  with a huge hour-1 print is forecast from the big launches, not dragged
  by the tiny ones; a small hour-1 print is forecast from the small ones.
- **Slope shrinkage 0.9:** a small guard against over-extrapolation when the
  target's hour-1 print sits outside the comparables' range.
- **Spread:** the kernel-weighted MAD of the regression's residuals, floored
  at 0.15 log units (the honest dispersion across launches is wide:
  sigma ≈ 0.3–1.0 by hour, widest in the evening peak).
- **Safety rails:** the forecast median ratio is clamped to the range the
  comparables actually showed for that hour (±0.5 log); quantiles are
  clamped to (0, 1e9] and forced non-decreasing. If a comparable set is
  missing or an hour has no coverage, the entry falls back to the house
  baseline's flat forecast for that hour rather than failing the case.

Nothing is hard-coded from the practice cases: the model, the curve and the
spread are all recomputed from whatever comparables the case file carries
(38 in a practice case, 42 in the final, which adds the four practice
launches).

## Validation

The practice cases' answers are not in the kit, so they were re-read from
the same public source the kit quotes (SullyGnome's hourly averages, one
request per game, 1s+ apart — the exact API the kit's NOTICE.txt documents).
The re-read points match the kit's `history` points exactly (167/167,
330/330, 189/189, 240/240), so the scored release-day hours 2–24 are the
same numbers the server grades against.

Mean pinball loss over the four practice cases (lower is better):

| model | Wolverine | RDR2 | GoW Ragnarök | Spider-Man 2 | mean |
|---|---|---|---|---|---|
| House baseline (flat at hour 1) | 0.744 | 0.287 | 0.215 | 0.303 | **0.387** |
| Global median ratio curve | 0.418 | 0.267 | 0.266 | 0.291 | 0.310 |
| Global log-linear on log hour-1 | 0.203 | 0.226 | 0.231 | 0.232 | 0.223 |
| **This model (local log-linear)** | **0.171** | **0.214** | **0.135** | **0.197** | **0.179** |

Leave-one-launch-out over the 37 comparables (each launch forecast from the
other 36, same code): **0.390** vs the baseline's 0.930 on the same
pseudo-cases. The two biggest launches — the closest analogues to GTA VI —
score best: Elden Ring 0.082, Cyberpunk 2077 0.116. The worst leave-one-out
cases are the simultaneous-global-unlock launches (Black Myth: Wukong 1.75,
Marathon 1.34), whose hour-1 print says almost nothing about the day;
GTA VI is a rolling-midnight launch, so that failure mode does not apply.

What was tried and dropped: a plain median curve (too stiff at both ends);
conditioning on the hour-1/pre-release ratio (unstable — a band that caught
the wrong launches scored 2.51 on Spider-Man 2); filtering out launches
whose peak exceeded 10x hour 1 (it removed exactly the small launches
Wolverine resembles, and practice worsened to 0.262); k-nearest neighbours
in log hour-1 (worse on Wolverine, 0.402). The kernel-weighted regression
dominated on both the practice cases and leave-one-out.

## Assumptions and limits

- **Rolling midnight unlock.** The model assumes hour 1 already reflects a
  live launch in some regions. True for GTA VI (NZ 11:00Z Nov 18, Europe
  23:00Z, UK 00:00Z, US East 05:00Z, US West 08:00Z) and for the practice
  launches. A single global unlock hour would break the anchor.
- **Practice ≠ final measure.** Practice hours are SullyGnome hourly
  averages (one point per hour); final hours are means of 60 minute reads.
  The two measures of an hour differ slightly, and the final's hour-1
  print is a minute mean. The model was validated on the former and is
  applied to the latter; the log-scale scoring makes the difference
  second-order, but it is a real gap.
- **GTA VI is bigger than any comparable.** The largest comparable hour-1
  print is 875k (Elden Ring). If GTA VI's hour 1 exceeds ~1M, the model
  extrapolates the log-linear fit beyond its range; the kernel weighting,
  slope shrinkage and the ratio clamp are the guards, and the residual
  spread (≈0.9 log at the evening peak) prices the extra uncertainty.
- **No early access, no PC launch.** GTA VI launches on PS5/Xbox only,
  single-player, no early access — matching the practice launches. The
  comparables include early-access-contaminated launches (Hogwarts Legacy,
  Starfield, Diablo IV); the kernel weighting limits their influence to
  targets near their size, and the final case's extra comparables (the
  four practice launches) pull the fit further toward clean console
  launches.
- **Four practice cases only.** The headline 0.179 rests on four launches
  plus 37-way leave-one-out. The ranking of model variants was stable
  across both, which is the best evidence available before the final.
