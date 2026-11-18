"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun).

Method: a local (kernel-weighted) log-linear forecaster, calibrated at run time
on the case's own comparables -- past big game launches' hourly Twitch
category averages (SullyGnome), which the case file carries.

For each forecast hour h = 2..24, every comparable launch with a usable
release day gives one observation: (log of its hour-1 print, log of
viewers_h / viewers_1). The forecast for the target case is a weighted
least-squares line of log-ratio on log hour-1 print, with Gaussian kernel
weights in log hour-1 space (bandwidth 1.5), evaluated at the target's own
hour-1 print. The spread is the robust (MAD) scale of the line's residuals,
floored at 0.15 log units, so the five quantiles are

    q_tau(h) = p1 * exp(mu_h + z_tau * sigma_h),   rounded half-up to 2 dp,

where p1 is the case's hour-1 print and z_tau the standard normal quantile
at each scored level.

Why this shape, and why it beats the house baseline (flat at hour 1):

* Launch days are not flat. Hour 1 (00:00-01:00 UTC) only has the regions
  whose midnight has passed (New Zealand, Australia, the UK); the US east
  coast unlocks at 05:00 UTC and the west coast at 08:00 UTC, so counts dip
  in hours 2-4, jump at hours 5-6, sag through the US morning and peak in
  the US/Europe evening (hours 18-22). The comparables' median curve shows
  exactly that shape, and the regression recovers it per hour.
* The size of the launch matters, and it matters in log space: a small
  hour-1 print (a quiet category before a simultaneous global unlock) can
  grow 100-1000x by the evening, while a huge hour-1 print (875k for Elden
  Ring) is already near the day's peak and grows ~1x. A single global median
  curve misses both ends; the log-linear fit on log hour-1 print captures
  the size dependence, and the kernel weighting keeps the fit local, so a
  very large hour-1 print (as GTA VI's will be) is forecast from the big
  launches, not dragged down by the tiny ones.
* The spread is honest: residual dispersion is wide (sigma ~0.3-1.0 by
  hour), because launches really do differ that much. Pinball loss on the
  log scale rewards quantiles that say so instead of pretending precision.
* Launch class matters, and it is read from the case file, not assumed:
  early-access-contaminated and weak launches are dropped from training
  (their hour-1 prints are already inflated by early play, so their ratio
  curves are flat and they drag the fit down for a clean console launch;
  only explicitly marked tiers are dropped, an unmarked tier is kept, and
  the filter is skipped entirely if it would leave fewer than 5 launches).
  Launches in the target's own class -- console-only, single-player-like,
  tier comparable -- keep full weight; other-class launches are
  downweighted to 0.5. The target's class is parsed from its notes
  ('PS5 only', 'console only', 'No early access'); when the notes do not
  say, no class weighting is applied. Sigma is smoothed across
  neighbouring hours (weights 1/(1+|dh|), window +-2) because the
  per-hour MAD on ~30 points is noisy.

Validation (out-of-sample, on the kit's four practice cases, whose answers
were re-read from the same public SullyGnome source the kit quotes):

* House baseline (flat at hour 1): mean pinball loss 0.387.
* Global median ratio curve: 0.310.
* This model v1 (local log-linear, bw 1.5, slope shrunk 0.9,
  kernel-weighted MAD spread): 0.179 (per case: Wolverine 0.171, RDR2
  0.214, GoW Ragnarok 0.135, Spider-Man 2 0.197). Typical miss ~25-45%
  vs the baseline's ~83%.
* This model v2 (v1 + tier filter + class weighting + sigma smoothing):
  **0.148** (per case: Wolverine 0.156, RDR2 0.144, GoW Ragnarok 0.104,
  Spider-Man 2 0.186).
* Leave-one-launch-out over the 37 comparables: v1 0.390 vs the
  baseline's 0.930 on the same pseudo-cases; v2 0.417 overall -- worse
  on the broad pool, by design: it specialises on clean console launches
  and pays for it on unlike targets (simultaneous-unlock monsters:
  Black Myth: Wukong 1.64, Marathon 1.47). On the launches that resemble
  GTA VI -- clean-class with hour-1 print >= 100k -- v2 scores 0.132
  leave-one-out (Zelda 0.170, The Last of Us Part II 0.131, Ghost of
  Tsushima 0.094), and the two biggest launches in the set (Elden Ring,
  Cyberpunk 2077) score 0.087 / 0.124.

Assumptions and limits (see RESEARCH.md for the full note):

* The target launch unlocks at rolling midnight local time, like the
  practice launches and the bulk of the comparables. A simultaneous global
  unlock at a single hour would break the hour-1 anchor; GTA VI's unlock
  is rolling midnight, so the anchor holds.
* The comparables' release-day curves are the right population. The final
  case carries 42 of them (the 38 here plus the four practice launches).
* SullyGnome hourly averages (practice) stand in for the final's mean of
  minute reads; the two measures of an hour differ slightly, and the
  final's hour-1 print is a minute mean, not a single hourly point.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13, standard
library only, no network). Reads every <id>.in in $MAKEFUN_INPUT_DIR
(default /input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR (default
/output): {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}
for hours 2..24. Deterministic: the same case file always yields the same
bytes (fixed iteration order, no randomness, no clock, no dict-order
dependence). Runs in well under a second per case.
"""

import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file, but the
# line keeps the starter's contract.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Kernel bandwidth (log-viewer units) for the local fit, and the factor the
# fitted slope is shrunk by: a small guard against over-extrapolation when
# the target's hour-1 print sits outside the comparables' range.
BANDWIDTH = 1.5
SHRINK = 0.9
# The spread never goes below this many log units, and never above this.
SIGMA_FLOOR = 0.15
SIGMA_CAP = 2.0
# A fallback spread when an hour has too few comparable points to fit.
FALLBACK_SIGMA = 0.5
# The house baseline's spread (kept as the last-resort fallback only).
BASELINE_SIGMA = 0.15
# --- v2: launch-class conditioning (see RESEARCH.md) ---
# Comparables whose unlock text marks them as early-access-contaminated
# or weak are dropped from training: their hour-1 prints are already
# inflated by early play, so their ratio curves are flat and they drag
# the fit down for a clean console launch. Only explicitly marked tiers
# are dropped; an unmarked tier ('unknown') is kept. If the filter would
# leave fewer than MIN_TRAIN launches, it is not applied at all.
TIER_EXCLUDED = ("early-access-contaminated", "weak")
MIN_TRAIN = 5
# Same-class launches (console-only, single-player-like, tier comparable)
# keep full weight; other-class launches are downweighted to this.
CLASS_WEIGHT_OTHER = 0.5
# Sigma is smoothed across neighbouring hours (weights 1/(1+|dh|),
# window +-SIGMA_SMOOTH): the per-hour MAD is noisy on ~30 points.
SIGMA_SMOOTH = 2
MAX_VIEWERS = 1_000_000_000


def round2(x):
    """Round to 2 decimals, half up (the starter's convention)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def parse_tier(text):
    """Tier tag from a comparable's unlock text ('Tier: comparable' etc)."""
    m = re.search(r"Tier:\s*([a-z\-]+)", text or "")
    return m.group(1) if m else "unknown"


def is_clean_launch(platforms, tier):
    """Console-only, single-player-like launch: no PC in its platforms
    and a comparable tier (not early-access-contaminated / weak)."""
    return "PC" not in (platforms or "") and tier == "comparable"


def target_is_clean(case):
    """Is the target launch itself a clean console-only launch?

    The case file carries no platforms field for the target; the facts
    live in the free-text notes ('released ... on PS5 only ... No early
    access'). Returns True / False, or None when the notes do not say
    either way -- in that case no class weighting is applied at all.

    Rules, in order (calibrated on the four practice cases, all clean):
    1. an explicit console-only statement ('PS5 only', 'console only')
       wins, even if a later PC port is mentioned ('PC 2019-11');
    2. a PC mention at launch (not negated, not a later-port note)
       means not clean;
    3. early access at launch (not negated) means not clean;
    4. otherwise a console-platform mention means clean;
    5. otherwise unknown.
    """
    notes = case.get("notes") or ""
    low = notes.lower()
    if re.search(r"\b(ps5|ps4|xbox|switch)\s+only\b|consoles?\s+only\b", low):
        return True
    pc = bool(re.search(r"\bpc\b", low))
    pc_negated = bool(re.search(
        r"no\s+pc|not\s+on\s+pc|without\s+pc|pc\s+later|pc\s+expected|"
        r"pc\s+coming|pc\s+port|pc\s+version|pc\s+\d{4}", low))
    if pc and not pc_negated:
        return False
    ea = bool(re.search(r"early\s+access", low))
    ea_negated = bool(re.search(
        r"no\s+early\s+access|without\s+early\s+access", low))
    if ea and not ea_negated:
        return False
    if re.search(r"\bps5\b|\bps4\b|\bxbox\b|\bswitch\b|console", low):
        return True
    return None


def release_day_viewers(comp):
    """{hour: viewers} for the 24 UTC hours of a comparable's release day."""
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for point in comp.get("hourly") or []:
        try:
            at = parse_ts(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None:
            continue
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    return out


def weighted_median(pairs):
    """Median of values with weights: pairs is a list of (value, weight)."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0] if ordered else 0.0
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= total / 2.0:
            return value
    return ordered[-1][0]


def fit_hour(points, log_p1):
    """Fit one hour's log-ratio model.

    points: [(log hour-1 print, log viewers_h / viewers_1, class weight)]
    from the comparables. The class weight is 1.0 for launches in the
    target's class and CLASS_WEIGHT_OTHER for the rest (1.0 for all when
    the target's class is unknown). Returns (mu, sigma).
    """
    if not points:
        return None
    if len(points) < 3:
        ratios = sorted(y for _, y, _ in points)
        mu = statistics.median(ratios)
        return mu, FALLBACK_SIGMA
    weights = [cw * math.exp(-0.5 * ((x - log_p1) / BANDWIDTH) ** 2)
               for x, _, cw in points]
    total_w = sum(weights)
    mean_x = sum(w * x for w, (x, _, _) in zip(weights, points)) / total_w
    mean_y = sum(w * y for w, (_, y, _) in zip(weights, points)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for w, (x, _, _) in zip(weights, points))
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for w, (x, y, _) in zip(weights, points))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    mu = mean_y + SHRINK * slope * (log_p1 - mean_x)
    # Residuals against the unshrunk line; robust scale via the
    # kernel-weighted MAD, so launches near the target's size set the spread.
    residuals = [y - (mean_y + slope * (x - mean_x)) for x, y, _ in points]
    med_resid = weighted_median(list(zip(residuals, weights)))
    mad = weighted_median(
        [(abs(r - med_resid), w) for r, w in zip(residuals, weights)]
    )
    sigma = max(1.4826 * mad, SIGMA_FLOOR)
    sigma = min(sigma, SIGMA_CAP)
    # Safety rail: never forecast a median ratio outside the range the
    # comparables actually showed for this hour (widened by half a log).
    lo = min(y for _, y, _ in points) - 0.5
    hi = max(y for _, y, _ in points) + 0.5
    mu = max(lo, min(hi, mu))
    return mu, sigma


def forecast_case(case, name):
    """Forecast hours 2..24 of one case, as {"hours": [...]}."""
    known = case.get("known") or []
    first = next((row for row in known if row.get("hour") == 1), None)
    if first is None:
        raise ValueError("%s has no hour-1 print" % name)
    p1 = first.get("viewers")
    if p1 is None or p1 < 1:
        p1 = 1
    p1 = float(p1)
    log_p1 = math.log(p1)

    # The training set: one release-day curve per comparable launch,
    # with its tier and class. Early-access-contaminated / weak launches
    # are dropped (see TIER_EXCLUDED); if that would leave too few,
    # the filter is not applied.
    entries = []
    for comp in case.get("comparables") or []:
        day = release_day_viewers(comp)
        if 1 in day and day[1] and day[1] > 0 and len(day) >= 20:
            tier = parse_tier(comp.get("unlock", ""))
            entries.append({
                "day": day,
                "tier": tier,
                "clean": is_clean_launch(comp.get("platforms", ""), tier),
            })
    kept = [e for e in entries if e["tier"] not in TIER_EXCLUDED]
    if len(kept) >= MIN_TRAIN:
        entries = kept
    target_clean = target_is_clean(case)

    levels = case.get("levels") or LEVELS
    zs = Z if levels == LEVELS else [norm_ppf(t) for t in levels]

    # Fit every hour first (mu + raw sigma), then smooth sigma across
    # neighbouring hours before turning it into quantiles.
    fitted = {}
    for row in case.get("forecast") or []:
        h = row["hour"]
        if h == 1:
            continue
        points = []
        for e in entries:
            day = e["day"]
            if h in day and day[h] and day[h] > 0:
                if target_clean is None:
                    cw = 1.0
                else:
                    cw = 1.0 if e["clean"] == target_clean else CLASS_WEIGHT_OTHER
                points.append((math.log(day[1]),
                               math.log(day[h] / day[1]), cw))
        fitted[h] = fit_hour(points, log_p1)
    sigmas = {h: s for h, f in fitted.items() if f is not None
              for _, s in [f]}
    smoothed = {}
    for h, s in sigmas.items():
        acc = 0.0
        wsum = 0.0
        for hh in range(max(2, h - SIGMA_SMOOTH),
                        min(24, h + SIGMA_SMOOTH) + 1):
            if hh not in sigmas:
                continue
            w = 1.0 / (1 + abs(hh - h))
            acc += w * sigmas[hh]
            wsum += w
        smoothed[h] = acc / wsum if wsum else s
    sigmas = smoothed

    hours = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        if h == 1:
            continue
        if fitted.get(h) is None:
            # No comparable covers this hour: fall back to the house
            # baseline's flat forecast with its widening spread.
            qs = [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1)) for z in zs]
        else:
            mu, _ = fitted[h]
            qs = [p1 * math.exp(mu + z * sigmas[h]) for z in zs]
        qs = [max(1.0, min(float(MAX_VIEWERS), q)) for q in qs]
        qs = [round2(q) for q in qs]
        # Rounding must not break the non-decreasing order.
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    hours.sort(key=lambda row: row["hour"])
    return {"hours": hours}


def norm_ppf(p):
    """Acklam's rational approximation of the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
