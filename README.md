# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/episodee/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/episodee/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (19 November 2026, UTC), as five quantiles (5%, 25%, 50%, 75%, 95%) per hour, from the first hour's count alone. A single Python file, standard library only, no network: at run time it calibrates itself on the case's own comparables — 42 past big game launches' hourly Twitch category averages — fitting, per hour, a kernel-weighted log-linear model of the hour's viewers (as a ratio to hour 1) against the size of hour 1. Launch days are not flat: hour 1 only has the regions whose midnight has passed, the US unlocks at 05:00/08:00 UTC, and counts peak in the US/Europe evening — and a small hour 1 can grow 100x by evening while a huge one is already near its peak. The model captures both facts, and its spread is the honest residual dispersion across launches, not a guessed width.

On the kit's four practice launches (Marvel's Wolverine, Red Dead Redemption 2, God of War Ragnarök, Marvel's Spider-Man 2) it scores a mean pinball loss of **0.179**, against **0.387** for the house baseline — typical miss ~25–45% instead of ~83%. Leave-one-launch-out over the 37 comparables: 0.390 vs the baseline's 0.930. See [RESEARCH.md](RESEARCH.md) for the method, the validation and the limits.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2–24. Deterministic — the same case always yields the same bytes — and runs in under a second per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=6nyxpkac&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
