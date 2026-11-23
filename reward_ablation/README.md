# Reward-weight ablation (publication-grade)

A rigorous sensitivity study of the shaped-reward weights, designed so the
results can carry a real claim, not just "our weights are fine."

## The honest framing (read this first)

A bare "our chosen weights are valid" result is a **robustness** finding, which
reviewers rarely accept as a standalone contribution. This study is built so the
data can support a **transferable** claim instead. Which claim you can make is
decided by the results, not chosen in advance:

- **If each term is necessary AND performance is robust across a broad weight
  range, AND this holds zero-shot at 500:** the contribution is a *characterized
  safety/throughput operating region for shaped-reward UTM that transfers to 5x
  density without retuning* — a recipe others can reuse.
- **If the warning (soft, dense) term matters far more than the conflict (sparse)
  term:** the contribution is the design insight *dense soft-separation shaping
  dominates sparse conflict penalties in multi-agent routing*, with the
  sensitivity curves as evidence.
- **If a swept value clearly beats the chosen one:** report it honestly as
  near-optimal-plus-future-tuning. Still a clean result.

Do NOT frame a robustness result as more than it is. The framing follows the
numbers.

## The chosen weights (baseline)

| term      | weight | role                                  |
|-----------|--------|---------------------------------------|
| progress  | 0.001  | dense pull toward the goal            |
| arrival   | +10    | bonus for reaching the destination    |
| conflict  | -2.0   | penalty per conflict (sparse)         |
| warning   | -3.0   | soft penalty, band 1500 m -> 500 m    |

## Design

- **One weight at a time** (OAT): hold three at the chosen value, sweep the
  fourth. Isolates each weight's effect. A full grid is infeasible and would stop
  you attributing a change to any single weight.
- **Train at 100, evaluate zero-shot at 100 AND 500.** Reward weights change
  *training*, so each config is a fresh 100-drone run; the trained policy is then
  evaluated at both densities. 500 is never retrained, matching your zero-shot
  claim.
- **5 seeds**, reported with mean +/- std (I'll turn these into 95% CIs + error
  bars in the figures). This removes the n=3 low-power weakness.
- **Same 20 fixed held-out layouts for every config** (eval seeds 10000..10019),
  so cross-config comparison is fair and reproducible. **If your thesis has a
  specific held-out layout set, use those seeds instead** — tell me and I'll wire
  them in.
- **Three metrics**: conflicts/episode, throughput, and **minimum separation (m)**
  (worst-case safety, the episode-minimum horizontal distance), policy-only
  (shield off) so the reward's effect is visible.

### The sweep (~21 configs x 5 seeds = ~105 runs)

| group    | values swept                        | chosen |
|----------|-------------------------------------|--------|
| conflict | 0, 0.5, 1, **2**, 3, 4, 6           | 2      |
| warning  | 0, 1, 1.5, 2, **3**, 4, 6           | 3      |
| arrival  | 2.5, 5, **10**, 15, 20              | 10     |
| progress | 0, 0.0005, **0.001**, 0.002, 0.004  | 0.001  |

## How to read it (what supports a strong claim)

1. **Necessity** (the `=0` rows): turning a term off should clearly hurt its
   metric. Proves each term earns its place.
2. **Stability** (neighbours of the chosen value): a flat plateau around the
   chosen weight, with the chosen value on the good part of the
   safety/throughput trade-off. Proves the specific number isn't a knife-edge.
3. **Transfer**: the same shape of curve at 100 and 500. Proves the reward design,
   not just the policy, scales.
4. **Pareto**: plotting conflicts vs throughput across all configs, the chosen
   weights should sit on/near the efficient frontier.

## Files

- `run_all.sh` - **one command: patch, run everything, build figures.**
- `run_reward_ablation.py` - the runner.
- `patch_env.py` - idempotent patcher (adds the two configurable weights, keeps a `.bak`).
- `analyze_ablation.py` - builds the table + figures from the CSV.
- `PATCH_INSTRUCTIONS.md` - the manual patch, for reference.
- `continuous_utm_env_PATCHED_reference.txt` - patched copy for diffing only (not importable).

## Placement (important)

Your real code lives in an `env/` package. **Put these scripts anywhere in your
repo** (a `reward_ablation/` subfolder is fine). They auto-discover your `env/`
folder and patch your REAL `env/continuous_utm_env.py` (backup kept), so you do
NOT copy or move your env files. If auto-discovery fails, point it explicitly:

```bash
UTM_REPO=/Users/khadija/Desktop/Project/utm_bluesky bash run_all.sh
```

Do NOT keep a copy of `continuous_utm_env.py` next to these scripts, it would
shadow your real one. (The included `continuous_utm_env_PATCHED_reference.txt` is
only for diffing; it is not importable.)

## Run it once and walk away

Activate the environment you trained in (so numpy/torch/gymnasium are available),
then:

```bash
bash run_all.sh
```

That patches the env (idempotent, backup kept), runs the **full** ablation
(~21 configs x 5 seeds, train@100, eval zero-shot @100 and @500, 20 held-out
layouts), and writes the table + figures to `results/`. On macOS it wraps the
run in `caffeinate` so the Mac won't sleep. It logs to `ablation.log`.

**It's resumable.** If it stops (restart, closed lid, Ctrl-C), just run
`bash run_all.sh` again, it skips configs already in the CSV and continues.

Overrides via env vars (handy for chunking or a lighter run):

```bash
SEEDS=3 UPDATES=200 bash run_all.sh          # lighter
ONLY=conflict,warning bash run_all.sh        # just the two safety weights
EVAL_SCALES=100 bash run_all.sh              # skip the 500 eval
```

Run chunks in parallel on different machines/sessions, then combine:
`ONLY=conflict`, `ONLY=warning`, etc., each into the same CSV with `--resume`.

### Outputs (`results/`)

- `results_table.md` - every config, both scales, conflicts/throughput/min-sep.
- `sensitivity_s100.png`, `sensitivity_s500.png` - 4 panels each, conflicts +
  throughput vs each weight, error bars, chosen value marked.
- `minsep_s100.png`, `minsep_s500.png` - minimum separation vs each weight.
- `pareto_s100.png`, `pareto_s500.png` - safety/throughput trade-off, chosen
  highlighted.

Use your **real** `conflict_detector.py`. Send me `reward_ablation.csv` (or the
whole `results/` folder) and I'll turn it into the paper-ready paragraph and
tighten the figures.

### Fast preview first (recommended)

```bash
TRAIN_DRONES=30 UPDATES=60 SEEDS=1 EVAL_SCALES=30 EVAL_LAYOUTS=4 ONLY=conflict bash run_all.sh
```

Finishes in minutes and confirms the whole chain runs on your machine before you
commit the multi-day job. Preview numbers are NOT paper numbers.
