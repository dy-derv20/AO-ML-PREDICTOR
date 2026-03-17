# Compute Cost and Parallelism Analysis
## Hierarchical Bayesian Tennis Prediction Pipeline

---

## 1. Dataset Scale

```
Known:   2015–2026  ≈  30,000 matches  (11 years)
Derived: per year   ≈  2,727 matches

Full run: 2000–2026  =  26 years
Estimated:           ≈  70,000–75,000 matches

Feature space:
  Raw:        49 columns
  Engineered: ~80–120 columns by pipeline end
  Per match:  raw sample array of S=500–1000 
              posterior predictive values
```

---

## 2. Bottleneck Identification

Only one stage drives compute cost. Everything else is negligible by comparison.

| Stage | Operation | Estimated Time | Parallelizable? |
|---|---|---|---|
| Stage 0 | Sort, label, feature engineering | 5–15 min | Yes, trivially |
| Stage 1A | Prior initialization | < 5 min | N/A |
| **Stage 1B** | **Sequential HMC/NUTS loop** | **8–16 days (baseline)** | **Partially** |
| Stage 1C | Within-tournament state aggregation | Inline, negligible | Yes |
| Stage 2A | Logistic regression training | < 5 min | N/A |
| Stage 2B | Normalizing flow training | 1–4 hrs (GPU) / 20–60 hrs (CPU) | Yes (GPU) |
| Stage 3 | Evaluation + betting simulation | < 30 min | N/A |

**Stage 1B is the entire problem.** All optimization effort is directed here.

---

## 3. Why Stage 1B Is Slow

### 3.1 Per-Match HMC Cost

Each of the ~70,000 matches requires an independent HMC/NUTS call:

```
Per match HMC call involves:
  - 2 active players' surface-specific skill posteriors
  - Up to 6 latent dimensions in joint update
  - Burn-in / warm-up steps even when warm-started
  - S = 500–1000 posterior samples drawn
  - 4 chains run (baseline) or 2 chains (optimized)
  - R-hat and ESS diagnostics computed
```

Per-match timing on a modern CPU:

```
Configuration           Time per match
─────────────────────────────────────
4 chains, S=1000        15–30 seconds
4 chains, S=500         8–15 seconds
2 chains, S=500         5–10 seconds    ← proposed
2 chains, S=200         2–5 seconds     ← prototype
```

### 3.2 The Sequential Dependency Structure

This is the fundamental constraint. The posterior before match $t$ depends
on the posterior after match $t-1$. You cannot process match $t$ until
match $t-1$ is complete — **if those matches share a player**.

```
Player A plays match 1 on Monday
Player A plays match 2 on Wednesday
→ Match 2 posterior DEPENDS on match 1 posterior
→ These two are strictly sequential

Player A plays match 1 on Monday
Player B plays match 2 on Monday (different tournament)
→ No shared players
→ No dependency
→ These two are fully parallelizable
```

The sequential dependency is at the **player level**, not the match level.
This is the key insight that unlocks parallelism.

---

## 4. Parallelism Analysis

There are three independent levels at which parallelism exists in the ATP calendar.
Each level compounds the speedup of the others.

---

### 4.1 Level 1 — Simultaneous Tournaments

The ATP calendar regularly schedules multiple tournaments in the same week.
These tournaments involve entirely different player draws and share no
sequential dependency whatsoever.

```
Example ATP calendar weeks:

  Australian Open swing (Jan):
    AO + smaller Asian hardcourt events running simultaneously

  Clay season (Apr–May):
    Multiple 250s and 500s running alongside each other

  North American hardcourt (Aug):
    Masters 1000 + simultaneous 250s in other regions

  Regular tour weeks:
    Typically 2–4 tournaments per week globally
```

Full parallelism across simultaneous tournaments:

```
Week with 3 simultaneous tournaments:
  Tournament A  →  Process independently
  Tournament B  →  Process independently   (all 3 in parallel)
  Tournament C  →  Process independently

Speedup: 3x reduction in wall clock for that week
```

Over the full 26-year dataset:

```
~1,352 tournament weeks in 2000–2026
Average simultaneous tournaments per week: ~2.5
Effective sequential tournament weeks: ~540
```

---

### 4.2 Level 2 — Within-Tournament Round Parallelism

Within a single tournament, all matches belonging to the same round
involve different players. They share no sequential dependency.

```
Grand Slam round structure:
  R128:  64 matches   →  64 independent HMC updates
  R64:   32 matches   →  32 independent HMC updates
  R32:   16 matches   →  16 independent HMC updates
  R16:   8 matches    →  8 independent HMC updates
  QF:    4 matches    →  4 independent HMC updates
  SF:    2 matches    →  2 independent HMC updates
  F:     1 match      →  1 HMC update (irreducible)

Masters 1000 (96 draw):
  R96:   48 matches   →  48 independent
  ...

ATP 250 (28 draw):
  R32:   16 matches   →  16 independent
  ...
```

**Within a round, both pre-match feature extraction AND
post-match posterior updates are fully parallelizable.**

Pre-match: all players entering a round already have their posteriors
from the previous round — no within-round dependency.

Post-match: match results within the same round do not affect
each other's priors — updates are independent.

```
Barrier points (unavoidable sequential steps):
  - Between rounds: R1 must complete before R2
  - Within-tournament state aggregation after each round:
      M_i(r), Q_i(r), surprise_i computed once 
      all round results are known
  - These barriers are cheap — pure arithmetic,
    not HMC sampling
```

---

### 4.3 Level 3 — Within a Single HMC Call (Chain Reduction)

Reducing from 4 chains to 2 chains per match:

```
4 chains → 2 chains:
  Per-match time reduction: ~40–50%
  Diagnostic tradeoff:
    R-hat still computable and valid
    ESS per chain must be higher to compensate
    Mixing problems slightly harder to detect
  Mitigation:
    Warm-starting from previous posterior means
    chains initialize in a good region — 
    pathological mixing is unlikely
    Risk is low for sequential warm-started updates
```

This is the simplest optimization — no code restructuring required,
just a parameter change in the sampler configuration.

---

## 5. Combined Speedup Calculation

### 5.1 Baseline (No Optimization)

```
70,000 matches × 15 seconds (4 chains, S=500)
= 1,050,000 seconds
= ~291 hours
= ~12 days wall clock (single core, sequential)
```

### 5.2 Optimization 1 — Chain Reduction Only

```
2 chains instead of 4:
  Per-match time: ~8 seconds

70,000 × 8 seconds = 560,000 seconds = ~6.5 days

Saving vs baseline: ~5.5 days
Speedup factor:     ~1.9x
```

### 5.3 Optimization 2 — Round Parallelism Added

```
Key reframe: the unit of sequential processing 
is no longer matches — it is rounds.

ATP calendar 2000–2026:
  ~1,352 tournament weeks
  ~3–7 rounds per tournament
  ~4,000–6,000 total rounds (sequential barrier points)

Average matches per round across all tournament sizes:
  Grand Slam R1:      64 matches
  Masters 1000 R1:    48 matches
  ATP 500 R1:         16 matches
  ATP 250 R1:         16 matches
  Weighted average:   ~12–15 matches per round

With 8 parallel workers (8-core machine):
  Each round of 12–15 matches runs in:
    ceil(15 / 8) = 2 batches
    2 × 8 seconds = 16 seconds per round

  Total: 5,000 rounds × 16 seconds = 80,000 seconds
       = ~22 hours

With 16 parallel workers (16-core machine):
  ceil(15 / 16) = 1 batch per round
  5,000 rounds × 8 seconds = 40,000 seconds
  = ~11 hours
```

### 5.4 Optimization 3 — Tournament Parallelism Added

```
Simultaneous tournaments in same week 
are fully independent — process in parallel.

Average 2.5 simultaneous tournaments per week:
  Effective sequential load per week reduced by ~2.5x

Combined with round parallelism on 8 cores:

  Effective sequential rounds: ~5,000 / 2.5 = ~2,000
  Time per round: 16 seconds (8 workers)
  
  Total: 2,000 × 16 seconds = 32,000 seconds
       = ~9 hours

On 16 cores:
  2,000 rounds × 8 seconds = 16,000 seconds
  = ~4.5 hours
```

### 5.5 Summary Table

```
Configuration                          Wall Clock
──────────────────────────────────────────────────
Baseline (4 chains, sequential)        ~12 days
2 chains only                          ~6.5 days
+ Round parallelism (8 cores)          ~22 hours
+ Tournament parallelism (8 cores)     ~9 hours
+ Round + tournament (16 cores)        ~4–5 hours
Full optimization (16 cores, 2 chains) ~4–6 hours
──────────────────────────────────────────────────
Total reduction: ~40–70x speedup over baseline
```

> ⚠️ These are estimates. Actual wall clock depends heavily on:
> Python multiprocessing spawn overhead, PyMC/Stan thread safety
> configuration, I/O throughput writing sample arrays to disk,
> and memory pressure from holding multiple HMC states simultaneously.
> Real-world performance should be profiled on a subset first.

---

## 6. Implementation Structure

The parallelism structure maps directly onto a nested execution graph:

```python
# Pseudocode — execution structure only, not runnable

for week w in calendar_weeks:                    # sequential
    tournaments = get_tournaments(week=w)
    
    parallel_execute(tournaments):               # LEVEL 1 PARALLEL
        for tournament T in tournaments:
            
            for round r in T.rounds:             # sequential within T
                matches = get_matches(T, round=r)
                
                parallel_execute(matches):       # LEVEL 2 PARALLEL
                    for match m in matches:
                        
                        # Pre-match
                        features = extract_features(m)
                        samples  = run_hmc(                # LEVEL 3
                                     players = m.players, #  2 chains
                                     warm_start = True,
                                     chains = 2,
                                     S = 500)
                        record(features, samples, label=m.outcome)
                        
                        # Post-match posterior update
                        update_posterior(m.players, m.outcome)
                
                # BARRIER — full round complete
                aggregate_tournament_state(T, round=r)
                # M_i(r), Q_i(r), surprise_i computed here
```

### 6.1 The Barrier Points

```
Two unavoidable synchronization points:

1. Between rounds (within a tournament):
   Cost: pure arithmetic — M_i(r), Q_i(r), 
         surprise_i aggregation
   Time: milliseconds
   Cannot be avoided — next round matchups 
   depend on who advanced

2. Between weeks (across tournaments):
   Cost: none — just scheduling
   Some player posteriors must propagate 
   forward if a player appears in consecutive 
   weeks, but this is a lookup, not a recompute
```

### 6.2 Memory Considerations

```
Per match storage:
  Feature vector:         ~120 floats   ≈  1 KB
  Raw sample array S=500: 500 floats    ≈  4 KB
  Total per match:        ~5 KB

Full dataset:
  70,000 matches × 5 KB = ~350 MB

This fits entirely in RAM on any modern machine.
No disk I/O bottleneck for sample storage.

Active HMC states in parallel:
  16 workers × 6 latent dims × 2 chains × 500 samples
  ≈ 16 × 6 × 2 × 500 × 8 bytes ≈ ~750 KB active
  Negligible memory pressure
```

---

## 7. Framework Recommendation

The choice of probabilistic programming framework affects
parallelism implementation significantly.

| Framework | Sequential HMC | Multiprocessing | Thread Safety | Notes |
|---|---|---|---|---|
| **PyMC** | Clean API | Via `multiprocessing` | GIL issues with threads — use processes | Most Pythonic, best for prototyping |
| **CmdStanPy** | Excellent | Native parallel chains, clean subprocess model | Very clean | Best for production runs, Stan is faster per sample |
| **NumPyro** | JAX-based | GPU-native | Excellent | Best if GPU available for HMC |
| **Blackjax** | JAX-based | GPU-native | Excellent | Most control, most setup cost |

**Recommendation:**
- Prototype Stage 1B in PyMC with `multiprocessing.Pool`
- If wall clock on prototype is too slow, port to CmdStanPy
- If GPU is available, NumPyro enables GPU-accelerated HMC
  which changes the cost profile significantly

> ⚠️ PyMC's interaction with Python multiprocessing requires
> careful handling — models must be reconstructed inside each
> worker process, not passed directly. This is a known pattern
> but adds boilerplate. CmdStanPy avoids this entirely since
> Stan compiles to a standalone binary.

---

## 8. Phased Execution Plan

Rather than running the full 70,000-match dataset immediately,
a phased approach validates the architecture at each scale
before committing full compute.

```
Phase 1 — Architecture Validation
  Data:     2022–2026  (~13,500 matches)
  Config:   Sequential, 2 chains, S=200
  Purpose:  Confirm pipeline runs end-to-end,
            catch bugs, profile per-match timing
  Time:     2–6 hours (sequential, no parallelism)
  Cost:     $0 (local machine)

Phase 2 — Parallelism Validation  
  Data:     2019–2026  (~21,800 matches)
  Config:   2 chains, S=500, round + tournament parallelism
  Purpose:  Confirm parallel execution is correct,
            measure actual vs estimated speedup,
            generate first real results
  Time:     1–3 hours (8 cores)
  Cost:     $0 (local machine)

Phase 3 — Medium Run
  Data:     2015–2026  (~30,000 matches)
  Config:   Full optimized pipeline
  Purpose:  Test whether pre-2019 history improves results,
            train and evaluate flow model fully
  Time:     2–4 hours (8–16 cores)
  Cost:     $0–20 (local or small cloud instance)

Phase 4 — Full Run
  Data:     2000–2026  (~70,000 matches)
  Config:   Full optimized pipeline
  Purpose:  Full research results, final evaluation
  Time:     4–8 hours (16 cores)
  Cost:     $0 (local 16-core) or $20–80 (cloud)

Phase 5 — Flow Training
  Data:     Cached Stage 1B outputs from Phase 4
  Config:   GPU training (Colab Pro+ or cloud GPU)
  Purpose:  Train normalizing flow on full feature set
  Time:     1–4 hours (GPU)
  Cost:     $0–30 (Colab Pro+)
```

**Critical property of this plan:** Stage 1B outputs are cached after
each phase. Once the sequential HMC loop has run, it never needs to
run again. All Stage 2 experimentation — different flow architectures,
different feature combinations, different classifiers — runs on top
of the cached outputs in minutes.

---

## 9. Cloud Cost Reference

If local compute is insufficient for the full run:

```
Provider          Instance          vCPUs   $/hr    Est. hrs   Total
──────────────────────────────────────────────────────────────────────
AWS EC2           c6i.4xlarge       16      $0.68   6–8        $4–6
AWS EC2           c6i.8xlarge       32      $1.36   3–5        $4–7
Google Cloud      c2-standard-16    16      $0.67   6–8        $4–6
Google Cloud      c2-standard-30    30      $1.24   3–5        $4–6
Lambda Labs       1x A100 (GPU)     —       $1.10   1–2 (flow) $1–2
Google Colab Pro+ A100 GPU          —       ~$50/mo 1–4 (flow) $0–50/mo
──────────────────────────────────────────────────────────────────────

Full research compute budget estimate: $10–80 total
Most likely scenario (local 8-core + Colab for flow): ~$0–30
```

---

## 10. Key Findings Summary

```
1. The entire compute problem reduces to Stage 1B.
   Everything else is fast.

2. The sequential dependency is at the player level,
   not the match level. Matches within the same round
   involving different players are fully independent.

3. Three compounding parallelism levels exist:
     Level 1: Simultaneous tournaments (~2.5x)
     Level 2: Within-round matches (~10–15x on 16 cores)
     Level 3: 2 chains instead of 4 (~2x)

4. Combined speedup: ~40–70x over naive baseline.
   12 days → 4–6 hours on a 16-core machine.

5. Stage 1B outputs should be cached immediately.
   All downstream experimentation is then fast and
   fully iteratable without re-running HMC.

6. Phased execution is strongly recommended.
   Run 2022–2026 first. Validate. Then extend.
   Do not commit full compute until architecture
   is confirmed correct on a subset.

7. Total research compute cost: $10–80.
   This is an unusually affordable research project
   given its methodological complexity.
```
