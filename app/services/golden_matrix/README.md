# Golden Matrix

Design matrix generator for MindGenomics and Rule Developing Experimentation (RDE), extending the Gofman & Moskowitz methodology to non-square experimental configurations.

## Background

Howard Moskowitz's MindGenomics methodology uses experimental designs where:

- **Silos** (categories) contain mutually-exclusive elements
- Each task (vignette) presents a partial profile: some silos active, some absent
- **Silo absence is a treatment condition** — what happens when a factor isn't present is as informative as when it is
- Each respondent sees a **unique set of tasks** (Isomorphic Permuted Experimental Designs / IPED)
- **Individual-level OLS regression** on each respondent's data, not aggregate models
- Mind-set segments emerge from clustering individual coefficient patterns

The standard IPED approach (Gofman & Moskowitz) starts with a base orthogonal design and permutes element labels to create respondent-unique variants. This works well for **square configurations** (e.g., 4 silos x 4 elements each) where orthogonal arrays exist.

## What Golden Matrix does

Golden Matrix generates designs for **non-square configurations** where no base orthogonal array exists — for example, 6 silos with 4, 5, 6, 5, 4, 3 elements respectively. It satisfies the same requirements as IPED:

- Every respondent sees a unique design
- No two rows or columns are identical
- At most one element per active silo per task
- Silos can be absent (partial profiles)
- Each respondent's block is individually solvable for OLS regression (full rank)
- Element exposure is balanced within and across respondents

Instead of permuting a base design, it constructs each respondent's block via greedy stochastic search with dual-objective balance scoring, then polishes for D-efficiency.

## Quick start

```python
from golden_matrix import DesignConfig, generate_golden_matrix

# Non-square: 6 silos with different element counts
config = DesignConfig(
    n_respondents=10,
    tasks_per_respondent=42,
    n_categories=6,
    elements_per_category=[4, 5, 6, 5, 4, 3],  # P=27
    min_actives_per_row=2,
    max_actives_per_row=4,
)

df, report = generate_golden_matrix(config)
df.to_csv("design.csv", index=False)

# df has columns: respondent, task, A1..A4, B1..B5, C1..C6, D1..D5, E1..E4, F1..F3
# report["X"] is the raw 420x27 numpy array
# report["global_max_vif"] is the VIF score
```

## How it works

1. **Configure** — define silos, elements per silo, respondent count, tasks, activation range, optional element priors.
2. **Preflight** — validates feasibility: row variety, structural VIF floor, T/P headroom, exposure density. Fails fast with actionable recommendations.
3. **Greedy build** — for each respondent, selects T rows from 100 random candidates per slot, scoring by dual-objective balance (local within respondent + global across all respondents).
4. **Exchange polish** — pairwise row swaps to improve D-efficiency without degrading balance beyond 5%.
5. **Shuffle** — randomises task order within each respondent's block.
6. **Diagnostics** — verifies individual-level rank, global VIF, no dead/duplicate columns.

### Why silo absence balance is free

Because exactly one element is active per active silo, balancing element exposure automatically balances silo activation. If each of A1-A4 appears ~5 times in 42 tasks, silo A is active ~20 times and absent ~22 times. No separate silo-level balancing is needed — it is a mathematical consequence of the element-level balance under the one-per-silo constraint.

## Design properties

For a configuration with `elements_per_category=[4,5,6,5,4,3]`, `T=42`, `min=2`, `max=4`:

| Property | Value |
|---|---|
| Total elements (P) | 27 |
| Degrees of freedom per respondent | 15 |
| T/P ratio | 1.56 (comfortable) |
| Structural VIF floor | ~1.3 |
| Row variety | 7,892 distinct possible rows |
| Exposure per element per respondent | ~4.7 |

## Options

### Element priors

Shift exposure toward uncertain elements:

```python
# First element in each silo gets 2x exposure
priors = [2,1,1,1] + [2,1,1,1,1] + [2,1,1,1,1,1] + [2,1,1,1,1] + [2,1,1,1] + [2,1,1]

config = DesignConfig(
    ...,
    element_priors=priors,
)
```

### Custom silo and element names

```python
config = DesignConfig(
    n_respondents=50,
    tasks_per_respondent=36,
    n_categories=3,
    elements_per_category=[4, 4, 4],
    min_actives_per_row=1,
    max_actives_per_row=3,
    category_names=["Taste", "Texture", "Aroma"],
    element_names=[
        ["sweet", "sour", "bitter", "umami"],
        ["crunchy", "smooth", "chewy", "creamy"],
        ["floral", "earthy", "citrus", "smoky"],
    ],
)
# Columns: Taste:sweet, Taste:sour, ..., Aroma:smoky
```

### Incompatible pairs

Prevent specific element combinations from appearing in the same task:

```python
config = DesignConfig(
    ...,
    incompatible_pairs=[(0, 4), (2, 7)],  # column index pairs
)
```

### Polish and VIF control

```python
config = DesignConfig(
    ...,
    polish=True,       # D-efficiency exchange polish (default: on)
    max_vif=10.0,      # VIF threshold (adaptively relaxed on retry)
)
```

## Output

`generate_golden_matrix(config)` returns `(df, report)`:

- **df** — pandas DataFrame: `respondent`, `task`, then silo-labeled binary columns
- **report** — dict:
  - `status` — "PASS"
  - `global_max_vif` — achieved VIF
  - `X` — raw N x P numpy array
  - `preflight` — feasibility check results
  - `n_rows`, `n_cols`

## When to use / when not to use

**Use Golden Matrix for any MindGenomics / RDE study:**
- Square configurations (4x4, 6x6) — works the same as IPED but without needing a pre-computed base array
- Non-square configurations (4,5,6,5,4,3) — the case IPED cannot handle
- Variable factor activation (partial profiles with absent silos)
- Individual-level analysis where each respondent needs a unique, solvable design

## Validating the design

`validate_design()` tests whether the design can distinguish planted signal from pure noise. It simulates respondent-level heterogeneity (different coefficients per respondent, as in real mind-set studies) and compares signal vs null conditions:

```python
from golden_matrix import validate_design

result = validate_design(report["X"], config)
# result["verdict"]              — PASS / WEAK / FAIL
# result["individual_recovery"]  — correlation true vs estimated (per respondent, no intercept)
# result["group_recovery"]       — correlation true vs estimated (pooled, with intercept)
# result["signal_r_squared"]     — R-squared under signal condition
# result["null_r_squared"]       — R-squared under pure noise (should be near zero)
# result["false_positive_rate"]  — fraction of coefficients falsely significant under null
# result["individual_rank_ok"]   — fraction of respondent blocks with full rank
```

Typical results for `[4,5,6,5,4,3]` with T=42, R=10:

| Metric | Value | Notes |
|---|---|---|
| Individual recovery | 0.48 | Normal for MindGenomics (15 DOF, heterogeneous respondents) |
| Group recovery | 0.84 | Recovers shared mean signal across respondents |
| Signal R-squared | 0.24 | Clearly above null — design detects signal |
| Null R-squared | 0.06 | Near zero under pure noise — no false structure |
| False positive rate | 20% | Expected with tight DOF (14); controlled by design |
| Individual rank OK | 1.0 | All respondents individually solvable |

## Auto task count

Set `tasks_per_respondent=0` to auto-compute the minimum comfortable task count (T/P >= 1.5):

```python
config = DesignConfig(
    n_respondents=10,
    tasks_per_respondent=0,  # auto: computes T=41 for P=27
    n_categories=6,
    elements_per_category=[4, 5, 6, 5, 4, 3],
    min_actives_per_row=2,
    max_actives_per_row=4,
)
```

## Minimum configuration

The smallest viable design is 3 silos with 2 elements each (P=6). Below that, the design space has too few distinct rows for unique, non-collinear tasks.

| Config | P | Auto T | DOF | Works? |
|---|---|---|---|---|
| [2, 2, 2] | 6 | 9 | 3 | Yes (marginal — low DOF) |
| [3, 3, 3] | 9 | 14 | 5 | Yes |
| [4, 4, 4, 4] | 16 | 24 | 8 | Yes |
| [4, 5, 6, 5, 4, 3] | 27 | 41 | 14 | Yes (comfortable) |

**Use other tools when:**
- Aggregate-level choice models where a shared D-optimal design is sufficient (JMP)
- Sequential Bayesian adaptive designs where each respondent's design depends on their prior answers (idefix)
