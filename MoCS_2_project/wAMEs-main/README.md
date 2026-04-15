# wAMEs: Group dynamics in higher-order contagion

This repository contains the code used to reproduce the results of the paper:

**"Group dynamics drive transition shifts and multistable active phases under collectively reinforced contagion"**

S. Lamata-Otín *et al.*

The implementation is based on an approximate master equation (wAME) framework that incorporates temporal turnover in group interactions.

---
## Abstract

Group-based reinforcement can induce discontinuous transitions from inactive to active phases in
higher-order contagion models. However, these results are typically obtained on static interaction
structures or within mean-field approximations that neglect temporal changes in group composition.
Here, we show that group dynamics is not a secondary effect but a central aspect that determines
the macroscopic transition class of higher-order contagion processes. We develop an analytically
tractable approximate master equation model that effectively interpolates between quenched and
mean-field limits through a group composition turnover rate. Our results reveal the rich impact of time-
varying structures: it can induce discontinuous phase transition, broaden the bistable region, and at
the same time promote or suppress contagion near criticality. Moreover, when real-world turnover
rates and group-size heterogeneity are taken into account, the system exhibits a qualitatively richer
phase diagram with four distinct dynamical phases, combining continuous or discontinuous transitions
with localized or delocalized activity. In localized regimes, we uncover multistable active phases with
multiple coexisting active states, which are observed in neither the annealed nor the quenched limits,
and extend classical absorbing-active bistability. Finally, we demonstrate that the emergence of discontinuous transitions in real-world systems requires stronger nonlinear reinforcement than previously
thought, indicating that simulations in static structures can yield qualitatively misleading predictions.

---

## Repository structure

The code is organized in modular components under `src/wAMEs`:

- `core.py`  
  Core utilities and structural representations of the system.

- `temporal_dynamics.py`  
  Integration of the dynamical equations (Eqs. 7–11 in the paper).

- `fixed_points.py`  
  Computation of stationary states and branch structure (Eqs. 12–13 in the paper).

- `thresholds.py`  
  Computation of invasion and tricritical thresholds (Eqs. 14–15 in the paper).

- `plotting.py`  
  Plotting utilities and styling helpers.

Additionally:

- `Data/`  
  Contains empirical group and membership distributions.

- `Files/`  
  Output directory where results are stored.

- `Example_multistability.ipynb`  
  Notebook reproducing representative results of the paper.

---

## Requirements

All simulations were performed using:

- Python 3.12.2  
- NumPy 1.26.4  
- SciPy 1.16.2  
- Matplotlib 3.9.2  
- Numba 0.60.0

# wAMEs Notes

## Notes from Contains empirical group and membership distributions

- linear: think constant rate of change
- non-linear: non-constant rate of change
- superlinear: exceeds linear change, very fast acceleration of change (i.e. $x^2$)
- Simppson's Paradox: a phenomenon in probability and statistics in which a trend appears in several groups of data but disappears or reverses when the groups are combined

## Network Data

The file `Data/group_statistics.txt` is a JSON file containing empirical contact-group statistics from **8 real proximity/contact networks**. Each entry stores two normalized probability distributions:

| Field | Meaning |
|---|---|
| `group_size_n` | Group sizes `n` with nonzero probability |
| `group_size_p` | `p(n)` — probability a randomly chosen group has size `n` |
| `membership_k` | Membership values `m` with nonzero probability |
| `membership_g` | `g(m)` — probability a randomly chosen node belongs to exactly `m` groups |

### All 8 networks at a glance

| Network | Domain | `nmax` | `mmax` | `<n>` | `<m>` |
|---|---|---|---|---|---|
| **Thiers13** | High school (used in example) | 6 | 5 | 2.31 | 1.04 |
| LyonSchool | Primary school | 6 | 5 | 2.50 | 1.09 |
| SFHH | Conference | 11 | 4 | 2.42 | 1.05 |
| InVS15 | Workplace | 7 | 6 | 2.19 | 1.02 |
| InVS13 | Workplace | 4 | 4 | 2.01 | 1.02 |
| LH10 | Hospital | 7 | 4 | 2.41 | 1.03 |
| Malawi | Village | 5 | 3 | 2.06 | 1.00 |
| CNS | University | 20 | 233 | 3.58 | 2.33 |

### Thiers13 full distributions (used in the example notebook)

```
p(n): n=2→0.7419, n=3→0.2117, n=4→0.0388, n=5→0.0074, n=6→0.0002
g(m): m=1→0.9616, m=2→0.0344, m=3→0.0039, m=4→0.0001, m=5→3e-6
```

Most nodes belong to just 1 group, and most groups have only 2 members — a sparse, contact-event style network.

---

## `state_meta` — structural metadata tuple

`load_group_statistics` calls `get_state_meta`, which returns an 8-element tuple used throughout the codebase:

| Index | Name | Content |
|---|---|---|
| 0 | `mmax` | Maximum membership (scalar int) |
| 1 | `nmax` | Maximum group size (scalar int) |
| 2 | `m` | `arange(0, mmax+1)` — membership index array |
| 3 | `gm` | Membership distribution array, length `mmax+1` |
| 4 | `pn` | Group-size distribution array, length `nmax+1` |
| 5 | `imat` | `(nmax+1, nmax+1)` matrix where `imat[n, i] = i` for valid `(n,i)` pairs |
| 6 | `nmat` | Same shape, `nmat[n, i] = n` for valid pairs |
| 7 | `pnmat` | `outer(pn, ones(nmax+1))` — `pn` broadcast across columns |

---

## `integrate_I_traj()` parameters

Defined in `src/wAMEs/temporal_dynamics.py`. Integrates the wAMEs ODE system (Eqs. 7–11 of the paper) forward in time and returns the infected-fraction trajectory.

| Parameter | Type | Role |
|---|---|---|
| `lam` | `float` | Base transmission rate λ. Scales the infection-rate function `β(n,i) = λ·i^ν`. Larger → more infectious. |
| `state_meta` | `tuple` | The 8-element structural metadata tuple from `get_state_meta`. Encodes `nmax`, `mmax`, `gm`, `pn`, and the precomputed index matrices. |
| `nmax` | `int` | Maximum group size. Determines the dimension of `fni` (the group-state matrix). |
| `mmax` | `int` | Maximum membership. Determines the length of `sm` (the node-state vector). |
| `gm` | `ndarray` | Membership distribution `g(m)`. Used to compute `I(t) = Σ_m (1 − s_m) g(m)` — the global infected fraction. |
| `mu` | `float` | Recovery rate. Infected nodes recover at rate `μ`. Sets the timescale (typically set to 1). |
| `w` | `float` | Group switching (rewiring) rate ω. At rate `w`, nodes leave their current group and join a new one drawn from the stationary distribution. Controls how fast groups remix. |
| `nu` | `float` | Synergy exponent ν in `β(n,i) = λ·i^ν`. Controls collective reinforcement: `ν=1` → linear, `ν>1` → superlinear (groups with more infected spread disproportionately faster). High `ν` (like 9.5) drives the multistability. |
| `I0` | `float` | Initial infected fraction (default `1e-5`). Used in `initialize()` to set `s_m(0) = 1 − I0` and seed `fni` binomially. |
| `traj_points` | `int` | Number of time points stored (default `200000`). Controls output resolution, not integration accuracy (solver adapts internally via LSODA). |
| `t_max` | `float` or `None` | Final integration time. If `None`, defaults to `float(traj_points)` — so the time axis runs from 0 to `traj_points`. |

### What happens internally

1. Builds `inf_mat[n,i] = λ·i^ν` for all valid `(n,i)` pairs
2. Initializes `sm` (susceptible probabilities by membership) and `fni` (group-state distribution) from `I0` via binomial seeding
3. Integrates `vector_field_w` using `scipy.solve_ivp` with the **LSODA** method (stiff/non-stiff adaptive solver)
4. Returns `(t, I(t))` where `I(t) = Σ_m (1 − s_m(t))·g(m)`

# Higher-Order Contagion Model: Extension Discussion Summary

## Context

This document summarizes a discussion about potential extensions to a temporal higher-order contagion model, specifically the AME (Approximate Master Equation) implementation in `temporal_dynamics.py`. The base model is drawn from the `gcm` codebase and formalizes group-based social contagion with two key parameters: **ν** (synergy exponent) and **ω** (group switching/annealing rate).

---

## Model Recap

### Key Parameters

- **λ (lambda):** Baseline infection/transmission rate
- **μ (mu):** Recovery rate
- **ν (nu):** Synergy exponent — controls how superlinearly the infection rate scales with the number of infected group members. Infection rate within a group = `λ * i**ν`
- **ω (omega):** Group switching/annealing rate — how fast individuals change group membership

### Key State Variables

- **n:** Total number of individuals in a group (integer count)
- **i:** Number of *infected* individuals currently in a group (integer count, not a rate)
- **fni[n, i]:** Fraction of groups of size `n` containing exactly `i` infected members

### How Switching Works (Base Model)

In `vector_field_w`, ω is a **state-blind scalar** — every node leaves its group at the same rate regardless of infection status. The `(1 - I)` and `I` factors alongside `w` describe who *replaces* a departing node (drawn randomly from the global population), not who leaves. There is no behavioral response to infection in the base model.

---

## Proposed Extensions

### 1. Diversity-Seeking Behavior (Kernel Variant)

**Concept:** Individuals who *prefer* mixed groups — they become uncomfortable when their group becomes too homogeneous and are more likely to leave.

**Implementation:** Replace scalar `ω` with a group-state-dependent function `w(n, i)` using the existing `vector_field_w_kernel` infrastructure:

```python
# Switching rate increases as i/n increases (group becomes more infected)
w_func = lambda n, i: base_w * (1 + alpha * (i / n))
```

**Interpretation:** Not about modeling a *diverse group*, but about modeling individuals who *value diversity* — their switching rate `w(n, i)` increases when `i/n` is high, because they seek out more mixed social contexts.

---

### 2. Allegiance-Driven Behavior (Kernel Variant)

**Concept:** Individuals who are loyal to their in-group — they are *less* likely to leave a group they share infection/belief status with.

**Implementation:** Also uses `vector_field_w_kernel`, but `w(n, i)` now *decreases* as `i/n` increases:

```python
# Switching rate decreases as group becomes more infected (loyal to aligned group)
w_func = lambda n, i: base_w * (1 - alpha * (i / n))
```

**Interpretation:** The individual stays put when surrounded by like-minded group members. Allegiance is modeled as a reduction in the group-leaving rate proportional to ideological alignment.

---

### 3. Heterogeneous Synergy (Group-Specific ν)

**Concept:** Rather than a single global synergy exponent ν, each group has its own ν_g — some groups exert stronger collective peer pressure than others.

**Behavioral Interpretation:**
- High ν group: tight-knit ideological community where peer pressure scales superlinearly
- Low ν (≈1) group: loose acquaintance network where individuals influence each other more or less independently

**Implementation:** Change `infection_matrix` construction from a global scalar ν to a group-size-dependent function `ν(n)`:

```python
# Before: global nu
inf_mat = infection_matrix(lambda n, i: lam * i**nu, nmax)

# After: group-specific nu
inf_mat = infection_matrix(lambda n, i: lam * i**nu_func(n), nmax)
```

`nu_func(n)` encodes how synergy varies with group size — e.g., larger groups might have higher synergy, or smaller tight-knit groups might. ν ceases to be a global parameter and becomes a property of individual groups.

**Code Impact:** Contained to how `inf_mat` is constructed. The vector field itself (`vector_field_w`) does not need structural changes.

---

## Summary Table

| Extension | Mechanism | Parameter Changed | Behavioral Story |
|---|---|---|---|
| Diversity-seeking | `w(n, i)` increases with `i/n` | Scalar ω → kernel `w_mat` | Individuals seek mixed groups |
| Allegiance | `w(n, i)` decreases with `i/n` | Scalar ω → kernel `w_mat` | Individuals loyal to aligned groups |
| Heterogeneous synergy | ν varies by group | Global ν → `ν_func(n)` | Groups differ in collective peer pressure |

---

---

## The Master Equation in `vector_field_w`

Yes — `fni_field` is where the master equation lives. The state variable `fni[n, i]` tracks the fraction of groups of size `n` that currently have exactly `i` infected members. The master equation describes how that distribution evolves over time by accounting for every way `i` can increase or decrease by 1.

There are exactly three transition types, corresponding to the three `fni_field` blocks in the code.

---

### Mean-Field Quantities (computed first)

Before the master equation terms, two global quantities are computed:

```python
r = np.sum(
    inf_mat[2:, :] * (nmat[2:, :] - imat[2:, :]) * fni[2:, :] * pnmat[2:, :]
) / np.sum((nmat[2:, :] - imat[2:, :]) * fni[2:, :] * pnmat[2:, :])
```

**`r`** is the mean infection pressure experienced by a susceptible node — a weighted average of the within-group infection rate `inf_mat[n, i] = λ * i**ν` across all group sizes and compositions, weighted by the number of susceptibles and the group size distribution. It answers: *on average, how fast is a susceptible node getting infected right now?*

```python
rho = r * excess_susceptible_membership(m, gm, sm)
```

**`rho`** is the **mean-field infection pressure from outside the current group**. When a node leaves a group and a new one joins, that new node is drawn from the global pool. `rho` captures the rate at which a susceptible node picks up infection through this background channel — i.e., from groups it is *not* currently being tracked in. This is the AME's way of closing the equations without tracking every group a node belongs to simultaneously.

---

### Transition 1: i+1 → i (infected node exits state)

```python
fni_field[2:, :nmax] += imat[2:, 1:] * (mu + w * (1.0 - I)) * fni[2:, 1:]
```

**What happens:** A group in state `(n, i+1)` loses one infected member, dropping to `(n, i)`.

**Two mechanisms cause this:**
- **Recovery** at rate `μ`: an infected node spontaneously recovers and becomes susceptible
- **Switching** at rate `w * (1 - I)`: an infected node leaves the group, and its replacement is drawn from the global pool — which is susceptible with probability `(1 - I)`

**Rate:** `(i+1) * (μ + w*(1-I))` — the `i+1` factor is `imat[2:, 1:]`, the number of infected nodes available to make this transition.

**In words:** *Groups with one more infected member feed into the current state whenever an infected node recovers or is replaced by a susceptible.*

---

### Transition 2: Diagonal outflow from state (n, i)

```python
fni_field[2:, :] += (
    -imat[2:, :] * (mu + w * (1.0 - I))
    - (nmat[2:, :] - imat[2:, :]) * (inf_mat[2:, :] + rho + w * I)
) * fni[2:, :]
```

**What happens:** The current state `(n, i)` loses probability through any transition *away* from it — either `i` decreases by 1 or increases by 1.

**Two outflow channels:**

**Infected node exits** (i decreases): rate `i * (μ + w*(1-I))`
- `i` infected nodes can each recover (rate μ) or switch out and be replaced by susceptible (rate `w*(1-I)`)

**Susceptible node becomes infected** (i increases): rate `(n-i) * (inf_mat[n,i] + rho + w*I)`
- `(n-i)` susceptible nodes can each be infected by within-group pressure (`inf_mat[n,i] = λ*i**ν`), by mean-field background pressure (`rho`), or by switching out and being replaced by an infected node from the global pool (rate `w*I`)

**In words:** *The current state drains whenever anything changes — an infected node leaves or a susceptible becomes infected.*

---

### Transition 3: i-1 → i (susceptible node exits or gets infected)

```python
fni_field[2:, 1:nmax + 1] += (
    (nmat[2:, :nmax] - imat[2:, :nmax])
    * (inf_mat[2:, :nmax] + rho + w * I)
    * fni[2:, :nmax]
)
```

**What happens:** A group in state `(n, i-1)` gains one infected member, rising to `(n, i)`.

**Three mechanisms cause this:**
- **Within-group infection** at rate `λ*(i-1)**ν`: a susceptible in a group with `i-1` infected members gets infected by group pressure
- **Mean-field infection** at rate `rho`: a susceptible gets infected via the background channel
- **Switching** at rate `w * I`: a susceptible node leaves and its replacement is drawn from the global pool, which is infected with probability `I`

**Rate:** `(n-(i-1)) * (inf_mat[n,i-1] + rho + w*I)` — the `(n-(i-1))` factor is the number of susceptibles available in state `(n, i-1)`.

**In words:** *Groups with one fewer infected member feed into the current state whenever a susceptible gets infected or is replaced by an infected newcomer.*

---

### Full Master Equation (assembled)

For group state `(n, i)`:

```
d/dt fni[n,i] = (i+1)(μ + w(1-I)) * fni[n, i+1]        # inflow from i+1
              - i(μ + w(1-I)) * fni[n, i]                 # outflow: i decreases
              - (n-i)(λi^ν + ρ + wI) * fni[n, i]          # outflow: i increases
              + (n-i+1)(λ(i-1)^ν + ρ + wI) * fni[n, i-1] # inflow from i-1
```

This is a **birth-death process on i**, where "birth" means gaining an infected member and "death" means losing one. The rates are not constant — they depend on the current group state `(n, i)`, the global prevalence `I`, and the mean-field pressure `ρ`. That coupling between local group state and global quantities is the defining feature of the AME closure.

---

## Key Distinction

Diversity-seeking and allegiance are **individual behavioral dispositions** encoded in the switching rate. Heterogeneous synergy is a **group structural property** encoded in the infection rate. These are orthogonal extensions that could in principle be combined.
