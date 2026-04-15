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
