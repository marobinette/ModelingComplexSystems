# Measuring Localization: Non-Kernel vs Kernel

## What is Localization?

The **effective participation ratio** P quantifies how concentrated infection is across group sizes:

```
P = (Σ_n p_n I_n²)² / (Σ_n p_n I_n⁴)  ∈  [p_{n_max}, 1]
```

where `I_n` is the mean infected fraction within groups of size `n`. P is large when infection is spread evenly across all group sizes (delocalized) and small when it is concentrated in a subset (localized, typically the largest groups). Defined in the paper around Fig. 3; implemented as `1 / y4_tilde(In, pn)` in `core.py`.

---

## Step-by-Step: Non-Kernel Version

### Step 1 — Build the infection rate matrix

**Where:** `core.py:189`, called from `fixed_points.py:157`

```python
inf_mat = build_inf_mat(lam, nu, nmax)
# inf_mat[n, i] = λ · iᵛ  for valid (n, i) pairs
```

Uses `infection_matrix` internally, which loops `i` from `0` to `n-1` (no susceptibles when `i=n`).

---

### Step 2 — Find fixed points analytically via self-consistency

**Where:** `fixed_points.py:3` (`find_fixed_points_for_lambda`)

The scalar ω allows the stationary `f_{n,i}` to be written analytically as a function of a single scalar `r` (the mean-field infection pressure). The fixed points are roots of the 1D self-consistency equation:

```
r = M(r)
```

`mf_map_w` evaluates `M(r)` by reconstructing the stationary state from `r` and reading back the implied `r`. `brentq` finds the roots on a log grid. Each root is a fixed point; its stability is determined from the slope `M'(r*)`.

This step returns a list of `(r*, slope, is_stable)` tuples.

---

### Step 3 — Reconstruct the full stationary state from r*

**Where:** `core.py:271` (`state_from_mf_w`)

```python
v = state_from_mf_w(r_star, mu, w, inf_mat, state_meta)
sm, fni = unflatten(v, state_meta)
```

Given `r*`, analytically reconstructs `sm` and `f_{n,i}` using a row-by-row recurrence. This is possible because scalar ω decouples the stationary equations for each group size `n`.

---

### Step 4 — Compute I_n from fni

**Where:** `core.py:415` (`compute_In_from_fni`)

```python
In = compute_In_from_fni(fni, nmax)
# In[n] = Σ_i (i/n) · f_{n,i}
```

For each group size `n`, averages the infected fraction over all compositions `i`.

---

### Step 5 — Compute Ỹ₄ and P

**Where:** `core.py:439` (`y4_tilde`), `fixed_points.py:177`

```python
Y4 = y4_tilde(In, pn_filtered)
# Ỹ₄ = Σ_n p_n I_n⁴ / (Σ_n p_n I_n²)²

P = 1.0 / Y4
```

P is stored alongside I* and stability for each fixed point in `collect_fixed_points_by_lam`.

---

## Step-by-Step: Kernel Version — What Needs to Change

### Step 1 — Build the infection rate matrix

**Status: Bug exists, functionally harmless**

`build_kernel_inf_mat` currently calls `switching_matrix` instead of `infection_matrix`, so it sets `inf_mat[n, n] = λ · nᵛ` instead of leaving it 0. This entry is always multiplied by `(n - i) = 0` in the vector field, so results are unaffected — but it is conceptually wrong.

**Fix:** Replace `build_kernel_inf_mat` with a direct call to `build_inf_mat`. The infection rate does not change between scalar and kernel versions — only the switching rate does.

Also build the switching rate matrix:
```python
inf_mat = build_inf_mat(lam, nu, nmax)          # same as non-kernel
w_mat   = switching_matrix(w_func, nmax, args=w_args)  # new
```

---

### Step 2 — Find fixed points

**Status: No equivalent exists**

`mf_map_w` and `state_from_mf_w` hardcode scalar ω in the row-by-row recurrence that builds stationary `f_{n,i}`. With a kernel `w(n, i)`, the stationary equations for different `(n, i)` entries couple in a way that cannot be reduced to a 1D self-consistency equation.

**Fix:** Integrate the ODE forward until convergence and treat the final state as the fixed point. To capture both branches of any bistable region, run from two initial conditions per λ:
- Low: `I0 = 1e-5` (invasion from near-zero)
- High: `I0 ≈ 0.99` (persistence from near-full infection)

---

### Step 3 — Reconstruct the full stationary state

**Status: `integrate_I_traj_kernel` discards `v_final`**

Currently `integrate_I_traj_kernel` computes `sol.y.T` but only extracts `I(t)` and throws away the full state. The final state vector `v_final = sol.y.T[-1]` contains both `sm` and `fni` and is needed to compute P.

**Fix:** Add a variant (or optional return flag) to `integrate_I_traj_kernel` that returns `v_final`:

```python
def integrate_full_traj_kernel(...) -> (t, I_traj, v_final):
    ...
    v_final = sol.y.T[-1]
    return t, I_traj, v_final
```

Then recover `fni` with the existing `unflatten`:
```python
sm, fni = unflatten(v_final, state_meta)
```

---

### Step 4 — Compute I_n from fni

**Status: No change needed**

`compute_In_from_fni(fni, nmax)` works identically — it only depends on `fni`, which is obtained from `v_final` above.

---

### Step 5 — Compute Ỹ₄ and P

**Status: No change needed**

`y4_tilde(In, pn)` and `P = 1.0 / Y4` are identical.

---

## Summary Table

| Step | Non-Kernel | Kernel — Status | Kernel — Fix Needed |
|---|---|---|---|
| 1. Build inf_mat | `build_inf_mat` | Bug in `build_kernel_inf_mat` (harmless) | Use `build_inf_mat` directly |
| 2. Find fixed points | `mf_map_w` root-finding (1D, analytical) | No equivalent | Integrate ODE to steady state from high and low I0 |
| 3. Get fni at steady state | `state_from_mf_w` (analytical recurrence) | `v_final` discarded by integrator | Add `integrate_full_traj_kernel` returning `v_final` |
| 4. Compute I_n | `compute_In_from_fni` | Identical | None |
| 5. Compute P | `y4_tilde` → `1/Y4` | Identical | None |

---

## Hypothesis

The allegiance kernel:
```python
w(n, i) = scale * 4 * (i/n) * (1 - i/n)
```
makes nodes flee mixed groups and remain in homogeneous ones (all-S or all-I). This is expected to concentrate infection in the largest all-infected groups, reducing P (more localized) relative to the scalar-ω baseline and potentially widening the bistable region.
