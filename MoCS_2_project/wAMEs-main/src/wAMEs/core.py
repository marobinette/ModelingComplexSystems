import numpy as np
from numba import jit
from scipy.optimize import brentq
from matplotlib.ticker import LogLocator, FuncFormatter
from scipy.special import gamma
from scipy.stats import binom
from scipy.integrate import solve_ivp
import json

def get_state_meta(mmax, nmax, gm, pn):
    """
    From https://github.com/gstonge/gcm

    Return a tuple containing structural metadata for the system.

    Parameters
    ----------
    mmax : int
        Maximum membership.
    nmax : int
        Maximum group size, with nmax >= 2.
    gm : array_like
        Membership distribution of length mmax + 1.
    pn : array_like
        Group-size distribution of length nmax + 1.

    Returns
    -------
    tuple
        Tuple containing useful arrays that describe the structure.
    """
    m = np.arange(0, mmax + 1)
    imat = np.zeros((nmax + 1, nmax + 1))
    nmat = np.zeros((nmax + 1, nmax + 1))

    for n in range(2, nmax + 1):
        imat[n, 0:n + 1] = np.arange(n + 1)
        nmat[n, 0:n + 1] = np.ones(n + 1) * n

    pnmat = np.outer(pn, np.ones(nmax + 1))
    return (mmax, nmax, m, np.array(gm), np.array(pn), imat, nmat, pnmat)


@jit(nopython=True)
def flatten(sm, fni, state_meta):
    """
    From https://github.com/gstonge/gcm

    Flatten the node and group states into a single vector.

    Parameters
    ----------
    sm : ndarray
        Node-state vector, where sm[m] is the probability that a node of
        membership m is susceptible.
    fni : ndarray
        Group-state distribution, where fni[n, i] is the fraction of groups of size n that contain i infected nodes.

    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.

    Returns
    -------
    ndarray
        Flattened state vector suitable for numerical integration.
    """
    nmax = state_meta[1]
    return np.concatenate((sm, fni.reshape((nmax + 1) ** 2)))


def unflatten(v, state_meta):
    """
    From https://github.com/gstonge/gcm

    Reconstruct the node and group states from a flattened vector.

    Parameters
    ----------
    v : ndarray
        Flattened state vector.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.

    Returns
    -------
    tuple of ndarray
        (sm, fni) where sm is the node-state vector and fni is the
        group-state matrix.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    return v[:mmax + 1], v[mmax + 1:].reshape((nmax + 1, nmax + 1))

@jit(nopython=True)
def infected_fraction(sm, gm):
    """
    From https://github.com/gstonge/gcm

    Compute the total fraction of infected nodes.

    Parameters
    ----------
    sm : ndarray
        Probability that a node of membership m is susceptible.
    gm : ndarray
        Membership distribution.

    Returns
    -------
    float
        Fraction of infected nodes.
    """
    return np.sum((1.0 - sm) * gm)

@jit(nopython=True)
def excess_susceptible_membership(m, gm, sm):
    """
    From https://github.com/gstonge/gcm

    Compute the excess membership of a susceptible node reached by following
    a random group-edge.

    Parameters
    ----------
    m : ndarray
        Membership values.
    gm : ndarray
        Membership distribution.
    sm : ndarray
        Probability that a node of membership m is susceptible.

    Returns
    -------
    float
        Excess membership of susceptible nodes.
    """
    return np.sum(m * (m - 1) * sm * gm) / np.sum(m * sm * gm)

def infection_matrix(beta, nmax, args=()):
    """
    From https://github.com/gstonge/gcm

    Return the infection-rate matrix evaluated at each pair (n, i).

    Parameters
    ----------
    beta : callable
        Infection-rate function of the form beta(n, i, *args).
    nmax : int
        Maximum group size, with nmax >= 2.
    args : tuple, optional
        Additional arguments passed to the infection-rate function.

    Returns
    -------
    ndarray
        Array of shape (nmax + 1, nmax + 1) containing the infection rate
        at each pair (n, i).
    """
    inf_mat = np.zeros((nmax + 1, nmax + 1))
    for n in range(2, nmax + 1):
        for i in range(n):
            inf_mat[n, i] = beta(n, i, *args)
    return inf_mat


def beta_power_synergy(n, i, lam, nu):
    """
    Power-law synergistic infection rate.

    Parameters
    ----------
    n : int
        Group size.
    i : int
        Number of infected individuals in the group.
    lam : float
        Base infection rate (λ).
    nu : float
        Synergy exponent controlling nonlinear reinforcement.

    Returns
    -------
    float
        Infection rate β(n, i) = λ * i^ν.
    """
    return lam * (i ** nu)

def build_inf_mat(lam, nu, nmax):
    """
    Construct the infection-rate matrix for the power-law synergy model.

    Parameters
    ----------
    lam : float
        Base infection rate (λ).
    nu : float
        Synergy exponent.
    nmax : int
        Maximum group size.

    Returns
    -------
    ndarray
        Infection-rate matrix β(n, i) evaluated for all (n, i).
    """
    return infection_matrix(beta_power_synergy, nmax, args=(lam, nu))


def switching_matrix(w_func, nmax, args=()):
    """
    Construct the group-switching-rate matrix evaluated at each (n, i) pair.

    Analogous to ``infection_matrix``, but for the group switching rate ω.
    The resulting matrix can be passed to ``vector_field_w_kernel`` in place
    of a scalar ω, allowing the switching rate to depend on group size ``n``
    and the number of infected members ``i``.

    Parameters
    ----------
    w_func : callable
        Switching-rate function of the form ``w_func(n, i, *args)``.
        Returns the rate at which a node in a group of size ``n`` with ``i``
        infected members switches to a new group.
    nmax : int
        Maximum group size.
    args : tuple, optional
        Additional arguments passed to ``w_func``.

    Returns
    -------
    ndarray
        Array of shape ``(nmax + 1, nmax + 1)`` where entry ``[n, i]`` is
        ``w_func(n, i, *args)`` for valid pairs and 0 elsewhere.
    """
    w_mat = np.zeros((nmax + 1, nmax + 1))
    for n in range(2, nmax + 1):
        for i in range(n + 1):
            w_mat[n, i] = w_func(n, i, *args)
    return w_mat

@jit(nopython=True)
def mf_from_state(fni, inf_mat, state_meta):
    """
    From https://github.com/gstonge/gcm

    Compute the mean-field quantity r from the group-state distribution.

    Parameters
    ----------
    fni : ndarray
        Group-state distribution, where fni[n, i] is the fraction of groups of size n that contain i infected nodes.
    inf_mat : ndarray
        Infection-rate matrix.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.

    Returns
    -------
    float
        Mean-field infection pressure r.
    """
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]

    r = np.sum(inf_mat[2:, :] * (nmat[2:, :] - imat[2:, :]) * fni[2:, :] * pnmat[2:, :])
    r /= np.sum((nmat[2:, :] - imat[2:, :]) * fni[2:, :] * pnmat[2:, :])
    return r

@jit(nopython=True)
def state_from_mf_w(r, mu, w, inf_mat, state_meta):
    """
    Adapted from https://github.com/gstonge/gcm

    Reconstruct the stationary node and group states from a mean-field value r
    in the presence of rewiring at rate w.

    Parameters
    ----------
    r : float
        Mean-field infection pressure.
    mu : float
        Recovery rate.
    w : float
        Rewiring rate.
    inf_mat : ndarray
        Infection-rate matrix.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.

    Returns
    -------
    ndarray
        Flattened stationary state vector containing both node and group states.
    """
    nmax = state_meta[1]
    m = state_meta[2]
    gm = state_meta[3]
    pn = state_meta[4]

    # Node state
    sm = mu / (mu + m * r)
    rho = r * excess_susceptible_membership(m, gm, sm)
    I = infected_fraction(sm, gm)

    # Group state
    fni = np.zeros((nmax + 1, nmax + 1), dtype=np.float64)
    for n in range(2, nmax + 1):
        if pn[n] > 0:
            fni[n, 0] = 1.0
            for i in range(n):
                denom = (i + 1) * (mu + w * (1.0 - I))

                # Unnormalized assignment
                fni[n, i + 1] += (
                    ((n - i) * (rho + inf_mat[n, i] + w * I) + i * (mu + w * (1.0 - I)))
                    * fni[n, i]
                    / denom
                )

                if i > 0:
                    fni[n, i + 1] -= (
                        (n - i + 1) * (inf_mat[n, i - 1] + rho + w * I) * fni[n, i - 1]
                        / denom
                    )

            # Normalize
            fni[n] /= np.sum(fni[n])

    return flatten(sm, fni, state_meta)

@jit(nopython=True)
def mf_map_w(r, mu, w, inf_mat, state_meta):
    """
    Adapted from https://github.com/gstonge/gcm

    Evaluate the self-consistent mean-field map at a given value of r.

    Parameters
    ----------
    r : float
        Mean-field infection pressure.
    mu : float
        Recovery rate.
    w : float
        Rewiring rate.
    inf_mat : ndarray
        Infection-rate matrix.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.

    Returns
    -------
    float
        Updated mean-field value obtained from the reconstructed stationary state.
    """
    v = state_from_mf_w(r, mu, w, inf_mat, state_meta)
    mmax = state_meta[0]
    nmax = state_meta[1]
    fni = v[mmax + 1:].reshape((nmax + 1, nmax + 1))
    return mf_from_state(fni, inf_mat, state_meta)


def I_from_r(r, mu, m_arr, gm):
    """
    Compute the stationary fraction of infected nodes from a mean-field value r.

    Parameters
    ----------
    r : float
        Mean-field infection pressure.
    mu : float
        Recovery rate.
    m_arr : ndarray
        Array of membership values.
    gm : ndarray
        Membership distribution.

    Returns
    -------
    float
        Fraction of infected nodes I.
    """
    sm = mu / (mu + m_arr * r)
    return float(np.sum((1.0 - sm) * gm))


def normalize_group_distribution(pn, nmin=2):
    """
    Normalize a group-size distribution after removing small groups.

    Parameters
    ----------
    pn : array_like
        Original group-size distribution.
    nmin : int, optional
        Minimum group size to retain (default is 2).

    Returns
    -------
    ndarray
        Renormalized group-size distribution with pn[n < nmin] = 0.
    """
    pn = np.asarray(pn, dtype=float).copy()
    pn[:nmin] = 0.0

    s = pn.sum()
    if s <= 0:
        raise ValueError("Normalization failed: distribution sums to zero after filtering.")

    return pn / s


def compute_In_from_fni(fni, nmax):
    """
    Compute the average infected fraction within groups of size n.

    Parameters
    ----------
    fni : ndarray
        Group-state distribution, where fni[n, i] is the fraction of groups of size n that contain i infected nodes.
    nmax : int
        Maximum group size.

    Returns
    -------
    ndarray
        Array In where In[n] is the average fraction of infected nodes
        in groups of size n.
    """
    In = np.zeros(nmax + 1, dtype=float)
    for n in range(2, nmax + 1):
        i = np.arange(n + 1, dtype=float)
        In[n] = np.sum((i / n) * fni[n, :n + 1])
    return In


def y4_tilde(In, pn_use):
    """
    Compute the normalized fourth-order moment of group infection levels.

    Parameters
    ----------
    In : ndarray
        Array where In[n] is the average infected fraction in groups of size n.
    pn_use : ndarray
        Group-size distribution used for averaging.

    Returns
    -------
    float
        Normalized fourth-order moment:
            Ỹ₄ = ⟨I_n⁴⟩ / ⟨I_n²⟩²
        Returns NaN if the denominator is zero.
    """
    num = np.sum(pn_use * (In ** 4))
    den = np.sum(pn_use * (In ** 2))

    if den <= 0:
        return np.nan

    return num / (den ** 2)



def load_group_statistics(network, path="Data/group_statistics.txt"):
    """
    Load group-size and membership statistics for a given network.

    The function reads a JSON file containing empirical group statistics and
    returns the corresponding membership distribution, group-size distribution,
    and structural metadata required by the wAMEs framework.

    Parameters
    ----------
    network : str
        Name of the network to load.
    path : str, optional
        Path to the JSON file containing the group statistics.

    Returns
    -------
    tuple
        ``(gm, pn, mmax, nmax, state_meta)``, where

        - ``gm`` is the membership distribution,
        - ``pn`` is the group-size distribution,
        - ``mmax`` is the maximum membership,
        - ``nmax`` is the maximum group size,
        - ``state_meta`` is the structural metadata returned by
          ``get_state_meta``.

    Raises
    ------
    FileNotFoundError
        If the statistics file does not exist.
    KeyError
        If the requested network is not found in the file.
    ValueError
        If the loaded distributions are empty or invalid.
    """
    with open(path, "r") as f:
        results = json.load(f)

    if network not in results:
        available = ", ".join(sorted(results.keys()))
        raise KeyError(
            f"Network '{network}' not found in '{path}'. "
            f"Available networks: {available}"
        )

    data = results[network]

    n_vals = data["group_size_n"]
    p_vals = data["group_size_p"]
    k_vals = data["membership_k"]
    g_vals = data["membership_g"]

    if len(n_vals) == 0 or len(k_vals) == 0:
        raise ValueError(f"Network '{network}' has empty group statistics.")

    nmax = int(max(n_vals))
    mmax = int(max(k_vals))

    pn = np.zeros(nmax + 1, dtype=float)
    gm = np.zeros(mmax + 1, dtype=float)

    for n, p in zip(n_vals, p_vals):
        pn[int(n)] = float(p)

    for k, g in zip(k_vals, g_vals):
        gm[int(k)] = float(g)
        
    pmax = float(np.max(pn[2:]))

    state_meta = get_state_meta(mmax, nmax, gm, pn)

    return gm, pn, mmax, nmax, pmax, state_meta