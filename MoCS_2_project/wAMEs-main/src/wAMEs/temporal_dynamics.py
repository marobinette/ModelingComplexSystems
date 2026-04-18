from .core import *

def initialize(state_meta, initial_density=0.5):
    """
    Initialize the node and group states from a homogeneous infected fraction.

    From https://github.com/gstonge/gcm

    The initial condition assumes that each node is infected independently
    with probability ``initial_density``. Under this assumption, the
    distribution of infected nodes within groups of size ``n`` is binomial.

    Parameters
    ----------
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.
    initial_density : float, optional
        Initial fraction of infected nodes.

    Returns
    -------
    tuple of ndarray
        ``(sm, fni)``, where ``sm[m]`` is the probability that a node of
        membership ``m`` is susceptible, and ``fni[n, i]`` is the fraction
        of groups of size ``n`` containing ``i`` infected nodes.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]

    sm = np.full(mmax + 1, 1.0 - initial_density, dtype=float)
    fni = np.zeros((nmax + 1, nmax + 1), dtype=float)

    for n in range(2, nmax + 1):
        infected_counts = np.arange(n + 1, dtype=int)
        fni[n, :n + 1] = binom.pmf(infected_counts, n, initial_density)

    return sm, fni




@jit(nopython=True)
def vector_field_w(v, t, inf_mat, state_meta, mu, w):
    """
    Evaluate the dynamical vector field in the presence of rewiring.

    Adapted from https://github.com/gstonge/gcm

    Parameters
    ----------
    v : ndarray
        Flattened state vector containing both node and group states, as
        produced by ``flatten``.
    t : float
        Time variable (unused, included for compatibility with ODE solvers).
    inf_mat : ndarray
        Infection-rate matrix of shape ``(nmax + 1, nmax + 1)``.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.
    mu : float
        Recovery rate.
    w : float
        Group switching rate.

    Returns
    -------
    ndarray
        Time derivative of the flattened state vector.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    m = state_meta[2]
    gm = state_meta[3]
    pn = state_meta[4]
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]

    sm = v[:mmax + 1]
    fni = v[mmax + 1:].reshape((nmax + 1, nmax + 1))

    sm_field = np.zeros(sm.shape)
    fni_field = np.zeros(fni.shape)

    # Mean-field quantities
    r = np.sum(
        inf_mat[2:, :] * (nmat[2:, :] - imat[2:, :]) * fni[2:, :] * pnmat[2:, :]
    )
    r /= np.sum((nmat[2:, :] - imat[2:, :]) * fni[2:, :] * pnmat[2:, :])

    # equation 10
    rho = r * excess_susceptible_membership(m, gm, sm)
    I = infected_fraction(sm, gm)

    # Node dynamics
    # equation 8 
    # describes the change in the fraction of susceptible individuals with k group memberships
    # mapping to the terms in the equation from paper:
    # mu = mu, sk = sm (susceptible probability per membership class), k = m (membership), r = r (The mean-field infection rate)
    sm_field = mu * (1.0 - sm) - sm * m * r

    # Group dynamics
    # Equation 9
    # Contribution from i + 1 -> i
    fni_field[2:, :nmax] += imat[2:, 1:] * (mu + w * (1.0 - I)) * fni[2:, 1:]

    # Contribution from i -> i
    fni_field[2:, :] += (
        -imat[2:, :] * (mu + w * (1.0 - I))
        - (nmat[2:, :] - imat[2:, :]) * (inf_mat[2:, :] + rho + w * I)
    ) * fni[2:, :]

    # Contribution from i - 1 -> i
    fni_field[2:, 1:nmax + 1] += (
        (nmat[2:, :nmax] - imat[2:, :nmax])
        * (inf_mat[2:, :nmax] + rho + w * I)
        * fni[2:, :nmax]
    )

    return np.concatenate((sm_field, fni_field.reshape((nmax + 1) ** 2)))


@jit(nopython=True)
def vector_field_w_kernel(v, t, inf_mat, w_mat, state_meta, mu):
    """
    Evaluate the dynamical vector field with a group-state-dependent switching
    rate kernel.

    Drop-in replacement for ``vector_field_w`` where the scalar ``w`` is
    replaced by a precomputed matrix ``w_mat[n, i]`` built via
    ``switching_matrix``.  This allows the group switching rate to depend on
    group size ``n`` and the number of infected members ``i``, enabling
    modelling of diversity-seeking or allegiance-driven group dynamics.

    Parameters
    ----------
    v : ndarray
        Flattened state vector containing both node and group states, as
        produced by ``flatten``.
    t : float
        Time variable (unused, included for compatibility with ODE solvers).
    inf_mat : ndarray
        Infection-rate matrix of shape ``(nmax + 1, nmax + 1)``.
    w_mat : ndarray
        Switching-rate matrix of shape ``(nmax + 1, nmax + 1)``, where
        ``w_mat[n, i]`` is the rate at which a node in a group of size ``n``
        with ``i`` infected members switches to a new group.  Build this with
        ``switching_matrix``.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.
    mu : float
        Recovery rate.

    Returns
    -------
    ndarray
        Time derivative of the flattened state vector.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    m = state_meta[2]
    gm = state_meta[3]
    pn = state_meta[4]
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]

    sm = v[:mmax + 1]
    fni = v[mmax + 1:].reshape((nmax + 1, nmax + 1))

    sm_field = np.zeros(sm.shape)
    fni_field = np.zeros(fni.shape)

    # Mean-field quantities
    r = np.sum(
        inf_mat[2:, :] * (nmat[2:, :] - imat[2:, :]) * fni[2:, :] * pnmat[2:, :]
    )
    r /= np.sum((nmat[2:, :] - imat[2:, :]) * fni[2:, :] * pnmat[2:, :])

    rho = r * excess_susceptible_membership(m, gm, sm)
    I = infected_fraction(sm, gm)

    # need to be more defensive here:
    # if denom is > 0 there are mixed groups contributing to the switching flux,
    # so LHD's weighted formula is valid and gives the correct bias in who gets swapped.
    S_w_denom = np.sum(nmat[2:, :] * w_mat[2:, :] * fni[2:, :] * pnmat[2:, :])
    if S_w_denom > 1e-14:
        S_w = (
            np.sum((nmat[2:, :] - imat[2:, :]) * w_mat[2:, :] * fni[2:, :] * pnmat[2:, :])
            / S_w_denom
        )
    else:
        S_w = 1.0 - I  # fallback to the original mean-field switching term if the kernel is degenerate
        print('hitting else code')
    sm_field = mu * (1.0 - sm) - sm * m * r

    # i+1 -> i
    fni_field[2:, :nmax] += (
        imat[2:, 1:]
        * (mu + w_mat[2:, 1:] * S_w)          # was (1 - S_w)
        * fni[2:, 1:]
    )

    # diagonal outflow
    fni_field[2:, :] += (
        -imat[2:, :] * (mu + w_mat[2:, :] * S_w)          # was (1 - S_w)
        - (nmat[2:, :] - imat[2:, :]) * (inf_mat[2:, :] + rho + w_mat[2:, :] * (1.0 - S_w))  # was S_w
    ) * fni[2:, :]

    # i-1 -> i
    fni_field[2:, 1:nmax + 1] += (
        (nmat[2:, :nmax] - imat[2:, :nmax])
        * (inf_mat[2:, :nmax] + rho + w_mat[2:, :nmax] * (1.0 - S_w))  # was S_w
        * fni[2:, :nmax]
    )
    return np.concatenate((sm_field, fni_field.reshape((nmax + 1) ** 2)))


def integrate_I_traj(
    lam,
    state_meta,
    nmax,
    mmax,
    gm,
    mu,
    w,
    nu,
    I0=1e-5,
    traj_points=200000,
    t_max=None,
):
    """
    Integrate the time evolution of the infected fraction from a prescribed
    initial density.

    The dynamics are solved using the AME system with group dynamics, starting from
    a homogeneous initial condition in which each node is infected
    independently with probability ``I0``.

    Parameters
    ----------
    lam : float
        Transmission-rate prefactor.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.
    nmax : int
        Maximum group size.
    mmax : int
        Maximum membership.
    gm : ndarray
        Membership distribution.
    mu : float
        Recovery rate.
    w : float
        Group switching rate.
    nu : float
        Synergy exponent in the infection rate
        ``beta(n, i) = lam * i**nu``.
    I0 : float, optional
        Initial infected fraction.
    traj_points : int, optional
        Number of time points stored along the trajectory.
    t_max : float, optional
        Final integration time. If ``None``, the final time is set equal to
        ``traj_points``.

    Returns
    -------
    tuple of ndarray
        ``(t, I)``, where ``t`` is the integration time grid and ``I`` is the
        corresponding trajectory of the infected fraction.
    """
    if not (0.0 <= I0 <= 1.0):
        raise ValueError("I0 must be between 0 and 1.")
    if traj_points < 2:
        raise ValueError("traj_points must be at least 2.")

    inf_mat = infection_matrix(
        lambda n, i: lam * i**nu,
        nmax,
    )

    sm, fni = initialize(state_meta, initial_density=I0)
    v0 = np.concatenate((sm, fni.reshape((nmax + 1) ** 2)))

    if t_max is None:
        t_max = float(traj_points)

    t = np.linspace(0.0, t_max, traj_points)
    t_span = (t[0], t[-1])

    sol = solve_ivp(
        lambda time, state: vector_field_w(state, time, inf_mat, state_meta, mu, w),
        t_span=t_span,
        y0=v0,
        method="LSODA",
        t_eval=t,
    )

    if not sol.success:
        raise RuntimeError(f"Trajectory integration failed: {sol.message}")

    v_traj = sol.y.T
    I_traj = np.array(
        [infected_fraction(v[:mmax + 1], gm) for v in v_traj],
        dtype=float,
    )

    return t, I_traj


def integrate_I_traj_kernel(
    lam,
    w_func,
    state_meta,
    nmax,
    mmax,
    gm,
    mu,
    nu,
    w_args=(),
    I0=1e-5,
    traj_points=200000,
    t_max=None,
):
    """
    Integrate the infected-fraction trajectory using a group-state-dependent
    switching-rate kernel.

    Mirrors ``integrate_I_traj`` but replaces the scalar ``w`` with a
    callable ``w_func(n, i, *w_args)`` that can encode diversity-seeking,
    allegiance-driven, or any other (n, i)-dependent group switching
    behaviour.  Internally the kernel is evaluated once into a precomputed
    matrix and passed to ``vector_field_w_kernel``.

    Parameters
    ----------
    lam : float
        Transmission-rate prefactor.
    w_func : callable
        Switching-rate function ``w_func(n, i, *w_args)``.  Returns the
        rate at which a node in a group of size ``n`` with ``i`` infected
        members switches to a new group.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.
    nmax : int
        Maximum group size.
    mmax : int
        Maximum membership.
    gm : ndarray
        Membership distribution.
    mu : float
        Recovery rate.
    nu : float
        Synergy exponent in the infection rate ``beta(n, i) = lam * i**nu``.
    w_args : tuple, optional
        Extra arguments forwarded to ``w_func``.
    I0 : float, optional
        Initial infected fraction.
    traj_points : int, optional
        Number of time points stored along the trajectory.
    t_max : float, optional
        Final integration time.  If ``None``, defaults to ``traj_points``.

    Returns
    -------
    tuple of ndarray
        ``(t, I)``, where ``t`` is the integration time grid and ``I`` is the
        corresponding trajectory of the infected fraction.
    """
    if not (0.0 <= I0 <= 1.0):
        raise ValueError("I0 must be between 0 and 1.")
    if traj_points < 2:
        raise ValueError("traj_points must be at least 2.")

    inf_mat = infection_matrix(
        lambda n, i: lam * i**nu,
        nmax,
    )
    w_mat = switching_matrix(w_func, nmax, args=w_args)

    sm, fni = initialize(state_meta, initial_density=I0)
    v0 = np.concatenate((sm, fni.reshape((nmax + 1) ** 2)))

    if t_max is None:
        t_max = float(traj_points)

    t = np.linspace(0.0, t_max, traj_points)
    t_span = (t[0], t[-1])

    sol = solve_ivp(
        lambda time, state: vector_field_w_kernel(state, time, inf_mat, w_mat, state_meta, mu),
        t_span=t_span,
        y0=v0,
        method="LSODA",
        t_eval=t,
    )

    if not sol.success:
        raise RuntimeError(f"Trajectory integration failed: {sol.message}")

    v_traj = sol.y.T
    I_traj = np.array(
        [infected_fraction(v[:mmax + 1], gm) for v in v_traj],
        dtype=float,
    )

    return t, I_traj