from .core import *

def _cumulative_infection_product(beta, nmax, args=()):
    """
    Compute cumulative products of the infection rate over the number of
    infected individuals in a group.

    From https://github.com/gstonge/gcm

    Parameters
    ----------
    beta : callable
        Infection-rate function of the form ``beta(n, i, *args)``.
    nmax : int
        Maximum group size.
    args : tuple, optional
        Additional arguments passed to ``beta``.

    Returns
    -------
    ndarray
        Array of shape ``(nmax + 1, nmax + 1)`` such that
        ``mat[n, i] = \prod_{j=1}^{i} beta(n, j, *args)`` for ``n >= 2``.
        By convention, ``mat[n, 0] = 1``.
    """
    mat = np.zeros((nmax + 1, nmax + 1), dtype=float)
    nvec = np.arange(2, nmax + 1)

    mat[2:, 0] = 1.0

    for i in range(1, nmax + 1):
        mat[2:, i] = mat[2:, i - 1] * beta(nvec, i, *args)

    return mat

def _sum_w(beta, w, mu, pn, args=()):
    """
    Compute the group dynamics-dependent sum entering the invasion threshold condition.

    Adapted from https://github.com/gstonge/gcm

    Parameters
    ----------
    beta : callable
        Infection-rate function of the form ``beta(n, i, *args)``.
    w : float
        Group switching rate.
    mu : float
        Recovery rate.
    pn : array_like
        Group-size distribution.
    args : tuple, optional
        Additional arguments passed to ``beta``.

    Returns
    -------
    float
        Value of the weighted sum over group sizes and infected counts that
        appears in the invasion-threshold condition.
    """
    nmax = len(pn) - 1
    cum_prod = _cumulative_infection_product(beta, nmax, args=args)

    mat = np.zeros((nmax + 1, nmax + 1), dtype=float)
    ivec = np.arange(1, nmax + 1)

    for n in range(2, nmax + 1):
        coeff = gamma(n + 1) / (gamma(n - ivec) * gamma(ivec + 1))
        mat[n, 1:] = coeff / (mu + w) ** ivec
        mat[n, 1:] *= cum_prod[n, 1:] * pn[n]
        mat[n, 1:][~np.isfinite(mat[n, 1:])] = 0.0

    return float(np.sum(mat))

def _compute_hl_coefficients(beta, w, mu, nmax, args=()):
    """
    Compute first- and second-order coefficient arrays used in the
    tricritical-point condition.

    These arrays encode the expansion of the group-state probabilities around
    the absorbing state and are used to evaluate the derivatives entering the
    tricritical condition.
    
    Adapted from https://github.com/gstonge/gcm


    Parameters
    ----------
    beta : callable
        Infection-rate function of the form ``beta(n, i, *args)``.
    w : float
        Group switching rate.
    mu : float
        Recovery rate.
    nmax : int
        Maximum group size.
    args : tuple, optional
        Additional arguments passed to ``beta``.

    Returns
    -------
    tuple of ndarray
        ``(h_rho, h_I, l_rho, l_I, l_mix)``, each of shape
        ``(nmax + 1, nmax + 1)``.
    """
    h_I = np.zeros((nmax + 1, nmax + 1), dtype=float)
    h_rho = np.zeros((nmax + 1, nmax + 1), dtype=float)
    l_I = np.zeros((nmax + 1, nmax + 1), dtype=float)
    l_rho = np.zeros((nmax + 1, nmax + 1), dtype=float)
    l_mix = np.zeros((nmax + 1, nmax + 1), dtype=float)

    for n in range(2, nmax + 1):
        # First-order coefficients
        h_rho[n, 1] = n / (mu + w)
        for i in range(1, n):
            h_rho[n, i + 1] = (
                h_rho[n, i] * (n - i) * beta(n, i, *args) / ((i + 1) * (mu + w))
            )

        h_rho[n, 0] = -np.sum(h_rho[n])
        h_I[n] = w * h_rho[n]

        # Second-order coefficients
        l_rho[n, 1] = 2 * n * h_rho[n, 0] / (mu + w)
        l_I[n, 1] = (2 * n * w * h_I[n, 0] + 2 * w * h_I[n, 1]) / (mu + w)
        l_mix[n, 1] = (
            n * h_I[n, 0] + w * n * h_rho[n, 0] + w * h_rho[n, 1]
        ) / (mu + w)

        for i in range(1, n):
            l_rho[n, i + 1] = (
                2 * ((n - i) * h_rho[n, i] - (n - i + 1) * h_rho[n, i - 1])
                + (i * (mu + w) + (n - i) * beta(n, i, *args)) * l_rho[n, i]
                - (n - i + 1) * beta(n, i - 1, *args) * l_rho[n, i - 1]
            ) / ((i + 1) * (mu + w))

            l_I[n, i + 1] = (
                2 * w * ((n - 2 * i) * h_I[n, i] - (n - i + 1) * h_I[n, i - 1])
                + (i * (mu + w) + (n - i) * beta(n, i, *args)) * l_I[n, i]
                - (n - i + 1) * beta(n, i - 1, *args) * l_I[n, i - 1]
                + 2 * w * (i + 1) * h_I[n, i + 1]
            ) / ((i + 1) * (mu + w))

            l_mix[n, i + 1] = (
                (n - i) * h_I[n, i]
                + w * (n - 2 * i) * h_rho[n, i]
                - (n - i + 1) * (h_I[n, i - 1] + w * h_rho[n, i - 1])
                + (i * (mu + w) + (n - i) * beta(n, i, *args)) * l_mix[n, i]
                - (n - i + 1) * beta(n, i - 1, *args) * l_mix[n, i - 1]
                + w * (i + 1) * h_rho[n, i + 1]
            ) / ((i + 1) * (mu + w))

        l_rho[n, 0] = -np.sum(l_rho[n])
        l_I[n, 0] = -np.sum(l_I[n])
        l_mix[n, 0] = -np.sum(l_mix[n])

    return h_rho, h_I, l_rho, l_I, l_mix



def invasion_threshold_w(
    beta,
    w,
    mu,
    gm,
    pn,
    fixed_args=(),
    param_min=1e-14,
    param_max=10.0,
):
    """
    Compute the invasion threshold in the presence of group dynamics.

    The threshold is obtained as the root of the invasion condition
    associated with the linear stability of the absorbing state. The root is
    computed with Brent's method over the interval
    ``[param_min, param_max]``.
    
    Adapted from https://github.com/gstonge/gcm

    Parameters
    ----------
    beta : callable
        Infection-rate function of the form ``beta(n, i, *args)``.
    w : float
        Group switching rate.
    mu : float
        Recovery rate.
    gm : array_like
        Membership distribution.
    pn : array_like
        Group-size distribution.
    fixed_args : tuple, optional
        Additional parameters passed to ``beta`` after the threshold
        parameter. For example, if ``beta`` depends on ``(lam, nu)``, then
        ``fixed_args`` may contain ``(nu,)``.
    param_min : float, optional
        Lower bound of the search interval for the threshold parameter.
    param_max : float, optional
        Upper bound of the search interval for the threshold parameter.

    Returns
    -------
    float
        Invasion threshold. Returns ``np.inf`` if no root is found within the
        specified interval.
    """
    nmax = len(pn) - 1
    mmax = len(gm) - 1

    n = np.arange(nmax + 1, dtype=float)
    m = np.arange(mmax + 1, dtype=float)

    mean_group_size = np.sum(n * pn)
    mean_membership = np.sum(m * gm)

    const = (
        np.sum(m * (m - 1) * gm) / (mean_membership * mean_group_size)
        + (w / mu) * (mean_membership / mean_group_size)
    )

    def threshold_function(param):
        return const * _sum_w(beta, w, mu, pn, args=(param, *fixed_args)) - 1.0

    try:
        threshold = brentq(threshold_function, param_min, param_max)
    except ValueError:
        threshold = np.inf

    return float(threshold)

def tricritical_condition(
    nu,
    w,
    beta,
    mu,
    gm,
    pn,
    param_min=1e-14,
    param_max=30.0,
):
    """
    Evaluate the tricritical condition at fixed synergy and group dynamics.

    This function computes the quantity
    ``F(nu, w) = d²M/dr² |_{r=0, lambda=lambda_c}``,
    where ``M`` is the self-consistent message map and ``lambda_c`` is the
    invasion threshold for the given values of ``nu`` and ``w``. A zero of
    ``F`` identifies a tricritical point.

    Parameters
    ----------
    nu : float
        Synergy exponent.
    w : float
        Group switching rate.
    beta : callable
        Infection-rate function of the form ``beta(n, i, *args)``.
    mu : float
        Recovery rate.
    gm : array_like
        Membership distribution.
    pn : array_like
        Group-size distribution.
    param_min : float, optional
        Lower bound of the interval used to compute the invasion threshold.
    param_max : float, optional
        Upper bound of the interval used to compute the invasion threshold.

    Returns
    -------
    float
        Value of the tricritical condition. Returns ``np.nan`` if the invasion
        threshold cannot be determined.
    """
    nmax = len(pn) - 1
    mmax = len(gm) - 1

    n = np.arange(nmax + 1, dtype=float)
    m = np.arange(mmax + 1, dtype=float)

    mean_membership = np.sum(m * gm)
    mean_group_size = np.sum(n * pn)

    if mu <= 0:
        raise ValueError("mu must be positive.")
    if mean_membership <= 0:
        raise ValueError("The mean membership must be positive.")
    if mean_group_size <= 0:
        raise ValueError("The mean group size must be positive.")

    # Derivatives of rho(r) and I(r) at r = 0
    d_rho_dr = np.sum(m * (m - 1) * gm) / mean_membership
    d_I_dr = mean_membership / mu

    d2_rho_dr2 = 2.0 * (
        np.sum(m**2 * gm) ** 2 / mean_membership**2
        - np.sum(m**3 * gm) / mean_membership
    )
    d2_I_dr2 = -2.0 * np.sum(m**2 * gm) / mu**2

    # Invasion threshold lambda_c(w, nu)
    lam_c = invasion_threshold_w(
        beta,
        w,
        mu,
        gm,
        pn,
        fixed_args=(nu,),
        param_min=param_min,
        param_max=param_max,
    )
    if not np.isfinite(lam_c):
        return np.nan

    args = (lam_c, nu)

    # Expansion coefficients entering the derivatives of M
    h_rho, h_I, l_rho, l_I, l_mix = _compute_hl_coefficients(
        beta, w, mu, nmax, args=args
    )

    weight_rho = np.zeros((nmax + 1, nmax + 1), dtype=float)
    weight_beta = np.zeros((nmax + 1, nmax + 1), dtype=float)

    for n_ in range(2, nmax + 1):
        for i in range(n_ + 1):
            weight_rho[n_, i] = (n_ - i) * pn[n_]
            weight_beta[n_, i] = beta(n_, i, *args) * (n_ - i) * pn[n_]

    v_rho = np.sum(weight_rho * h_rho)
    v_I = np.sum(weight_rho * h_I)

    u_rho = np.sum(weight_beta * h_rho)
    u_I = np.sum(weight_beta * h_I)
    u_rho_rho = np.sum(weight_beta * l_rho)
    u_I_I = np.sum(weight_beta * l_I)
    u_mix = np.sum(weight_beta * l_mix)

    dM_drho = u_rho / mean_group_size
    dM_dI = u_I / mean_group_size

    d2M_drho2 = u_rho_rho / mean_group_size - 2.0 * v_rho * u_rho / mean_group_size**2
    d2M_dI2 = u_I_I / mean_group_size - 2.0 * v_I * u_I / mean_group_size**2
    d2M_drho_dI = (
        u_mix / mean_group_size
        - (v_rho * u_I + v_I * u_rho) / mean_group_size**2
    )

    F = (
        d2M_drho2 * d_rho_dr**2
        + d2M_dI2 * d_I_dr**2
        + 2.0 * d2M_drho_dI * d_rho_dr * d_I_dr
        + dM_drho * d2_rho_dr2
        + dM_dI * d2_I_dr2
    )

    return float(F)



def find_tricritical_points_joint(
    tricritical_condition,
    invasion_threshold_func,
    beta_func,
    mu,
    gm,
    pn,
    lambda_min,
    lambda_max,
    w_min,
    w_max,
    nu_min,
    nu_max,
    nu_inner_min,
    nu_inner_max,
    n_w_outer=101,
    n_nu_outer=41,
    n_w_inner=400,
    n_nu_inner=400,
    rtol_dup=1e-4,
    atol_dup=1e-6,
    verbose=False,
):
    """
    Find tricritical points by combining scans in the ``(nu, w)`` plane.

    The search is performed in two complementary ways:
    (i) fixing ``nu`` and scanning in ``w``, and
    (ii) fixing ``w`` and scanning in ``nu``.
    The resulting candidate tricritical points are then merged,
    deduplicated, and sorted.

    Parameters
    ----------
    tricritical_condition : callable
        Function returning the tricritical condition ``F(nu, w)``.
    invasion_threshold_func : callable
        Function used to compute the invasion threshold at each candidate
        tricritical point.
    beta_func : callable
        Infection-rate function of the form ``beta(n, i, *args)``.
    mu : float
        Recovery rate.
    gm : array_like
        Membership distribution.
    pn : array_like
        Group-size distribution.
    lambda_min : float
        Lower bound used in the invasion-threshold search.
    lambda_max : float
        Upper bound used in the invasion-threshold search.
    w_min : float
        Minimum rewiring rate in the scan.
    w_max : float
        Maximum rewiring rate in the scan.
    nu_min : float
        Minimum value of the outer ``nu`` scan.
    nu_max : float
        Maximum value of the outer ``nu`` scan.
    nu_inner_min : float
        Minimum value of the inner ``nu`` scan.
    nu_inner_max : float
        Maximum value of the inner ``nu`` scan.
    n_w_outer : int, optional
        Number of points in the outer ``w`` grid.
    n_nu_outer : int, optional
        Number of points in the outer ``nu`` grid.
    n_w_inner : int, optional
        Number of points in the inner ``w`` grid used for root bracketing.
    n_nu_inner : int, optional
        Number of points in the inner ``nu`` grid used for root bracketing.
    rtol_dup : float, optional
        Relative tolerance used to merge duplicate tricritical points.
    atol_dup : float, optional
        Absolute tolerance used to merge duplicate tricritical points.
    verbose : bool, optional
        If True, print progress information during the scan.

    Returns
    -------
    tuple of ndarray
        Arrays ``(nu_vals, w_vals, lambda_vals)`` containing the merged
        tricritical points, sorted by ``nu``.
    """
    if w_min <= 0:
        raise ValueError("w_min must be positive because a logarithmic grid is used.")
    if w_max <= w_min:
        raise ValueError("w_max must be greater than w_min.")
    if nu_max <= nu_min:
        raise ValueError("nu_max must be greater than nu_min.")
    if nu_inner_max <= nu_inner_min:
        raise ValueError("nu_inner_max must be greater than nu_inner_min.")

    w_outer = np.logspace(np.log10(w_min), np.log10(w_max), n_w_outer)
    nu_outer = np.linspace(nu_min, nu_max, n_nu_outer)

    w_inner = np.logspace(np.log10(w_min), np.log10(w_max), n_w_inner)
    nu_inner = np.linspace(nu_inner_min, nu_inner_max, n_nu_inner)

    tricritical_nu_A = []
    tricritical_w_A = []
    tricritical_lambda_A = []

    tricritical_nu_B = []
    tricritical_w_B = []
    tricritical_lambda_B = []

    # Scan A: fix nu, scan w
    if verbose:
        print("=== Scan A: fixing nu, scanning w ===")

    for nu in nu_outer:
        if verbose:
            print(f"[A] nu = {nu:.6f}")

        def F_of_w(w):
            return tricritical_condition(
                nu,
                w,
                beta_func,
                mu,
                gm,
                pn,
                param_min=lambda_min,
                param_max=lambda_max,
            )

        F_vals = np.array([F_of_w(w) for w in w_inner], dtype=float)

        roots_w = []
        roots_lambda = []

        for i in range(len(w_inner) - 1):
            F1 = F_vals[i]
            F2 = F_vals[i + 1]

            if not (np.isfinite(F1) and np.isfinite(F2)):
                continue

            w_root = None

            if F1 == 0:
                w_root = w_inner[i]
            elif F2 == 0:
                w_root = w_inner[i + 1]
            elif F1 * F2 < 0:
                try:
                    w_root = brentq(F_of_w, w_inner[i], w_inner[i + 1])
                except ValueError:
                    pass

            if w_root is None:
                continue

            if np.any(np.isclose(w_root, roots_w, rtol=1e-6, atol=1e-9)):
                continue

            lambda_root = invasion_threshold_func(
                beta_func,
                w_root,
                mu,
                gm,
                pn,
                fixed_args=(nu,),
                param_min=lambda_min,
                param_max=lambda_max,
            )

            if not np.isfinite(lambda_root):
                continue

            roots_w.append(w_root)
            roots_lambda.append(lambda_root)

        if len(roots_w) == 0:
            continue

        order = np.argsort(roots_w)
        roots_w = np.asarray(roots_w, dtype=float)[order]
        roots_lambda = np.asarray(roots_lambda, dtype=float)[order]

        tricritical_nu_A.extend([nu] * len(roots_w))
        tricritical_w_A.extend(roots_w)
        tricritical_lambda_A.extend(roots_lambda)

    # Scan B: fix w, scan nu
    if verbose:
        print("=== Scan B: fixing w, scanning nu ===")

    for w in w_outer:
        if verbose:
            print(f"[B] w = {w:.6e}")

        def F_of_nu(nu):
            return tricritical_condition(
                nu,
                w,
                beta_func,
                mu,
                gm,
                pn,
                param_min=lambda_min,
                param_max=lambda_max,
            )

        F_vals = np.array([F_of_nu(nu) for nu in nu_inner], dtype=float)

        roots_nu = []
        roots_lambda = []

        for i in range(len(nu_inner) - 1):
            F1 = F_vals[i]
            F2 = F_vals[i + 1]

            if not (np.isfinite(F1) and np.isfinite(F2)):
                continue

            nu_root = None

            if F1 == 0:
                nu_root = nu_inner[i]
            elif F2 == 0:
                nu_root = nu_inner[i + 1]
            elif F1 * F2 < 0:
                try:
                    nu_root = brentq(F_of_nu, nu_inner[i], nu_inner[i + 1])
                except ValueError:
                    pass

            if nu_root is None:
                continue

            if np.any(np.isclose(nu_root, roots_nu, rtol=1e-6, atol=1e-9)):
                continue

            lambda_root = invasion_threshold_func(
                beta_func,
                w,
                mu,
                gm,
                pn,
                fixed_args=(nu_root,),
                param_min=lambda_min,
                param_max=lambda_max,
            )

            if not np.isfinite(lambda_root):
                continue

            roots_nu.append(nu_root)
            roots_lambda.append(lambda_root)

        if len(roots_nu) == 0:
            continue

        order = np.argsort(roots_nu)
        roots_nu = np.asarray(roots_nu, dtype=float)[order]
        roots_lambda = np.asarray(roots_lambda, dtype=float)[order]

        tricritical_w_B.extend([w] * len(roots_nu))
        tricritical_nu_B.extend(roots_nu)
        tricritical_lambda_B.extend(roots_lambda)

    all_nu = np.concatenate([tricritical_nu_A, tricritical_nu_B])
    all_w = np.concatenate([tricritical_w_A, tricritical_w_B])
    all_lambda = np.concatenate([tricritical_lambda_A, tricritical_lambda_B])

    if len(all_nu) == 0:
        return np.array([]), np.array([]), np.array([])

    points = np.column_stack((all_nu, all_w, all_lambda))

    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    merged = [points[0]]
    for point in points[1:]:
        prev = merged[-1]
        same_nu = np.isclose(point[0], prev[0], rtol=rtol_dup, atol=atol_dup)
        same_w = np.isclose(point[1], prev[1], rtol=rtol_dup, atol=atol_dup)
        if not (same_nu and same_w):
            merged.append(point)

    merged = np.asarray(merged, dtype=float)

    if verbose:
        print(f"Total unique tricritical points: {len(merged)}")

    return merged[:, 0], merged[:, 1], merged[:, 2]

