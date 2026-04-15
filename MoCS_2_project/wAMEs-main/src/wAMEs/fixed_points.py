from .core import *

def find_fixed_points_for_lambda(
    inf_mat,
    mu,
    w,
    state_meta,
    r_min,
    r_max,
    n_grid=4000,
    xtol=1e-12,
):
    """
    Find the fixed points of the self-consistent mean-field map M(r).

    The function searches for roots of G(r) = M(r) - r on a logarithmic grid,
    refines them using Brent's method, removes near-duplicate solutions, and
    classifies their local stability from the numerical slope M'(r*).

    Parameters
    ----------
    inf_mat : ndarray
        Infection-rate matrix.
    mu : float
        Recovery rate.
    w : float
        Rewiring rate.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.
    r_min : float
        Minimum value of r used in the root search.
    r_max : float
        Maximum value of r used in the root search.
    n_grid : int, optional
        Number of logarithmically spaced points used to scan for sign changes.
    xtol : float, optional
        Absolute tolerance passed to ``scipy.optimize.brentq``.

    Returns
    -------
    list of tuple
        List of fixed points. Each entry is a tuple
        ``(r_star, slope, is_stable)``, where ``r_star`` is the fixed point,
        ``slope`` is the numerical estimate of M'(r_star), and ``is_stable``
        is ``True`` when ``slope < 1``.
    """
    def M_of_r(r):
        return mf_map_w(r, mu, w, inf_mat, state_meta)

    def G_of_r(r):
        return M_of_r(r) - r

    r_grid = np.geomspace(r_min, r_max, n_grid)
    M_vals = np.array([M_of_r(r) for r in r_grid])
    G_vals = M_vals - r_grid

    roots = []
    for a, b, Ga, Gb in zip(r_grid[:-1], r_grid[1:], G_vals[:-1], G_vals[1:]):
        if not (np.isfinite(Ga) and np.isfinite(Gb)):
            continue
        if abs(Ga) < 1e-10:
            roots.append(a)
            continue
        if Ga * Gb < 0:
            try:
                r_star = brentq(G_of_r, a, b, xtol=xtol, rtol=1e-12, maxiter=200)
                roots.append(r_star)
            except ValueError:
                pass

    if not roots:
        return []

    roots = np.array(sorted(roots))

    # Remove near-duplicate roots
    dedup = [roots[0]]
    for x in roots[1:]:
        if abs(x - dedup[-1]) > 1e-6 * max(1.0, abs(dedup[-1])):
            dedup.append(x)

    def numerical_Mprime(r_star, eps_rel=1e-5):
        eps = eps_rel * max(1.0, r_star)
        a = max(r_min, r_star - eps)
        b = min(r_max, r_star + eps)
        if b <= a * (1 + 1e-15):
            b = r_star + eps
            a = max(r_min, r_star - eps)
        return (M_of_r(b) - M_of_r(a)) / (b - a)

    fps = []
    for r_star in dedup:
        slope = numerical_Mprime(r_star)
        fps.append((float(r_star), float(slope), bool(slope < 1.0)))

    return fps


def collect_fixed_points_by_lam(
    lam_vals,
    nu,
    nmax,
    mu,
    w,
    state_meta,
    m_arr,
    gm,
    pn_filtered,
    r_min,
    r_max,
    n_grid_root,
):
    """
    Compute fixed-point branches as a function of the transmission rate.

    For each value of lam, this function computes the fixed points of the
    self-consistent mean-field map, reconstructs the corresponding stationary
    states, evaluates the observables I and P = 1 / Y4, and sorts the resulting
    branches by decreasing prevalence.

    Parameters
    ----------
    lam_vals : array_like
        Array of transmission-rate values.
    nu : float
        Synergy exponent.
    nmax : int
        Maximum group size.
    mu : float
        Recovery rate.
    w : float
        Rewiring rate.
    state_meta : tuple
        Structural metadata returned by ``get_state_meta``.
    m_arr : ndarray
        Array of membership values.
    gm : ndarray
        Membership distribution.
    pn_filtered : ndarray
        Group-size distribution used in the computation of Y4.
    r_min : float
        Minimum value of r used in the fixed-point search.
    r_max : float
        Maximum value of r used in the fixed-point search.
    n_grid_root : int
        Number of grid points used to scan for fixed points.

    Returns
    -------
    list of list of dict
        One list per lam value. Each element is a dictionary with keys
        ``lam``, ``I``, ``P``, and ``stable``.
    """
    out = []

    for lam in lam_vals:
        inf_mat = build_inf_mat(lam, nu, nmax)

        fps = find_fixed_points_for_lambda(
            inf_mat=inf_mat,
            mu=mu,
            w=w,
            state_meta=state_meta,
            r_min=r_min,
            r_max=r_max,
            n_grid=n_grid_root,
        )

        pts = []
        for r_star, slope, is_stable in fps:
            v = state_from_mf_w(r_star, mu, w, inf_mat, state_meta)
            sm, fni = unflatten(v, state_meta)

            I_star = I_from_r(r_star, mu, m_arr, gm)

            In_star = compute_In_from_fni(fni, nmax)
            Y4 = float(y4_tilde(In_star, pn_filtered))
            P = 1.0 / Y4 if (np.isfinite(Y4) and Y4 > 0) else np.nan

            pts.append(
                {
                    "lam": float(lam),
                    "I": float(I_star),
                    "P": float(P),
                    "stable": bool(is_stable),
                }
            )

        pts.sort(key=lambda d: d["I"], reverse=True)
        out.append(pts)

    return out


def pack_per_lam_pts(per_lam_pts, lam_vals):
    """
    Pack fixed-point data into dense arrays for plotting.

    Parameters
    ----------
    per_lam_pts : list of list of dict
        Output of ``collect_fixed_points_by_lam``.
    lam_vals : array_like
        Array of transmission-rate values.

    Returns
    -------
    tuple of ndarray
        I : array of shape (Kmax, N)
            Infection fraction for each branch and lam.
        P : array of shape (Kmax, N)
            Observable P (inverse fourth-order moment).
        stable : array of shape (Kmax, N)
            Stability indicator (1 = stable, 0 = unstable, -1 = missing).
    """
    N = len(lam_vals)
    Kmax = max((len(pts) for pts in per_lam_pts), default=0)

    I = np.full((Kmax, N), np.nan, dtype=float)
    P = np.full((Kmax, N), np.nan, dtype=float)
    stable = np.full((Kmax, N), -1, dtype=np.int8)

    for j, pts in enumerate(per_lam_pts):
        for k, p in enumerate(pts):
            I[k, j] = p["I"]
            P[k, j] = p["P"]
            stable[k, j] = 1 if p["stable"] else 0

    return I, P, stable


def unpack_per_lam_pts(lam_vals, I, P, stable):
    """
    Convert dense fixed-point arrays into a list-of-dictionaries structure.

    This function is the inverse of ``pack_per_lam_pts``. It reconstructs
    a per-parameter list of fixed points from dense arrays, which is more
    convenient for plotting and branch tracking.

    Parameters
    ----------
    lam_vals : array_like
        Array of transmission-rate values of length N.
    I : ndarray
        Infection fraction array of shape (Kmax, N).
    P : ndarray
        Observable array (e.g., inverse fourth-order moment) of shape (Kmax, N).
    stable : ndarray
        Stability array of shape (Kmax, N), where
        1 = stable, 0 = unstable, -1 = missing.

    Returns
    -------
    list of list of dict
        One list per lam value. Each element is a dictionary with keys
        ``lam``, ``I``, ``P``, and ``stable``.
    """
    Kmax, N = I.shape
    per_lam_pts = []

    for j in range(N):
        pts = []
        for k in range(Kmax):
            if stable[k, j] == -1:
                continue

            pts.append(
                {
                    "lam": float(lam_vals[j]),
                    "I": float(I[k, j]),
                    "P": float(P[k, j]),
                    "stable": bool(stable[k, j] == 1),
                }
            )
        per_lam_pts.append(pts)

    return per_lam_pts


def add_break_marks(ax_top, ax_bot, size=0.015, lw=1.1, color="0.55"):
    """
    Draw diagonal break marks between two vertically stacked axes.

    This helper is intended for broken-axis plots, where ``ax_top`` and
    ``ax_bot`` represent the upper and lower panels of the same quantity.
    The function adds diagonal marks at the left and right edges to indicate
    the discontinuity between both y-axis ranges.

    Parameters
    ----------
    ax_top : matplotlib.axes.Axes
        Upper axis of the broken-axis pair.
    ax_bot : matplotlib.axes.Axes
        Lower axis of the broken-axis pair.
    size : float, optional
        Half-length of each diagonal mark in axes coordinates.
    lw : float, optional
        Line width of the break marks.
    color : str, optional
        Color of the break marks.
    """
    d = size

    top_kwargs = dict(
        transform=ax_top.transAxes,
        color=color,
        clip_on=False,
        linewidth=lw,
    )
    ax_top.plot((-d, +d), (-d, +d), **top_kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **top_kwargs)

    bottom_kwargs = dict(
        transform=ax_bot.transAxes,
        color=color,
        clip_on=False,
        linewidth=lw,
    )
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **bottom_kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **bottom_kwargs)


def format_x_axis_times10(ax, color="0.55", fontsize=8, xpos=0.90, ypos=-0.43):
    """
    Format a logarithmic x-axis as mantissa × 10^k.

    This function rescales tick labels so that the axis is displayed in terms
    of a common power of ten, i.e., x = m × 10^k, where m is shown on the axis
    and the factor ×10^k is displayed as an annotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to format (must have log-scaled x-axis).
    color : str, optional
        Color of tick labels and exponent annotation.
    fontsize : int, optional
        Font size of the exponent annotation.
    xpos : float, optional
        X position of the exponent annotation in axes coordinates.
    ypos : float, optional
        Y position of the exponent annotation in axes coordinates.
    """
    xmin, xmax = ax.get_xlim()

    if xmin <= 0:
        raise ValueError("Log-scale axis must have strictly positive limits.")

    # Extract common exponent from lower bound
    exponent = int(np.floor(np.log10(xmin)))

    # Log ticks (minor-style grid at all decades)
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1)
    )

    # Rescaled tick labels: show mantissa only
    def mantissa_formatter(x, _):
        return f"{x / 10**exponent:.1f}"

    ax.xaxis.set_major_formatter(FuncFormatter(mantissa_formatter))

    # Add ×10^k label
    ax.text(
        xpos, ypos,
        rf"$\times 10^{{{exponent}}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
    )


def plot_rank_tracked_branches(
    ax,
    per_lam_pts,
    ykey,
    I_filter=None,
    y_min=None,
    y_max=None,
    linewidth=2.2,
    color=None,
    mark_unstable_extrema=True,
    marker_size=26,
    mark_y_min=5e-4,
):
    """
    Plot fixed-point branches obtained by rank tracking across lam values.

    The function assumes that, for each lam value, fixed points have already
    been sorted consistently (e.g., by decreasing prevalence). Branches are
    then reconstructed by connecting points with the same rank index across
    successive lam values.

    Stable and unstable segments are plotted with different linestyles:
    solid lines for stable branches and dashed lines for unstable ones.
    Optionally, the endpoints of unstable segments can be highlighted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which the branches are plotted.
    per_lam_pts : list of list of dict
        Fixed-point data grouped by lam value. Each inner list contains
        dictionaries with keys such as ``lam``, ``I``, ``P``, and ``stable``.
    ykey : str
        Key of the observable to plot on the y-axis (e.g., ``"I"`` or ``"P"``).
    I_filter : float, optional
        If given, only points with ``I > I_filter`` are retained.
    y_min : float, optional
        Minimum allowed y value. Points below this threshold are excluded.
    y_max : float, optional
        Maximum allowed y value. Points above this threshold are excluded.
    linewidth : float, optional
        Line width used for branch segments.
    color : str or tuple, optional
        Line and marker color.
    mark_unstable_extrema : bool, optional
        If True, mark the endpoints of unstable segments when ``ykey == "I"``.
    marker_size : float, optional
        Marker size used for unstable segment endpoints.
    mark_y_min : float, optional
        Minimum y value required to draw endpoint markers.

    Returns
    -------
    None
        The function modifies ``ax`` in place.
    """
    Kmax = max((len(pts) for pts in per_lam_pts), default=0)
    if Kmax == 0:
        return

    def in_window(y):
        if y_min is not None and y < y_min:
            return False
        if y_max is not None and y > y_max:
            return False
        return True

    for k in range(Kmax):
        seg_lam = []
        seg_y = []
        seg_stable = None

        def flush_segment():
            nonlocal seg_lam, seg_y, seg_stable

            if len(seg_lam) >= 2:
                ax.plot(
                    seg_lam,
                    seg_y,
                    linestyle="-" if seg_stable else "--",
                    linewidth=linewidth,
                    color=color,
                )

                if mark_unstable_extrema and (seg_stable is False) and (ykey == "I"):
                    endpoints = [(seg_lam[0], seg_y[0]), (seg_lam[-1], seg_y[-1])]
                    for lam_val, y_val in endpoints:
                        if y_val >= mark_y_min:
                            ax.scatter(
                                lam_val,
                                y_val,
                                s=marker_size,
                                facecolor="white",
                                edgecolor=color,
                                linewidth=1.0,
                                zorder=80,
                            )

            seg_lam = []
            seg_y = []
            seg_stable = None

        for pts in per_lam_pts:
            if len(pts) <= k:
                flush_segment()
                continue

            point = pts[k]

            if I_filter is not None and not (point["I"] > I_filter):
                flush_segment()
                continue

            y_val = point[ykey]
            if not np.isfinite(y_val):
                flush_segment()
                continue
            if ax.get_yscale() == "log" and y_val <= 0:
                flush_segment()
                continue
            if not in_window(y_val):
                flush_segment()
                continue

            is_stable = point["stable"]

            if seg_stable is None:
                seg_stable = is_stable
            elif is_stable != seg_stable:
                flush_segment()
                seg_stable = is_stable

            seg_lam.append(point["lam"])
            seg_y.append(y_val)

        flush_segment()
        

