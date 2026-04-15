import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.collections import LineCollection
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator


DEFAULT_GRAY = "0.55"
DEFAULT_TICK_LABELSIZE = 12
DEFAULT_AXIS_LABELSIZE = 15


def get_plot_style(gray=DEFAULT_GRAY, tick_labelsize=DEFAULT_TICK_LABELSIZE, axis_labelsize=DEFAULT_AXIS_LABELSIZE):
    """
    Return a dictionary with default plotting style parameters.
    """
    return {
        "gray": gray,
        "tick_labelsize": tick_labelsize,
        "axis_labelsize": axis_labelsize,
    }


def make_truncated_colormap(name="YlOrBr_r", vmin=0.1, vmax=0.65, n=256):
    """
    Build a truncated matplotlib colormap.
    """
    base_cmap = plt.get_cmap(name)
    trunc_colors = base_cmap(np.linspace(vmin, vmax, n))
    return LinearSegmentedColormap.from_list(f"{name}_trunc", trunc_colors)


def apply_axis_style(ax, style=None):
    """
    Apply the default axis style using ``style_axis``.
    """
    if style is None:
        style = get_plot_style()

    style_axis(
        ax=ax,
        gray=style["gray"],
        labelsize=style["tick_labelsize"],
    )


def add_log_minor_ticks(ax, axis="y"):
    """
    Add minor logarithmic ticks and hide their labels.
    """
    locator = LogLocator(base=10.0, subs=np.arange(2, 10))
    formatter = NullFormatter()

    if axis == "x":
        ax.xaxis.set_minor_locator(locator)
        ax.xaxis.set_minor_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_minor_locator(locator)
        ax.yaxis.set_minor_formatter(formatter)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def load_npz(path):
    """
    Load a NumPy ``.npz`` file.
    """
    return np.load(path, allow_pickle=True)

def build_colored_line_segments(x, y):
    """
    Convert 1D arrays x and y into segments for a LineCollection.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)

def style_axis(ax, gray="0.55", labelsize=12):
    """
    Apply a consistent style to matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to be styled.
    gray : str, optional
        Color used for ticks, labels, and spines.
    labelsize : int, optional
        Font size for tick labels.
    """
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=labelsize,
        color=gray,
        labelcolor=gray,
        length=3,
        width=1.0
    )

    for spine in ax.spines.values():
        spine.set_edgecolor(gray)
        spine.set_linewidth(1.0)
        