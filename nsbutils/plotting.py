"""Plotting utilities for brain surfaces and heatmaps."""

from __future__ import annotations
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from matplotlib.axes import Axes

def plot_heatmap(
    data: ArrayLike,
    ax: Union[Axes, None] = None,
    center: Union[float, None] = None,
    cmap: Union[str, colors.Colormap] = "turbo",
    cbar: bool = False,
    square: bool = True,
    downsample: float = 1,
    annot: bool = False,
    fmt: str = ".1f"
) -> Axes:
    """
    Plot a heatmap of the data with optional colorbar and annotations.

    Parameters
    ----------
    data : 2D array-like
        The data to be plotted as a heatmap.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the heatmap. If None, the current axes are used.
    center : float, optional
        Center value for colormap scaling. If None, the colormap is not centered.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to be used for the heatmap. Default is "viridis".
    cbar : bool, optional
        Whether to display a colorbar beside the heatmap. Default is False.
    square : bool, optional
        If True, set the aspect ratio of the plot to be equal, so the cells are square-shaped.
        Default is True.
    downsample : float, optional
        Factor by which to downsample the data before plotting. Should be between 0 and 1.
    annot : bool, optional
        Whether to annotate each cell with its value. Default is False.
    fmt : str, optional
        String format for the annotations. Default is ".1f" (one decimal place).

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the heatmap is plotted.
    """
    data = np.asarray(data)

    if ax is None:
        ax = plt.gca()

    if 0 < downsample < 1:
        data = zoom(data, zoom=downsample, order=1) # bilinear interpolation
    elif downsample != 1:
        raise ValueError("`downsample` must be in the range (0, 1].")

    vmin = np.min(data)
    vmax = np.max(data)

    cmap = plt.get_cmap(cmap)
    if center is not None:
        # Compute a symmetric range around center
        vrange = max(abs(vmax - center), abs(center - vmin))
        norm = colors.Normalize(vmin=center - vrange, vmax=center + vrange)

        # Remap colormap to the new range
        cmin, cmax = norm([vmin, vmax])
        cc = np.linspace(cmin, cmax, 256)
        cmap = colors.ListedColormap(cmap(cc))

    # Plot heatmap with colorbar
    mesh = ax.pcolormesh(data, cmap=cmap, **{"vmin": vmin, "vmax": vmax})

    # Invert the y axis to show the plot in matrix form
    ax.invert_yaxis()

    # Annotate each cell with its value
    if annot:
        norm = mesh.norm  # get the normalization used by pcolormesh
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Use black or white text depending on background luminance
                val = data[i, j]
                r, g, b = cmap(norm(val))[:3]
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'white' if luminance < 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, format(val, fmt), ha='center', va='center',
                        color=text_color)

    # Create a colorbar with the same height as the heatmap
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)  # Adjust size and padding
        cb = plt.colorbar(mesh, cax=cax)

    # Set frame around heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    ax.set_xticks([])
    ax.set_yticks([])
    if square:
        ax.set_aspect("equal")

    return ax