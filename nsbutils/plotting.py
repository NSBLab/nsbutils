import tempfile
from pathlib import Path
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from surfplot import Plot
from typing import Optional, List, Union, Tuple, Dict, Any
from numpy.typing import NDArray
import warnings
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neuromodes.io import read_surf

# Set 1e6 distance to avoid "cutting" mesh when zooming in
camera_views = {
    'lateral':   dict(eye=dict(x=-1e6, y=0, z=0), up=dict(x=0, y=0, z=1)),
    'medial':    dict(eye=dict(x=1e6, y=0, z=0), up=dict(x=0, y=0, z=1)),
    'dorsal':    dict(eye=dict(x=0, y=0, z=1e6), up=dict(x=0, y=1, z=0)),
    'ventral':   dict(eye=dict(x=0, y=0, z=-1e6), up=dict(x=0, y=1, z=0)),
    'anterior':  dict(eye=dict(x=0, y=1e6, z=0), up=dict(x=0, z=1, y=0)),
    'posterior': dict(eye=dict(x=0, y=-1e6, z=0), up=dict(x=0, z=1, y=0))
}

_DEFAULT_PANEL_SIZE: Tuple[int, int] = (350, 200)  # (width, height) in pixels
_DEFAULT_ZOOM_FACTOR = 2.8

def _plotly_figure_to_rgba_array(fig: go.Figure, scale: float = 1.0) -> NDArray[np.uint8]:
    """Rasterize a Plotly figure to an RGBA uint8 image array in-memory.

    Notes
    -----
    Plotly image export typically requires the `kaleido` package.
    """
    try:
        png_bytes = fig.to_image(format="png", scale=scale)
    except Exception as e:  # plotly may raise ValueError if kaleido is missing
        raise RuntimeError(
            "Failed to export Plotly figure to an image. "
            "This typically requires the optional dependency 'kaleido' (e.g. `pip install kaleido`)."
        ) from e

    image = Image.open(BytesIO(png_bytes)).convert("RGBA")
    return np.asarray(image, dtype=np.uint8)

def _draw_image_on_axis(ax: plt.Axes, image: NDArray[np.uint8]) -> None:
    """Draw an RGBA image array onto a Matplotlib axis."""
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_surf_single(
        surf,
        data=None,
        rois=None,
        fig=None,
        row=None,
        col=None,
        view='lateral',
        zoom=1.0,
        size: Optional[Tuple[int, int]] = None,
        cbar=False,
        cmap='turbo',
        mesh_edges=False,
        roi_outlines=False,
) -> Optional[go.Figure]:
    """Render a single surface into a single Plotly 3D scene.

    This is a helper function for :func:`plot_surf`. It draws one hemisphere surface
    (optionally with a data overlay) into one subplot cell.

    Parameters
    ----------
    surf : str, Path, GiftiImage, lapy.TriaMesh, or dict
        Surface mesh understood by ``neuromodes.io.read_surf`` (e.g., a file path or
        compatible mesh object).
    data : array_like, optional
        Vertex-wise data to overlay.

        Expected shape is ``(n_vertices,)``.
        If ``rois`` is provided and ``data`` has shape ``(n_rois,)`` (where ROI labels
        are assumed to be 1..n_rois), it is expanded to vertex-wise values.
    rois : array_like, optional
        ROI labels of shape ``(n_vertices,)``. Vertices labeled ``0`` are treated as
        masked (e.g., medial wall) and are drawn in the base surface color.
    fig : plotly.graph_objects.Figure, optional
        If provided, add traces to this figure. If ``None``, a new 1x1 figure is created.
    row, col : int, optional
        Target subplot (1-indexing). Only used when ``fig`` is provided.
    view : str
        Camera view name. Must be a key in ``camera_views`` (e.g., ``"lateral"``,
        ``"medial"``, ``"dorsal"``, ``"ventral"``, ``"anterior"``, ``"posterior"``).
    zoom : float
        Zoom factor controlling how much the mesh fills the scene. Internally this
        is multiplied by ``_DEFAULT_ZOOM_FACTOR`` to preserve historical appearance.
    size : (int, int), optional
        Panel size as ``(width, height)`` in pixels. Only used when creating a new
        figure (i.e., when ``fig is None``).
    cbar : bool
        If True, show a colorbar for the data overlay.
    cmap : str
        Plotly colorscale name used for the data overlay.
    mesh_edges : bool
        If True, overlay triangle edges as black lines.
    roi_outlines : bool
        If True and ``rois`` is provided, draw ROI boundary outlines.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure that was created or modified.
    """

    # TODO: consider moving this to plot_surf so the same surface doesn't get loaded multiple
    # times when plotting multiple views or maps
    mesh = read_surf(surf)
    verts, faces = mesh.v, mesh.t
    n_verts = verts.shape[0]
    x, y, z = zip(*verts)
    i, j, k = zip(*faces)

    if data is not None:
        data = np.asarray(data)
        if data.shape != (n_verts,):
            raise ValueError(f"Data shape {data.shape} does not match mesh of shape (n_verts,) = ({n_verts},).")
    # TODO: how to handle nans in data?
    # TODO: allow data to have number of surface vertices or number of non-masked vertices (if rois provided).

    # Define mask based on 0-labeled vertices in rois (if provided)
    if rois is not None:
        # Validate rois shape
        if rois.shape != (n_verts,):
            raise ValueError(f"ROIs shape {rois.shape} does not match mesh of shape (n_verts,) = ({n_verts},).")

        roi_labels = rois
        mask = np.where(roi_labels == 0, False, True)

        # If data is provided as ROI-wise, map it to vertices
        if data is not None:
            n_rois = int(np.max(roi_labels))
            if len(data) == n_rois:
                # Map ROI data to vertices
                vertex_data = np.zeros(len(roi_labels))
                for roi_id in range(1, n_rois + 1):
                    vertex_data[roi_labels == roi_id] = data[roi_id - 1]
                data = vertex_data
    else:   # no mask
        mask = np.ones(n_verts, dtype=bool)

    panel_size = size or _DEFAULT_PANEL_SIZE

    # If fig, row, and col are not provided, create a new figure
    if fig is None:
        if row is not None or col is not None:
            warnings.warn("`row` and `col` are ignored when `fig` is not provided. A new figure will be created.", UserWarning)
        
        fig = make_subplots(
            rows=1, cols=1, 
            specs=[[{'type': 'scene'}]],
            horizontal_spacing=0.01,
            vertical_spacing=0.01  
        )
        row, col = 1, 1

        panel_w, panel_h = panel_size
        fig.update_layout(
            height=panel_h, width=panel_w,
            margin=dict(l=0, r=0, t=0, b=0)
        )

    if view not in camera_views:
        raise ValueError(f"Invalid view '{view}'. Valid options are: {list(camera_views.keys())}.")

    # Create mesh
    mesh_kwargs = dict(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        flatshading=False,
        lighting=dict(
            ambient=0.01,  # lower ambient = less overall light
            diffuse=1,     # full diffuse = full light reflection
            specular=0.1,  # very low specular = almost no shine
            roughness=1e-6,   # no roughness (needs to be non-zero when writing to image)
            fresnel=0      # no fresnel effect
        ),
        lightposition=dict(x=0, y=0, z=-1e5),
        intensitymode='vertex',
    )
    if data is not None:
        # If masked, create 2 meshes: one for masked, one for unmasked
        if mask is not None:
            tris_data = faces[np.all(mask[faces], axis=1)]
            tris_mask = faces[~np.all(mask[faces], axis=1)]

        # Create main mesh with data overlay
        mesh_data_kwargs = mesh_kwargs.copy()
        mesh_data_kwargs['i'] = tris_data[:, 0]
        mesh_data_kwargs['j'] = tris_data[:, 1]
        mesh_data_kwargs['k'] = tris_data[:, 2]
        mesh_data_kwargs['intensity'] = data
        mesh_data_kwargs['colorscale'] = cmap   # TODO: allow all default matplotlib colormaps as well
        mesh_data_kwargs['showscale'] = cbar
        if cbar:
            mesh_data_kwargs['showscale'] = True
            mesh_data_kwargs['colorbar'] = dict(len=1)
        # TODO: decide whether to name traces for easier retrieval later
        # mesh_data_kwargs['name'] = f'data-{hemi}-{map_idx}-{v_idx}'
        fig.add_trace(go.Mesh3d(**mesh_data_kwargs), row=row, col=col)

        # Create masked mesh
        mesh_mask_kwargs = mesh_kwargs.copy()
        mesh_mask_kwargs['i'] = tris_mask[:, 0]
        mesh_mask_kwargs['j'] = tris_mask[:, 1]
        mesh_mask_kwargs['k'] = tris_mask[:, 2]
        mesh_mask_kwargs['color'] = "lightgrey"
        mesh_mask_kwargs['showscale'] = False
        # mesh_mask_kwargs['name'] = f'mask-{hemi}-{map_idx}-{v_idx}'
        fig.add_trace(go.Mesh3d(**mesh_mask_kwargs), row=row, col=col)
        
    else:
        mesh_kwargs['color'] = "lightgrey"
        # mesh_kwargs['name'] = f'mesh-{hemi}-{map_idx}-{v_idx}'
        fig.add_trace(go.Mesh3d(**mesh_kwargs), row=row, col=col)

    # Add edges
    if mesh_edges:
        xe = verts[faces[:, [0,1,2,0]], 0].flatten()
        ye = verts[faces[:, [0,1,2,0]], 1].flatten()
        ze = verts[faces[:, [0,1,2,0]], 2].flatten()
        xe_sep, ye_sep, ze_sep = [], [], []
        for idx in range(0, len(xe), 4):
            xe_sep.extend(xe[idx:idx+4]); xe_sep.append(np.nan)
            ye_sep.extend(ye[idx:idx+4]); ye_sep.append(np.nan)
            ze_sep.extend(ze[idx:idx+4]); ze_sep.append(np.nan)

        fig.add_trace(
            go.Scatter3d(
                x=xe_sep, y=ye_sep, z=ze_sep,
                mode="lines",
                line=dict(color="black", width=0.5),
                # name=f'edges-{hemi}-{map_idx}-{v_idx}',
                showlegend=False
            ), row=row, col=col
        )

    # Add ROI outlines
    if roi_outlines and rois is not None:
        xe, ye, ze = compute_roi_midline_edges(verts, faces, roi_labels)
        fig.add_trace(
            go.Scatter3d(
                x=xe, y=ye, z=ze,
                mode="lines",
                marker=dict(color="black", size=20),
                # name=f'rois-{hemi}-{map_idx}-{v_idx}',
                showlegend=False
            ), row=row, col=col
        )
        # TODO: define the outline between data and mask differently such that it perfectly
        # follows the triangle boundaries instead of cutting through them.

    # Remove axes
    noaxis = dict(
        showbackground=False, showline=False, zeroline=False, showgrid=False, 
        showticklabels=False, title="", visible=False
    )
    # Compute initial aspect ratio for zoom
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)
    max_range = max(x_range, y_range, z_range)

    fig.update_scenes(
        xaxis=noaxis, yaxis=noaxis, zaxis=noaxis,
        aspectmode='manual',
        aspectratio=dict(
            x=x_range/max_range * zoom * _DEFAULT_ZOOM_FACTOR,
            y=y_range/max_range * zoom * _DEFAULT_ZOOM_FACTOR,
            z=z_range/max_range * zoom * _DEFAULT_ZOOM_FACTOR
        ),
        # TODO: allow users to input custom view angles
        camera=dict(
            center=dict(x=0, y=0, z=0),
            eye=camera_views[view]['eye'],
            up=camera_views[view]['up'],
            projection=dict(type='orthographic'),
        ),
        row=row, col=col
    )

    return fig


def plot_surf(
        surf,
        data=None,
        rois=None,
        views=['lateral', 'medial'],
        layout_indiv='row',
        layout_group='row',
        zoom=1.0,
        size: Tuple[int, int] = _DEFAULT_PANEL_SIZE,
        cbar=False,
        cmap='turbo',
        mesh_edges=False, 
        roi_outlines=False,
        ax: Optional[plt.Axes] = None,
        scale: float = 1.0
    ) -> Optional[go.Figure]:
    """Plot surface data across hemispheres, views, and (optionally) multiple maps.

    This function builds a Plotly subplot grid and repeatedly calls
    :func:`plot_surf_single` to populate each panel.

    Parameters
    ----------
    surf : dict
        Dictionary mapping hemisphere keys (typically ``"lh"`` and/or ``"rh"``) to
        surface meshes understood by ``neuromodes.io.read_surf``.
    data : dict, optional
        Dictionary mapping hemisphere keys to data arrays.

        Per-hemisphere accepted shapes:
        - ``(n_vertices,)`` for a single map
        - ``(n_vertices, n_maps)`` for multiple maps

        If 1D, arrays are promoted to 2D internally.
    rois : dict, optional
        Dictionary mapping hemisphere keys to ROI labels of shape ``(n_vertices,)``.
        Vertices labeled ``0`` are treated as masked.
    views : list of str
        View names to render. Each entry must be a key in ``camera_views``.
    layout_indiv : str
        Layout of panels within a single map:
        - ``"row"``: panels laid out left-to-right
        - ``"col"``: panels laid out top-to-bottom
        - ``"grid"``: rows = views, cols = hemis
    layout_group : str
        How multiple maps are tiled:
        - ``"row"``: maps laid out left-to-right
        - ``"col"``: maps laid out top-to-bottom
    zoom : float
        Zoom factor forwarded to :func:`plot_surf_single`. Internally the effective
        scaling includes ``_DEFAULT_ZOOM_FACTOR``.
    size : (int, int)
        Per-panel size as ``(width, height)`` in pixels. Overall figure size is
        computed as ``(width * cols, height * rows)``.
    cbar : bool
        If True, show a colorbar for each data mesh.
    cmap : str
        Plotly colorscale name used for the data overlay.
    mesh_edges : bool
        If True, overlay triangle edges as black lines.
    roi_outlines : bool
        If True and ``rois`` is provided, draw ROI boundary outlines.
    ax : matplotlib.axes.Axes, optional
        If provided, the composed Plotly figure is rasterized in-memory via
        ``fig.to_image`` and drawn into this axis with ``imshow``. In this case the
        function returns ``None``.
    scale : float
        Rasterization scale passed to ``fig.to_image(scale=...)`` when ``ax`` is used.

    Notes
    -----
    For the right hemisphere, view names are swapped with their opposites so that labels 
    correspond to the same anatomical surface as the left hemisphere: 
    ``lateral`` <-> ``medial``, ``dorsal`` <-> ``ventral``, ``anterior`` <-> ``posterior``.

    When plotting both hemispheres and ``layout_indiv`` is ``"row"`` or ``"col"``, the 
    order of the RH views is also reversed to visually mirror the LH block. For 
    ``layout_indiv="grid"``, RH view order is not reversed so that rows remain aligned by 
    view name.

    Returns
    -------
    plotly.graph_objects.Figure or None
        The interactive Plotly figure if ``ax is None``; otherwise ``None``.
    """
    # TODO: consider using multiple arguments (i.e. *_lh and *_rh) instead of dict with 
    # 'lh' and 'rh' for surf, data, and rois since setting up a dict can be a bit tedious.
    # FWIW, surfplot has muiltiple *_lh and *_rh arguments.

    hemis = list(surf.keys())
    n_hemi = len(hemis)
    n_views = len(views)

    # Determine number of maps and ensure hemi data is 2D (n_verts, n_maps)
    if data is None:
        n_maps = 1
    else:
        for hemi in hemis:
            if hemi not in data:
                raise ValueError(f"Missing data for hemisphere '{hemi}'.")
            if np.ndim(data[hemi]) == 1:
                data[hemi] = data[hemi][:, np.newaxis]
        n_maps = data[hemis[0]].shape[1]

    # Individual block size (hemis × views) for a single map
    if layout_indiv == 'row':
        indiv_rows, indiv_cols = 1, n_hemi * n_views
    elif layout_indiv == 'col':
        indiv_rows, indiv_cols = n_hemi * n_views, 1
    elif layout_indiv == 'grid':
        # rows correspond to views, cols correspond to hemis
        indiv_rows, indiv_cols = n_views, n_hemi
    else:
        raise ValueError("`layout_indiv` must be one of 'row', 'col', or 'grid'.")

    # Group layout: tile the individual block across maps
    if layout_group == 'row':
        rows, cols = indiv_rows, indiv_cols * n_maps
    elif layout_group == 'col':
        rows, cols = indiv_rows * n_maps, indiv_cols
    else:
        raise ValueError("`layout_group` must be one of 'row' or 'col'.")

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)],
        horizontal_spacing=0.01,
        vertical_spacing=0.01
    )
    # TODO: add labels. Could use one or a combination of these in make_subplots:
    # subplot_titles, row_titles, column_titles  

    # Only reverse RH view order when both hemis are present *and* views are laid
    # out sequentially (row/col). For grid layout, rows correspond to view names,
    # so reversing RH view order would misalign rows across hemis.
    mirror_rh = ('lh' in hemis and 'rh' in hemis)
    reverse_rh_view_order = mirror_rh and (layout_indiv in ('row', 'col'))

    # Loop through maps, hemispheres, and views
    for map_idx in range(n_maps):
        # Per-map tiling offset
        if layout_group == 'row':
            row_offset, col_offset = 0, map_idx * indiv_cols
        else:  # layout_group == 'col'
            row_offset, col_offset = map_idx * indiv_rows, 0

        for h_idx, hemi in enumerate(hemis):
            views_for_hemi = views[::-1] if (hemi == 'rh' and reverse_rh_view_order) else views

            for v_idx, view in enumerate(views_for_hemi):
                # For the right hemisphere, swap paired opposite views so that view
                # labels correspond to the same anatomical surface as for the left.
                # (e.g., looking from -x shows LH lateral but RH medial.)
                camera_view = view
                if hemi == 'rh':
                    rh_view_swap = {
                        'lateral': 'medial',
                        'medial': 'lateral',
                        'dorsal': 'ventral',
                        'ventral': 'dorsal',
                        'anterior': 'posterior',
                        'posterior': 'anterior',
                    }
                    camera_view = rh_view_swap.get(view, view)

                # Determine subplot row/col within the individual block (1-based)
                if layout_indiv == 'row':
                    r0, c0 = 1, h_idx * n_views + v_idx + 1
                elif layout_indiv == 'col':
                    r0, c0 = h_idx * n_views + v_idx + 1, 1
                else:  # grid
                    r0, c0 = v_idx + 1, h_idx + 1

                r, c = r0 + row_offset, c0 + col_offset

                plot_surf_single(
                    surf=surf[hemi],
                    fig=fig,
                    row=r,
                    col=c,
                    data=None if data is None else data[hemi][:, map_idx],
                    rois=rois[hemi] if rois is not None else None,
                    view=camera_view,
                    zoom=zoom,
                    size=size,
                    cbar=cbar,  # TODO: give option to show individual or group colorbars
                    cmap=cmap,
                    mesh_edges=mesh_edges,
                    roi_outlines=roi_outlines
                )
    
    # General layout
    panel_w, panel_h = size
    fig.update_layout(
        height=panel_h * rows, width=panel_w * cols,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    if ax is not None:
        image = _plotly_figure_to_rgba_array(fig, scale=scale)
        _draw_image_on_axis(ax, image)
        return None

    return fig

def compute_roi_midline_edges(verts, faces, labeling, verbose=False):
    """
    Compute ROI boundaries using midpoints between label boundaries.
    Matches MATLAB findROIboundaries.m behavior, including medial wall borders.
    """
    labeling = np.asarray(labeling)
    labeling = np.nan_to_num(labeling, nan=0).astype(int)

    tri_labels = labeling[faces]
    tri_coords = verts[faces]

    line_segments = []
    for lbls, coords in zip(tri_labels, tri_coords):
        unique_lbls = np.unique(lbls)

        # Skip all-zero triangles (pure medial wall)
        if np.all(unique_lbls == 0):
            continue

        edges = [(0, 1), (1, 2), (2, 0)]

        # Two or more distinct labels — boundary triangle
        if len(unique_lbls) == 2:
            # Includes case {0, X}
            diff_edges = [e for e in edges if lbls[e[0]] != lbls[e[1]]]
            if len(diff_edges) == 2:
                mids = [coords[list(e)].mean(axis=0) for e in diff_edges]
                line_segments.append(np.vstack(mids))

        elif len(unique_lbls) == 3:
            # Three-way junction: draw centroid-to-midpoint lines
            centroid = coords.mean(axis=0)
            mids = [coords[list(e)].mean(axis=0) for e in edges]
            for m in mids:
                line_segments.append(np.vstack([centroid, m]))

    if not line_segments:
        if verbose:
            print("No ROI boundaries found.")
        return np.array([]), np.array([]), np.array([])

    segs = np.stack(line_segments)
    n = len(segs)
    xe = np.empty(n * 3)
    ye = np.empty_like(xe)
    ze = np.empty_like(xe)

    xe[0::3] = segs[:, 0, 0]; xe[1::3] = segs[:, 1, 0]; xe[2::3] = np.nan
    ye[0::3] = segs[:, 0, 1]; ye[1::3] = segs[:, 1, 1]; ye[2::3] = np.nan
    ze[0::3] = segs[:, 0, 2]; ze[1::3] = segs[:, 1, 2]; ze[2::3] = np.nan

    return xe, ye, ze


def fetch_trace(fig, name):
    """
    Fetch a trace from a Plotly figure by its name.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure containing the trace.
    name : str
        The name of the trace to fetch.

    Returns
    -------
    trace : plotly.graph_objects.Trace or None
        The trace with the specified name, or None if not found.
    """
    return next((t for t in fig.data if t.name == name), None)

def update_trace_type(fig, trace_type, **kwargs):
    """
    Update traces by name pattern with given kwargs.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure containing traces to update.
    trace_type : str
        The trace name or prefix to match. Will match traces where name starts with this string.
        Examples: 'data-lh-0' (exact match), 'data' (all data traces).
    **kwargs
        Keyword arguments to pass to trace.update().
    
    """
    for trace in fig.data:
        if trace.name and trace.name.startswith(trace_type):
            trace.update(**kwargs)
    return

def compute_roi_outlines(surf, labeling):
    """
    Compute a binary mask of border vertices given a parcellation labeling.

    Parameters
    ----------
    surf : dict
        Must contain keys 'v' (vertices, n×3) and 't' (triangles, m×3).
    labeling : array_like of shape (n,)
        ROI label per vertex. Typically integers, with 0 for medial wall.

    Returns
    -------
    border : np.ndarray of shape (n,)
        Binary mask where 1 indicates a vertex on a label boundary.
    """
    tris = np.asarray(surf['t'])
    labeling = np.asarray(labeling)
    n_verts = len(labeling)

    # Get all edges (each as a sorted vertex pair)
    edges = np.sort(
        np.vstack([
            tris[:, [0, 1]],
            tris[:, [1, 2]],
            tris[:, [2, 0]],
        ]),
        axis=1
    )

    # Remove duplicates
    edges = np.unique(edges, axis=0)

    # Find edges with label mismatch
    edge_labels = labeling[edges]
    border_edges = edges[edge_labels[:, 0] != edge_labels[:, 1]]

    # Mark border vertices
    border = np.zeros(n_verts, dtype=np.uint8)
    border[np.unique(border_edges)] = 1

    return border


def plot_surf_surfplot(
    mesh: Union[str, Path],
    data: Union[NDArray, List[float], List[List[float]]],
    layout: str = "row",
    views: List[str] = ["lateral", "medial"],
    color_range: Union[Tuple[float, float], str] = "individual",
    center: Optional[float] = None,
    cmap: Union[str, colors.Colormap] = "viridis",
    cbar: bool = False,
    cbar_label: Optional[str] = None,
    cbar_kws: Optional[Dict[str, Any]] = None,
    labels: Optional[List[str]] = None,
    label_kws: Optional[Dict[str, Any]] = None,
    outline: bool = False,
    zoom: float = 1.25,
    ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None
) -> Optional[plt.Figure]:
    """
    Plot brain surface data on a given surface mesh.

    Parameters
    ----------
    mesh : str or pathlib.Path
        The surface mesh to be used.
    data : array-like
        Data to be plotted on the surface. Can be 1D or 2D with shape (n_verts, n_maps). Note that 
        NaNs are not colored, but zeros are.
    layout : str, optional
        Layout of the subplots, either "row" or "col", by default "row".
    views : list of str, optional
        List of views to display, by default ["lateral", "medial"].
    color_range : tuple of float, str, or None, optional
        Defines the color limits for the colormap. Can be:
        - A tuple (vmin, vmax) to apply the same color scale across all maps.
        - "group" to compute global (min, max) across all data columns and apply uniformly.
        - "individual" to compute limits separately for each brain map.
        By default, color range is determined individually per map.
    center : float, optional
        Center value for colormap scaling. If provided, color range will be symmetric around center.
        Note that `center` is ignored if `color_range` is a tuple.
    cmap : matplotlib colormap name or object, optional
        Colormap to use for the data, by default "viridis".
    cbar : bool, optional
        Whether to display a colorbar, by default False.
    cbar_label : str, optional
        Label for the colorbar, by default None.
    cbar_kws : dict, optional
        Additional keyword arguments for the colorbar, by default None.
    labels : list of str, optional
        List of labels for each subplot, by default None.
    label_kws : dict, optional
        Additional keyword arguments for the labels, by default None.
    outline : bool, optional
        Whether to outline the data, by default False. Useful for parcellations.
    zoom : float, optional
        Zoom factor for the brain plot, by default 1.25.
    ax : matplotlib.axes.Axes or list of Axes, optional
        Axis or list of axes to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure or None
        The resulting figure if a new one is created, otherwise None.
    """
    data = np.asarray(data)

    cbar_kws_ = {**dict(pad=0.01, fontsize=20, aspect=25, shrink=1, decimals=2, location="bottom"),
                 **(cbar_kws or {})}
    label_kws_ = {**dict(fontsize=20), **(label_kws or {})}
    
    data = np.squeeze(data)
    if np.ndim(data) == 1 or np.shape(data)[1] == 1:
        data = data.reshape(-1, 1)
    
    n_data = np.shape(data)[1]
    
    # Create the figure and axes
    if ax is None:
        if layout == "row":
            fig, axs = plt.subplots(1, n_data, figsize=(len(views) * n_data * 1.5, 2))
        elif layout == "col":
            fig, axs = plt.subplots(n_data, 1, figsize=(3, n_data * 1.25))
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        axs = [axs] if n_data == 1 else axs.flatten()
    else:
        if isinstance(ax, list):
            if len(ax) != n_data:
                raise ValueError("Number of provided axes must match the number of brains to plot.")
            axs = ax
        else:
            if n_data > 1:
                raise ValueError("Multiple brains require a list of axes.")
            axs = [ax]
    
    # Set the color range
    if isinstance(color_range, tuple):
        crange = color_range
        if center is not None:
            warnings.warn("`center` is ignored when `color_range` is a tuple.", UserWarning)
    if color_range == "group":
        vmax = np.nanmax(data)
        vmin = np.nanmin(data)
        if center is not None:
            vrange = max(abs(vmax - center), abs(center - vmin))
            crange = (center - vrange, center + vrange)
        else:
            crange = (vmin, vmax)
    else:
        crange = None

    # To plot multiple brain maps, save each figure to a temporary file then load it into the axes
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, ax in enumerate(axs):

            # Set color range for centered "individual"
            if color_range == "individual" and center is not None:
                vmax = np.nanmax(data[:, i])
                vmin = np.nanmin(data[:, i])
                vrange = max(abs(vmax - center), abs(center - vmin))
                crange = (center - vrange, center + vrange)
                    
            # Use surfplot to plot the data
            p = Plot(surf_lh=mesh, views=views, size=(500, 250), zoom=zoom)
            p.add_layer(data=data[:, i], cmap=cmap, cbar=cbar, color_range=crange,
                        cbar_label=cbar_label, zero_transparent=False)
            if outline:
                p.add_layer(data[:, i], as_outline=True, cmap="gray", cbar=False,
                            color_range=(1, 2), zero_transparent=False)
            temp_file = f"{temp_dir}/figure_{i}.png"
            fig = p.build(cbar_kws=cbar_kws_)
            plt.close(fig)
            
            # Save the surfplot figure
            fig.savefig(temp_file, bbox_inches='tight')
            # Load the figure into the axes
            ax.imshow(plt.imread(temp_file))
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            # Plot labels
            if labels is not None:
                if layout == "row":
                    ax.set_title(labels[i], pad=0, fontsize=label_kws_["fontsize"])
                elif layout == "col":
                    ax.set_ylabel(labels[i], labelpad=0, rotation=0, ha="right",
                                  fontsize=label_kws_["fontsize"])
    
    return fig if ax is None else None

def plot_heatmap(
    data: Union[NDArray, List[List[float]]],
    ax: Optional[plt.Axes] = None,
    center: Optional[float] = None,
    cmap: Union[str, colors.Colormap] = "viridis",
    cbar: bool = False,
    square: bool = True,
    downsample: float = 1,
    annot: bool = False,
    fmt: str = ".1f"
) -> plt.Axes:
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