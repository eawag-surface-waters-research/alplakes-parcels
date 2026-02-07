import parcels
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

def load_delf3d_fieldset(data_file):
    print("Loading delf3d fieldset: {}".format(data_file))
    filenames = {'U': data_file, 'V': data_file, 'W': data_file}
    variables = {'U': 'U', 'V': 'V', 'W': 'W'}
    c_dims = {
        'lon': 'glamf',      # f-point x-coordinates (from XCOR)
        'lat': 'gphif',      # f-point y-coordinates (from YCOR)
        'depth': 'depth',    # layer centre depths
        'time': 'time',
    }
    dimensions = {'U': c_dims, 'V': c_dims, 'W': c_dims}
    fieldset = parcels.FieldSet.from_c_grid_dataset(
        filenames, variables, dimensions,
        mesh='flat',                 # projected coordinates (meters)
        gridindexingtype='nemo',     # Delft3D matches NEMO low-padding
    )
    print("Successfully loaded delf3d fieldset")
    return fieldset


def fill_grid_coords(xcor, ycor):
    """Fill NaN coordinate values using smooth interpolation in index space.

    The valid grid points define a mapping from (i, j) indices to (x, y)
    coordinates. We fit a smooth surface through these points and
    evaluate it at the invalid indices to get reasonable coordinates.

    Uses scipy RBF interpolation which handles scattered data and
    extrapolates smoothly beyond the convex hull.
    """
    from scipy.interpolate import RBFInterpolator

    xf = xcor.copy().astype(np.float64)
    yf = ycor.copy().astype(np.float64)

    ny, nx = xf.shape
    invalid = np.isnan(xf) | np.isnan(yf)
    valid = ~invalid

    if not invalid.any():
        return xf, yf

    # Build arrays of (i, j) indices for valid and invalid points
    ii, jj = np.mgrid[0:ny, 0:nx]

    valid_ij = np.column_stack([ii[valid], jj[valid]])
    invalid_ij = np.column_stack([ii[invalid], jj[invalid]])

    valid_x = xf[valid]
    valid_y = yf[valid]

    # Subsample valid points if there are too many (RBF can be slow)
    max_points = 2000
    if len(valid_x) > max_points:
        step = len(valid_x) // max_points
        valid_ij_sub = valid_ij[::step]
        valid_x_sub = valid_x[::step]
        valid_y_sub = valid_y[::step]
    else:
        valid_ij_sub = valid_ij
        valid_x_sub = valid_x
        valid_y_sub = valid_y

    print(f"    RBF interpolation: {len(valid_x_sub)} source points, "
          f"{len(invalid_ij)} target points")

    # Fit RBF for x and y separately
    # thin_plate_spline gives smooth extrapolation
    rbf_x = RBFInterpolator(valid_ij_sub, valid_x_sub, kernel='thin_plate_spline')
    rbf_y = RBFInterpolator(valid_ij_sub, valid_y_sub, kernel='thin_plate_spline')

    xf[invalid] = rbf_x(invalid_ij)
    yf[invalid] = rbf_y(invalid_ij)

    return xf, yf


def convert_delft3d_to_parcels(input_file, output_file):
    print(f"Reading {input_file}...")
    ds = xr.open_dataset(input_file)

    # ─── Grid dimensions ───────────────────────────────────────────
    mc = ds.sizes['MC']  # 182
    nc = ds.sizes['NC']  # 36
    m = ds.sizes['M']  # 182
    n = ds.sizes['N']  # 36
    print(f"  Grid: MC={mc}, NC={nc}, M={m}, N={n}")
    print(f"  MC==M: {mc == m}, NC==N: {nc == n}")

    # ─── F-point (corner) coordinates ──────────────────────────────
    # XCOR, YCOR are the grid corners — these are the f-points that
    # Parcels needs for C-grid interpolation
    xcor = ds['XCOR'].values  # (MC=182, NC=36)
    ycor = ds['YCOR'].values

    print(f"  XCOR shape: {xcor.shape}, NaN count: {np.isnan(xcor).sum()}")
    print(f"  YCOR shape: {ycor.shape}, NaN count: {np.isnan(ycor).sum()}")

    # Delft3D often stores 0.0 (not NaN) at inactive grid points.
    # Detect inactive corners: where BOTH xcor and ycor are exactly 0
    # (valid projected coordinates are never exactly 0 for a lake)
    inactive_corners = (xcor == 0.0) & (ycor == 0.0)
    print(f"  Inactive corners (zero coords): {inactive_corners.sum()} / {inactive_corners.size}")

    # Set inactive points to NaN so fill_nans_2d can extrapolate them
    xcor[inactive_corners] = np.nan
    ycor[inactive_corners] = np.nan

    # Fill NaN in corner coordinates — fill X and Y together to maintain
    # grid cell consistency (no folded/degenerate cells)
    xcor_filled, ycor_filled = fill_grid_coords(xcor, ycor)
    print(f"  After filling — XCOR NaN: {np.isnan(xcor_filled).sum()}, "
          f"YCOR NaN: {np.isnan(ycor_filled).sum()}")

    plt.scatter(xcor_filled, ycor_filled)
    plt.scatter(xcor, ycor, color="red", marker="x")
    plt.show()

    # ─── Transpose to match Parcels convention ──────────────────────
    # Delft3D stores arrays as (MC, NC) = (182, 36) where:
    #   dim0 (MC=182) = along-lake (easting/lon-like varies most)
    #   dim1 (NC=36)  = across-lake (northing/lat-like varies most)
    # Parcels expects glamf(y, x) where y=lat-like (dim0), x=lon-like (dim1)
    # So we need (NC, MC) = (36, 182): transpose!
    # Note: the original arrays are already (MC=182, NC=36) = (lon-like, lat-like)
    # Transposing gives (NC=36, MC=182) = (lat-like, lon-like) = (y, x) ✓
    xcor_filled = xcor_filled.T  # (182,36) -> (36,182)
    ycor_filled = ycor_filled.T
    print(f"  After transpose: glamf shape = {xcor_filled.shape} (y, x)")
    zk_interfaces = ds['ZK'].values  # (101,) — layer interfaces, for W
    zk_centers = ds['ZK_LYR'].values  # (100,) — layer centres, for U/V

    print(f"  Depth interfaces: {len(zk_interfaces)}, "
          f"range: [{zk_interfaces.min():.1f}, {zk_interfaces.max():.1f}] m")
    print(f"  Depth centres: {len(zk_centers)}, "
          f"range: [{zk_centers.min():.1f}, {zk_centers.max():.1f}] m")

    # ─── Time ──────────────────────────────────────────────────────
    time_data = ds['time'].values
    nt = len(time_data)
    print(f"  Time steps: {nt}")

    # ─── Velocities ────────────────────────────────────────────────
    # U1: (time, KMAXOUT_RESTR=100, MC=182, N=36)  — edge1
    # V1: (time, KMAXOUT_RESTR=100, M=182, NC=36)  — edge2
    # Since MC==M and NC==N (both 182 and 36), the arrays are already
    # the same shape. In Delft3D with low-padding where M:MC, the
    # extra row/column at index 0 is the padding — matching NEMO style.

    print("  Loading U velocity...")
    u_data = ds['U1'].values  # (nt, 100, 182, 36)
    u_data = np.nan_to_num(u_data, nan=0.0)

    print("  Loading V velocity...")
    v_data = ds['V1'].values  # (nt, 100, 182, 36)
    v_data = np.nan_to_num(v_data, nan=0.0)

    print("  Loading W velocity...")
    # WPHY = physical vertical velocity at layer centres
    # shape: (time, KMAXOUT_RESTR=100, M=182, N=36)
    w_data = ds['WPHY'].values
    w_data = np.nan_to_num(w_data, nan=0.0)

    print(f"  U shape: {u_data.shape}")
    print(f"  V shape: {v_data.shape}")
    print(f"  W shape: {w_data.shape}")

    # Transpose spatial dims to match coordinate transpose: (t, K, 182, 36) -> (t, K, 36, 182)
    u_data = np.transpose(u_data, (0, 1, 3, 2))
    v_data = np.transpose(v_data, (0, 1, 3, 2))
    w_data = np.transpose(w_data, (0, 1, 3, 2))
    print(f"  After transpose: U={u_data.shape}, V={v_data.shape}, W={w_data.shape}")

    assert u_data.shape == v_data.shape == w_data.shape, \
        (f"Shape mismatch! U={u_data.shape}, V={v_data.shape}, W={w_data.shape}. "
         f"If MC!=M or NC!=N, padding is needed — see comments in code.")

    # ─── Water level (optional but useful) ─────────────────────────
    s1 = ds['S1'].values  # (time, M, N)
    s1 = np.nan_to_num(s1, nan=0.0)
    s1 = np.transpose(s1, (0, 2, 1))  # (time, 182, 36) -> (time, 36, 182)

    # ─── Build output dataset ──────────────────────────────────────
    print("  Building output dataset...")

    nk = len(zk_centers)

    out = xr.Dataset(
        {
            # ── Velocity fields (all same dims) ────────────────────
            'U': xr.DataArray(
                u_data.astype(np.float32),
                dims=['time', 'depth', 'y', 'x'],
                attrs={'units': 'm/s',
                       'long_name': 'U-velocity (grid-aligned, on west cell face)'}
            ),
            'V': xr.DataArray(
                v_data.astype(np.float32),
                dims=['time', 'depth', 'y', 'x'],
                attrs={'units': 'm/s',
                       'long_name': 'V-velocity (grid-aligned, on south cell face)'}
            ),
            'W': xr.DataArray(
                w_data.astype(np.float32),
                dims=['time', 'depth', 'y', 'x'],
                attrs={'units': 'm/s',
                       'long_name': 'W-velocity (physical vertical)'}
            ),

            # ── F-point (corner) coordinates ───────────────────────
            'glamf': xr.DataArray(
                xcor_filled.astype(np.float64),
                dims=['y', 'x'],
                attrs={'units': 'm',
                       'long_name': 'X-coordinate of grid corners (f-points)'}
            ),
            'gphif': xr.DataArray(
                ycor_filled.astype(np.float64),
                dims=['y', 'x'],
                attrs={'units': 'm',
                       'long_name': 'Y-coordinate of grid corners (f-points)'}
            ),

            # ── Water level ────────────────────────────────────────
            'S1': xr.DataArray(
                s1.astype(np.float32),
                dims=['time', 'y', 'x'],
                attrs={'units': 'm', 'long_name': 'Water level'}
            ),
        },
        coords={
            'time': ('time', time_data),
            'depth': ('depth', zk_centers.astype(np.float32),
                      {'units': 'm', 'positive': 'up',
                       'long_name': 'Depth at layer centres'}),
        },
        attrs={
            'Conventions': 'CF-1.6',
            'source': 'Converted from Delft3D-FLOW z-layer output for OceanParcels',
            'original_file': input_file,
            'grid_type': 'Arakawa C-grid (NEMO-style indexing, low padding)',
            'coordinate_system': 'projected (meters)',
            'parcels_usage': 'Use FieldSet.from_c_grid_dataset() with '
                             'gridindexingtype="nemo" and mesh="flat"',
            'note': ('glamf/gphif = f-point (corner) coordinates from XCOR/YCOR. '
                     'NaN coordinates filled via nearest-neighbor extrapolation. '
                     'NaN velocities replaced with 0 (land = no flow).'),
            'LAYER_MODEL': 'Z-MODEL',
        }
    )

    # ─── Write output ──────────────────────────────────────────────
    print(f"  Writing {output_file}...")
    encoding = {var: {'zlib': True, 'complevel': 4}
                for var in ['U', 'V', 'W', 'S1']}
    out.to_netcdf(output_file, encoding=encoding)

    print("  Done!\n")
    print(f"  Output variables: {list(out.data_vars)}")
    print(f"  Dimensions: time={nt}, depth={nk}, "
          f"y={out.dims['y']}, x={out.dims['x']}")

    ds.close()
    return out

