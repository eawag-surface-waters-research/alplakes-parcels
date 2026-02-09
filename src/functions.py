import json
import parcels
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

def load_delf3d_fieldset(data_file):
    print("Loading delf3d fieldset: {}".format(data_file))
    filenames = {'U': data_file, 'V': data_file, 'W': data_file}
    variables = {'U': 'U', 'V': 'V', 'W': 'W'}
    c_dims = {
        'lon': 'glamf',
        'lat': 'gphif',
        'depth': 'depth',
        'time': 'time',
    }
    dimensions = {'U': c_dims, 'V': c_dims, 'W': c_dims}
    fieldset = parcels.FieldSet.from_c_grid_dataset(
        filenames, variables, dimensions,
        mesh='flat',
        gridindexingtype='nemo',
    )
    print("Successfully loaded delf3d fieldset")
    return fieldset

def load_alplakes_mitgcm_fieldset(data_file):
    print("Loading Aplakes postprocessed mitgcm fieldset: {}".format(data_file))
    filenames = {'U': data_file, 'V': data_file, 'W': data_file}
    variables = {"U": "u", "V": "v", "W": "w"}
    dims = {"lon": "lng", "lat": "lat", "depth": "depth", "time": "time"}
    dimensions = {"U": dims, "V": dims, "W": dims}
    fieldset = parcels.FieldSet.from_netcdf(
        filenames,
        variables,
        dimensions,
    )
    print("Successfully loaded delf3d fieldset")
    return fieldset


def fill_grid_coords(xcor, ycor):
    xf = xcor.copy().astype(np.float64)
    yf = ycor.copy().astype(np.float64)
    ny, nx = xf.shape
    invalid = np.isnan(xf) | np.isnan(yf)
    valid = ~invalid

    if not invalid.any():
        return xf, yf

    ii, jj = np.mgrid[0:ny, 0:nx]

    valid_ij = np.column_stack([ii[valid], jj[valid]])
    invalid_ij = np.column_stack([ii[invalid], jj[invalid]])

    valid_x = xf[valid]
    valid_y = yf[valid]

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

    rbf_x = RBFInterpolator(valid_ij_sub, valid_x_sub, kernel='thin_plate_spline')
    rbf_y = RBFInterpolator(valid_ij_sub, valid_y_sub, kernel='thin_plate_spline')

    xf[invalid] = rbf_x(invalid_ij)
    yf[invalid] = rbf_y(invalid_ij)

    return xf, yf


def convert_delft3d_to_parcels(input_file, output_file, plot_grid=True):
    print(f"Reading {input_file}...")
    ds = xr.open_dataset(input_file)
    xcor = ds['XCOR'].values
    ycor = ds['YCOR'].values
    inactive_corners = (xcor == 0.0) & (ycor == 0.0)
    xcor[inactive_corners] = np.nan
    ycor[inactive_corners] = np.nan
    xcor_filled, ycor_filled = fill_grid_coords(xcor, ycor)
    if plot_grid:
        plt.scatter(xcor_filled, ycor_filled)
        plt.scatter(xcor, ycor, color="red", marker="x")
        plt.show()
    xcor_filled = xcor_filled.T
    ycor_filled = ycor_filled.T
    zk_centers = ds['ZK_LYR'].values
    time_data = ds['time'].values
    u_data = ds['U1'].values
    u_data = np.nan_to_num(u_data, nan=0.0)
    v_data = ds['V1'].values
    v_data = np.nan_to_num(v_data, nan=0.0)
    w_data = ds['WPHY'].values
    w_data = np.nan_to_num(w_data, nan=0.0)
    u_data = np.transpose(u_data, (0, 1, 3, 2))
    v_data = np.transpose(v_data, (0, 1, 3, 2))
    w_data = np.transpose(w_data, (0, 1, 3, 2))
    assert u_data.shape == v_data.shape == w_data.shape, \
        (f"Shape mismatch! U={u_data.shape}, V={v_data.shape}, W={w_data.shape}. "
         f"If MC!=M or NC!=N, padding is needed — see comments in code.")

    print("  Building output dataset...")
    out = xr.Dataset(
        {
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
    print(f"  Writing {output_file}...")
    encoding = {var: {'zlib': True, 'complevel': 4}
                for var in ['U', 'V', 'W']}
    out.to_netcdf(output_file, encoding=encoding)
    ds.close()
    return out


def rotate_velocity(u, v, alpha):
    u = np.asarray(u).astype(np.float64)
    v = np.asarray(v).astype(np.float64)
    alpha = np.asarray(alpha).astype(np.float64)

    u[u == -999.0] = np.nan
    v[v == -999.0] = np.nan
    alpha[alpha == 0.0] = np.nan

    alpha = np.radians(alpha)
    u_n = u * np.cos(alpha) - v * np.sin(alpha)
    v_e = v * np.cos(alpha) + u * np.sin(alpha)

    return np.concatenate((u_n, v_e), axis=2)
