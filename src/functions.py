import os
import parcels
import random
from PIL import Image
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

def load_delf3d_fieldset(data_file, grid="a"):
    print("Loading delf3d fieldset: {}".format(data_file))
    if grid == "c":
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
            gridindexingtype="nemo",
            mesh='flat',
        )
    elif grid == "a":
        filenames = {'U': data_file, 'V': data_file, 'W': data_file}
        variables = {'U': 'U', 'V': 'V', 'W': 'W'}
        dims = {
            'lon': 'glamf',
            'lat': 'gphif',
            'depth': 'depth',
            'time': 'time',
        }
        dimensions = {'U': dims, 'V': dims, 'W': dims}
        fieldset = parcels.FieldSet.from_netcdf(
            filenames, variables, dimensions,
            mesh='flat',
        )
    else:
        raise ValueError("Unrecognised grid type")
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


def convert_delft3d_to_parcels_c(input_file, output_file, plot_grid=False):
    print(f"Reading {input_file}...")
    ds = xr.open_dataset(input_file)
    xcor = ds['XCOR'].values
    ycor = ds['YCOR'].values
    inactive_corners = (xcor == 0.0) & (ycor == 0.0)
    xcor[inactive_corners] = np.nan
    ycor[inactive_corners] = np.nan
    xcor = np.pad(xcor, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
    ycor = np.pad(ycor, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
    xcor_filled, ycor_filled = fill_grid_coords(xcor, ycor)
    if plot_grid:
        plt.scatter(xcor_filled, ycor_filled)
        plt.scatter(xcor, ycor, color="red", marker="x")
        plt.show()
    xcor_filled = xcor_filled.T
    ycor_filled = ycor_filled.T
    depth_mask = ds['ZK_LYR'].values <= 0
    zk_centers = ds['ZK_LYR'].values[depth_mask] * -1
    zk_centers = zk_centers[::-1]
    time_data = ds['time'].values
    u_data = ds['U1'].values[:, depth_mask, :, :][:, ::-1, :, :]
    u_data[u_data == -999.0] = 0.0
    u_data = np.nan_to_num(u_data, nan=0.0)
    v_data = ds['V1'].values[:, depth_mask, :, :][:, ::-1, :, :]
    v_data[v_data == -999.0] = 0.0
    v_data = np.nan_to_num(v_data, nan=0.0)
    w_data = ds['WPHY'].values[:, depth_mask, :, :][:, ::-1, :, :]
    w_data[w_data == -999.0] = 0.0
    w_data = w_data * -1 # NEEDS TO BE CHECKED
    w_data = np.nan_to_num(w_data, nan=0.0)
    u_data = np.transpose(u_data, (0, 1, 3, 2))
    v_data = np.transpose(v_data, (0, 1, 3, 2))
    w_data = np.transpose(w_data, (0, 1, 3, 2))
    pad_width = ((0, 0), (0, 0), (1, 1), (1, 1))
    u_data = np.pad(u_data, pad_width, mode='constant', constant_values=0.0)
    v_data = np.pad(v_data, pad_width, mode='constant', constant_values=0.0)
    w_data = np.pad(w_data, pad_width, mode='constant', constant_values=0.0)

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
                      {'units': 'm', 'positive': 'down',
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


def convert_delft3d_to_parcels_a(input_file, output_file, plot_grid=False):
    print(f"Reading {input_file}...")
    ds = xr.open_dataset(input_file)
    xcor = ds['XZ'].values
    ycor = ds['YZ'].values
    inactive_corners = (xcor == 0.0) & (ycor == 0.0)
    xcor[inactive_corners] = np.nan
    ycor[inactive_corners] = np.nan
    xcor = np.pad(xcor, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
    ycor = np.pad(ycor, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
    xcor_filled, ycor_filled = fill_grid_coords(xcor, ycor)
    if plot_grid:
        plt.scatter(xcor_filled, ycor_filled)
        plt.scatter(xcor, ycor, color="red", marker="x")
        plt.show()
    xcor_filled = xcor_filled.T
    ycor_filled = ycor_filled.T
    depth_mask = ds['ZK_LYR'].values <= 0
    zk_centers = ds['ZK_LYR'].values[depth_mask] * -1
    zk_centers = zk_centers[::-1]
    time_data = ds['time'].values

    u_data = ds['U1'].values[:, depth_mask, :, :][:, ::-1, :, :]
    u_data[u_data == -999.0] = np.nan
    u_data = np.nan_to_num(u_data, nan=0.0)

    v_data = ds['V1'].values[:, depth_mask, :, :][:, ::-1, :, :]
    v_data[v_data == -999.0] = np.nan
    v_data = np.nan_to_num(v_data, nan=0.0)

    # Value at cell center
    u_data = 0.5 * (u_data[:, :, :, :-1] + u_data[:, :, :, 1:])
    u_data = np.pad(u_data, ((0, 0), (0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0.0)
    v_data = 0.5 * (v_data[:, :, :-1, :] + v_data[:, :, 1:, :])
    v_data = np.pad(v_data, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0.0)

    # Rotate to global
    u_data, v_data = rotate_velocity(u_data, v_data, ds["ALFAS"])

    w_data = ds['WPHY'].values[:, depth_mask, :, :][:, ::-1, :, :]
    w_data[w_data == -999.0] = 0.0
    w_data = w_data * -1 # NEEDS TO BE CHECKED
    w_data = np.nan_to_num(w_data, nan=0.0)

    u_data = np.transpose(u_data, (0, 1, 3, 2))
    v_data = np.transpose(v_data, (0, 1, 3, 2))
    w_data = np.transpose(w_data, (0, 1, 3, 2))
    pad_width = ((0, 0), (0, 0), (1, 1), (1, 1))
    u_data = np.pad(u_data, pad_width, mode='constant', constant_values=0.0)
    v_data = np.pad(v_data, pad_width, mode='constant', constant_values=0.0)
    w_data = np.pad(w_data, pad_width, mode='constant', constant_values=0.0)

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
                       'long_name': 'U-velocity'}
            ),
            'V': xr.DataArray(
                v_data.astype(np.float32),
                dims=['time', 'depth', 'y', 'x'],
                attrs={'units': 'm/s',
                       'long_name': 'V-velocity'}
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
                       'long_name': 'X-coordinate of cell centers'}
            ),
            'gphif': xr.DataArray(
                ycor_filled.astype(np.float64),
                dims=['y', 'x'],
                attrs={'units': 'm',
                       'long_name': 'Y-coordinate of cell centers'}
            ),
        },
        coords={
            'time': ('time', time_data),
            'depth': ('depth', zk_centers.astype(np.float32),
                      {'units': 'm', 'positive': 'down',
                       'long_name': 'Depth at layer centres'}),
        },
        attrs={
            'Conventions': 'CF-1.6',
            'source': 'Converted from Delft3D-FLOW z-layer output for OceanParcels',
            'original_file': input_file,
            'grid_type': 'Arakawa A-grid',
            'coordinate_system': 'projected (meters)',
            'parcels_usage': 'Use FieldSet.from_netcdf() with mesh="flat"',
            'note': ('glamf/gphif = cell center coordinates from XZ/YZ. '
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

    return u_n, v_e


def random_point_in_circle(x, y, radius):
    r = radius * random.random()
    theta = 2 * np.pi * random.random()
    return x + r * np.cos(theta), y + r * np.sin(theta)


def random_points_in_circle(x_center, y_center, radius, n, min_z, max_z):
    x = []
    y = []
    d = []
    for i in range(n):
        d.append(random.uniform(min_z, max_z))
        xx, yy = random_point_in_circle(x_center, y_center, radius)
        x.append(xx)
        y.append(yy)
    return np.array(x), np.array(y), np.array(d)


def load_circle_particles(params, fieldset):
    lon, lat, depth = random_points_in_circle(params["center"][0], params["center"][1], params["radius"], params["number"], params["min_z"], params["max_z"])
    dt = np.datetime64(params["time"])
    time = np.full(params["number"], dt, dtype=dt.dtype)
    return parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.JITParticle,
        time=time,
        lon=lon,
        lat=lat,
        depth=np.array(depth)
    )


def plot_delft3d_flow(file, particle_file):
    ds = xr.open_dataset(file)
    za = xr.open_zarr(particle_file)
    lines_x, lines_y = delft3d_flow_gridlines(ds)

    za = za.compute()
    n_obs = za.sizes['obs']

    output_dir = os.path.join(os.path.dirname(particle_file), "plots", "frames")
    os.makedirs(output_dir, exist_ok=True)

    # Pre-compute global color limits across all timesteps
    all_quantiles_lo = []
    all_quantiles_hi = []
    for t in range(n_obs):
        lons = za.lon[:, t].values
        lats = za.lat[:, t].values
        mask = ~(np.isnan(lons) | np.isnan(lats))
        lons, lats = lons[mask], lats[mask]
        if len(lons) < 2:
            continue
        H, _, _ = np.histogram2d(lons, lats, bins=50)
        H_plot = H.T.astype(float)
        H_plot[H_plot < 1] = np.nan
        all_quantiles_lo.append(np.nanquantile(H_plot, 0.25))
        all_quantiles_hi.append(np.nanquantile(H_plot, 0.75))

    vmin_global = np.nanmedian(all_quantiles_lo) if all_quantiles_lo else 0
    vmax_global = np.nanmedian(all_quantiles_hi) if all_quantiles_hi else 1

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(lines_x, lines_y, color='#cacccc', linewidth=0.5, zorder=1)
    ax.set_aspect('equal')

    # Create a dummy mappable for a persistent colorbar
    norm = plt.Normalize(vmin=vmin_global, vmax=vmax_global)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, extend='max')
    cbar.set_ticks([])
    cbar.set_label(label=r'$\mathrm{Particle \ concentration\ [-]}$')

    mesh = [None]
    frame_paths = []

    for t in range(n_obs - 1):
        if mesh[0] is not None:
            mesh[0].remove()
            mesh[0] = None

        lons = za.lon[:, t].values
        lats = za.lat[:, t].values
        mask = ~(np.isnan(lons) | np.isnan(lats))
        lons, lats = lons[mask], lats[mask]

        if len(lons) < 2:
            ax.set_title(f'Time: {t} (no data)')
        else:
            bins = 50
            H, xedges, yedges = np.histogram2d(lons, lats, bins=bins)
            H_plot = H.T.astype(float)
            H_plot[H_plot < 1] = np.nan

            X, Y = np.meshgrid(xedges, yedges)
            mesh[0] = ax.pcolormesh(X, Y, H_plot,
                                    vmin=vmin_global,
                                    vmax=vmax_global,
                                    cmap='viridis', zorder=2)

            time_val = pd.Timestamp(za.time.values[0, t])
            ax.set_title(f'{time_val:%Y-%m-%d %H:%M}')

        frame_path = os.path.join(output_dir, f'frame_{t:04d}.png')
        fig.savefig(frame_path, dpi=300, bbox_inches='tight')
        frame_paths.append(frame_path)
        print(f'\rSaved frame {t + 1}/{n_obs}', end='', flush=True)

    plt.close(fig)
    print()

    # Build GIF from saved frames
    images = [Image.open(p) for p in frame_paths]
    duration_ms = int(1000 / 5)
    images[0].save(os.path.join(os.path.dirname(output_dir), "animation.gif"), save_all=True, append_images=images[1:],
                   duration=duration_ms, loop=0)

    ds.close()
    za.close()


def delft3d_flow_gridlines(ds):
    x = ds["XCOR"].values
    y = ds["YCOR"].values
    mask = (x != 0) | (y != 0)
    x_fixed = x.copy()
    y_fixed = y.copy()
    x_fixed[~mask] = np.nan
    y_fixed[~mask] = np.nan

    lines_x = []
    lines_y = []

    # Horizontal lines (along j)
    for i in range(x.shape[0]):
        lines_x.extend(x_fixed[i, :].tolist() + [np.nan])
        lines_y.extend(y_fixed[i, :].tolist() + [np.nan])

    # Vertical lines (along i)
    for j in range(x.shape[1]):
        lines_x.extend(x_fixed[:, j].tolist() + [np.nan])
        lines_y.extend(y_fixed[:, j].tolist() + [np.nan])

    return lines_x, lines_y
