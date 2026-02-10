# Alplakes Parcels

Lagrangian particle tracking simulations for Alpine lakes using [OceanParcels](https://oceanparcels.org/). Simulates water particle trajectories in 3D velocity fields from hydrodynamic models (Delft3D-FLOW and MITGCM).

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate parcels
```

## Usage

```bash
python src/main.py --config <config_name> [--plot] [--grid]
```

### Arguments

- `--config, -c` (required): Configuration file name (without `.json` extension) from the `config/` directory.
- `--plot, -p` (optional): Generate an animation of particle distributions.
- `--grid, -g` (optional): Visualize the Delft3D grid interpolation.

### Examples

```bash
python src/main.py --config geneva
python src/main.py --config geneva --plot
python src/main.py --config geneva --grid --plot
```

## Configuration

Configuration files are stored in `config/` as JSON. See `config/example.json` for the template.

```json
{
  "file": "/path/to/hydrodynamic/data.nc",
  "model_type": "delft3d-flow",
  "particles": {
    "outputdt": 3600,
    "runtime": 129600,
    "dt": 60,
    "data": {
      "type": "circle",
      "center": [x, y],
      "min_z": 0.5,
      "max_z": 1,
      "number": 10,
      "radius": 500,
      "time": "2025-12-14T06:00:00.00"
    }
  }
}
```


## Output

Particle trajectories are saved as Zarr archives in `runs/<config_name>/particle_tracks.zarr`. When `--plot` is enabled, frame images and a GIF animation are generated in the same directory.