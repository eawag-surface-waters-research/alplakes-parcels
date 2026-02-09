# -*- coding: utf-8 -*-
import os
import sys
import json
import parcels
import argparse
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore', message='.*where.*out.*')

import functions

def main(run_id, file, model_type, particles, plot=False, grid=False):
    fieldset_loaders = {
        "delft3d-flow": functions.load_delf3d_fieldset,
        "alplakes-mitgcm": functions.load_alplakes_mitgcm_fieldset
    }

    run_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "..")), "runs", run_id)
    print("Creating run folder {}".format(run_folder))
    os.makedirs(run_folder, exist_ok=True)

    if model_type == "delft3d-flow":
        converted_file = os.path.join(run_folder, os.path.basename(file).replace(".nc", "_parcels.nc"))
        if not os.path.isfile(converted_file):
            functions.convert_delft3d_to_parcels(file, converted_file, plot_grid=grid)
        file = converted_file

    if model_type in fieldset_loaders:
        fieldset = fieldset_loaders[model_type](file)
    else:
        raise ValueError("Unknown model type {}".format(model_type))

    p = np.array(particles["data"])
    time = np.array(p[:, 0], dtype='datetime64')

    pest = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.JITParticle,
        time=time,
        lon=p[:, 1],
        lat=p[:, 2],
        depth=p[:, 3]
    )

    particle_file = os.path.join(run_folder, 'particle_tracks.zarr')
    if not os.path.exists(particle_file):
        output_file = pest.ParticleFile(name=particle_file, outputdt=timedelta(seconds=particles["outputdt"]))
        pest.execute(
             parcels.AdvectionRK4_3D,
             runtime=timedelta(seconds=particles["runtime"]),
             dt=timedelta(seconds=particles["dt"]),
             output_file=output_file
         )
    else:
        print("Particle tracks file {} already exists".format(particle_file))

    if plot:
        print("Plot results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="Config file name", required=True)
    parser.add_argument('--plot', '-p', help="Plot results", action='store_true')
    parser.add_argument('--grid', '-g', help="Plot Delft3D grid interpolation", action='store_true')
    args = parser.parse_args()
    folder = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), ".."))
    config_file = os.path.join(folder, "config", args.config + ".json")
    print("Loading config file {}".format(config_file))
    with open(config_file, 'r') as f:
        data = json.load(f)
    main(args.config, data["file"], data["model_type"], data["particles"], plot=args.plot, grid=args.grid)
