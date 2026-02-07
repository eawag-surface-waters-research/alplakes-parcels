# -*- coding: utf-8 -*-
import os
import json
import time
import requests
import parcels
import argparse
import numpy as np
import importlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import functions

def main(params):

    fieldset_loaders = {
        "delft3d-flow": functions.load_delf3d_fieldset
    }

    file = params['file']

    if params["type"] == "delft3d-flow":
        converted_file = params["file"].replace(".nc", "_parcels.nc")
        if not os.path.isfile(converted_file):
            functions.convert_delft3d_to_parcels(file, converted_file)
        file = converted_file

    if params["type"] in fieldset_loaders:
        fieldset = fieldset_loaders[params["type"]](file)

    pset = parcels.ParticleSet(
         fieldset=fieldset,
         pclass=parcels.JITParticle,
         lon=np.array([531549]),       # x in meters
         lat=np.array([145030]),       # y in meters
         depth=np.array([-1]),     # depth (matching ZK_LYR convention)
     )

    output_file = pset.ParticleFile(name='particle_tracks.zarr', outputdt=timedelta(hours=1))
    pset.execute(
         parcels.AdvectionRK4_3D,
         runtime=timedelta(minutes=20),
         dt=timedelta(minutes=5),
         output_file=output_file
     )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', help="Path to input file", type=str, required=True)
    parser.add_argument('--type', '-t', help="Type of model", choices=['delft3d-flow'], required=True)
    args = parser.parse_args()
    main(vars(args))