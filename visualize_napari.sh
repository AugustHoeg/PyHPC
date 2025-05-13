#!/bin/bash

OME_ZARR_FILE="ome_array.zarr"

echo "INFO: Starting napari visualization of $OME_ZARR_FILE"

# check if napari is installed
if ! command -v napari &> /dev/null
then
    echo "ERROR: napari is not installed. Please install it first."
    exit 1
fi

napari --plugin napari-ome-zarr $OME_ZARR_FILE