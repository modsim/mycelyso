#!/bin/sh
echo "Downloading example datasets from zenodo ( https://doi.org/10.5281/zenodo.376281 ) ..."
wget https://zenodo.org/record/376281/files/S_lividans_TK24_Complex_Medium_nd046_138.ome.tiff
echo "Verifying file integrity ..."
echo "25f6b6b06d707b0d72cf34ed3a17bbe7 S_lividans_TK24_Complex_Medium_nd046_138.ome.tiff" | md5sum -c -
