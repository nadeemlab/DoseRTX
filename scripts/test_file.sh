#!/bin/bash
d="$//pisidsmph/NadeemLab/Interns/Navdeep/msk-manual-3d-dvh-beamlet-dense-separate-ptv"
 
[ "$d" == "" ] && { echo "Usage: $0 directory"; exit 1; }
[ -d "${d}" ] &&  echo "Directory $d found." || echo "Directory $d not found."