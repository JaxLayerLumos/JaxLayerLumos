#!/bin/bash

module purge
module load lumerical/2023R1.2

export PATH=/ihome/crc/install/lumerical/2023R1_2/lumerical/v231/bin:$PATH
export PYTHONPATH=/ihome/crc/install/lumerical/2023R1_2/lumerical/v231/api/python:$PYTHONPATH

echo $PATH
echo $PYTHONPATH

which fdtd-solutions
echo $''

export QT_QPA_PLATFORM=offscreen
