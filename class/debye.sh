#!/bin/bash
#SBATCH --job-name=debye_kwhite
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --partition=amilan
#SBATCH --output=debye_kwhite.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7GB
#SBATCH --error=debye_kwhite.err
##SBATCH --mail-type=ALL
##SBATCH --mail-user=keith.white@colorado.edu

# Import modules
module purge
module load python
module load anaconda

# Activate the environment
source activate debyecalc

# Change to the hard-coded directory
cd /projects/kewh5868/debye/
# cd "$(dirname "$0")"

# Define the filenames
# xyzFilename="pbi2_dmso_01.xyz"  # Replace with the filename of your .xyz file
# xyzFilename="geo-opt_GGA-PBE_explicit-solvent_implicit-solvent-COSMO_mono_PbX3L3_LDMSO_XI_isoMER.xyz"
# xyzFilename="In37P20_SC_dg_03_0m_PA.xyz"
xyzFilename="Pb2X4L6_LDMSO_XI_iso4.xyz"

QInputFilename="pbi2_0p005m_dmso_merged_20240530_114828.iq"  # Replace with the filename of your .iq file

# Define the inputs
xyzPath="./inputs/xyz/$xyzFilename"
QInput="./inputs/qinput/$QInputFilename"
# adps="example_adps.txt"  # Replace with the path to your ADPs file
# suppress_plots="--suppress_plots"

# Run the Python script with arguments
python3 debye.py "$xyzPath" "$QInput"
# python3 debye.py $xyzPath $QInput $suppress_plots --adps $adps