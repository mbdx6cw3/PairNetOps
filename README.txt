PairNetOps
--------------------------------------------------------------------------------
This package can be used for the following tasks:
1. Prepare input files for MD simulations.
2. Run MD simulations using and empirical or machine learned potential.
3. Analyse MD simulation trajectories.
4. Prepare input from QM calculations from MD simulation output.
5. Convert and Analyse QM calculation data.
6. Train and Test ANNs
7. Query external datasets.

Environment Setup
--------------------------------------------------------------------------------
Before running a job, you first need to correctly set up your environment to
run the code. Create a Conda environment using the following commands:

1)  Load the up-to-date version of Anaconda, e.g.
    >   module load apps/binapps/anaconda3/2022.10

2)  Load tools that allow communication with the outside world,
    >   module load tools/env/proxy2

3)  Create a Conda environment for running PairNetOps.
    >   conda create -n pair-net-ops python==3.9.13

4)  Activate the Conda environment.
    >   source activate pair-net-ops

If you are only using CPUs only one more command is needed.

5)  Install some required Python packages using Pip.
    >   pip install matplotlib numpy tensorflow==2.12.0 plumed

6)  Install OpenMM, OpenMM Plumed plugin and OpenMM-ML (ANI)
    >   conda install -c conda-forge openmm openmm-plumed openmm-ml openmmtools


If you are using with GPUs some additional steps are required...

7)  Install cudatoolkit
    >   conda install -c conda-forge cudatoolkit=11.8.0

8)  Install tensorflow
    >   pip install --isolated nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.0

9)  Extra steps to fix bug in tensorflow 2.11 and 2.12.
Hopefully not needed in TF 2.13!!
    >   conda install -c nvidia cuda-nvcc --yes
    >   mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/
    >   cp -p $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/

9) Verify the setup:
    >   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

--------------------------------------------------------------------------------
The code is located in:     /mnt/iusers01/rb01/mbdx6cw3/bin/PairNetOps

Using PairNetOps
--------------------------------------------------------------------------------
Input flag options:
            [1] - Run MD simulation.
            [2] - Analyse MD output.
            [3] - Convert MD output into QM input.
            [4] - Analyse QM output.
            [5] - Convert QM output into ML input.
            [6] - Train or test an ANN.
            [7] - Query an external dataset.

This tool is used to query the MD17/rMD17 datasets and calculate pairwise
distance, bend angle and dihedral angle distributions between selected atoms.

Recommended to use interactively.

You will need to consult the relevant mapping.dat file for connectivity.

Outputs: .png image and output.csv file.
--------------------------------------------------------------------------------

Can either be run interactively (not recommended for running MD or Training an ANN)
or by submitting to queue.