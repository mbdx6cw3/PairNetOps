from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmtools import integrators
import numpy as np
import output, read_inputs, os, shutil
from network import Network
import tensorflow as tf

def setup(pairfenet, ani, plat):

    input_dir = "md_input"
    isExist = os.path.exists(input_dir)
    if not isExist:
        print("Error - no input files detected")
        exit()
    md_params = read_inputs.md(f"{input_dir}/md_params.txt")

    output_dir = "md_output"
    isExist = os.path.exists(output_dir)
    if isExist:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    shutil.copy2(f"./{input_dir}/md_params.txt", f"./{output_dir}/")

    temp = md_params["temp"]
    ts = md_params["ts"]
    coll_freq = md_params["coll_freq"]
    gro = GromacsGroFile(f"{input_dir}/input.gro")
    masses = [12,12,12,16,12,12,12,12,16,16,1,1,1,1,1,1]
    pdb = PDBFile(f"./{input_dir}/input.pdb")
    n_atoms = len(pdb.getPositions())
    system = System()
    for j in range(n_atoms):
        system.addParticle(masses[j])
    force = CustomExternalForce("-fx*x-fy*y-fz*z")
    system.addForce(force)
    force.addPerParticleParameter("fx")
    force.addPerParticleParameter("fy")
    force.addPerParticleParameter("fz")
    for j in range(n_atoms):
        force.addParticle(j, (0, 0, 0))
    integrator = LangevinMiddleIntegrator(temp*kelvin, coll_freq / picosecond, ts*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    return simulation, output_dir, md_params, gro, force

def MD(simulation, pairfenet, output_dir, md_params, gro, force):
    input_dir = "trained_model"
    n_steps = md_params["n_steps"]
    n_atoms = len(gro.getPositions())
    prescale = np.loadtxt(f"./{input_dir}/prescale.txt",dtype=np.float32).reshape(-1)
    atoms = np.loadtxt(f"./{input_dir}/atoms.txt",dtype=np.float32).reshape(-1)
    ann_params = read_inputs.ann(f"./{input_dir}/ann_params.txt")
    shutil.copy2(f"./{input_dir}/ann_params.txt", f"./{output_dir}")
    mol = read_inputs.Molecule()
    network = Network(mol)
    model = network.build(len(atoms), ann_params, prescale)
    model.summary()
    model.load_weights(f"./{input_dir}/best_ever_model")

    simulation.reporters.append(StateDataReporter(
        f"./{output_dir}/openmm.csv", 10,  # 10000,
        step=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True))

    # loop through total number of timesteps
    for i in range(n_steps):
        coords = simulation.context.getState(getPositions=True). \
            getPositions(asNumpy=True).in_units_of(angstrom)
        prediction = model.predict([np.reshape(coords, (1, -1, 3)),
                            np.reshape(atoms,(1, -1))])
        forces = prediction[0] * kilocalories_per_mole / angstrom
        forces = np.reshape(forces, (-1, 3))
        for j in range(n_atoms):
            force.setParticleParameters(j, j, forces[j])
        force.updateParametersInContext(simulation.context)

        simulation.step(1)

    return None

