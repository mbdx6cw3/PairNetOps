from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmplumed import PlumedForce
from openmmtools import integrators
from sys import stdout
import numpy as np
import output, plumed

def EP(input_dir, output_dir, md_params):
    '''setup system for MD with OpenMM
    Vars:
    '''
    # lots of this should be removed as it is duplicated in def MLP
    temp = md_params["temp"]
    ts = md_params["ts"]
    n_steps = md_params["n_steps"]
    print_steps = md_params["print_steps"]
    bias = md_params["bias"]
    ensemble = md_params["ensemble"]
    thermostat = md_params["thermostat"]
    platform = Platform.getPlatformByName('OpenCL')
    if bias:
        plumed_file = open(f"{input_dir}/plumed.dat", "r")
        plumed_script = plumed_file.read()
    gro = GromacsGroFile(f"{input_dir}/input.gro")
    top = GromacsTopFile(f"{input_dir}/input.top",
        periodicBoxVectors=gro.getPeriodicBoxVectors())
    vectors = gro.getUnitCellDimensions().value_in_unit(nanometer)
    n_atoms = len(gro.getPositions())
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer)
    # define ensemble, thermostat and integrator
    if ensemble == "nve":
        integrator = VerletIntegrator(ts*picoseconds)
    elif ensemble == "nvt":
        if thermostat == "nose_hoover":
            integrator = NoseHooverIntegrator(temp*kelvin,
                1 / picosecond, ts*picoseconds)
        elif thermostat == "langevin":
            integrator = LangevinMiddleIntegrator(temp*kelvin,
                1 / picoseconds, ts*picoseconds)
    if bias:
        system.addForce(PlumedForce(plumed_script))
    simulation = Simulation(top.topology, system, integrator, platform)
    simulation.context.setPositions(gro.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(StateDataReporter(f"./{output_dir}/openmm.csv",
        reportInterval=1000, step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, separator=" "))

    f1 = open(f"./{output_dir}/coords.txt", 'w')
    f2 = open(f"./{output_dir}/forces.txt", 'w')
    f3 = open(f"./{output_dir}/velocities.txt", 'w')
    f4 = open(f"./{output_dir}/energies.txt", 'w')
    for i in range(n_steps):
        if (i % print_steps) == 0 or i == 0:
            time = simulation.context.getState().getTime()
            coords = simulation.context.getState(getPositions=True).\
                getPositions(asNumpy=True)
            velocities = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True)
            forces = simulation.context.getState(getForces=True).\
                getForces(asNumpy=True)
            state = simulation.context.getState(getEnergy=True)
            PE = state.getPotentialEnergy() / kilojoule_per_mole
            output.gro(n_atoms, vectors, time/picoseconds, coords/nanometer,
                       gro.atomNames, output_dir, "output")
            np.savetxt(f1, coords[:n_atoms])
            np.savetxt(f2, forces[:n_atoms])
            np.savetxt(f3, velocities[:n_atoms])
            f4.write(f"{PE}\n")
        simulation.step(1)
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    return None


def MLP(model, input_dir, output_dir, md_params, atoms):

    # setup MD parameters
    temp = md_params["temp"]
    ts = md_params["ts"]
    n_steps = md_params["n_steps"]
    print_steps = md_params["print_steps"]
    bias = md_params["bias"]
    platform = Platform.getPlatformByName('OpenCL')
    if bias:
        plumed_file = open(f"{input_dir}/plumed.dat", "r")
        plumed_script = plumed_file.read()

    # ANI is available with openmm-ml also...

    # read gromacs input files
    gro = GromacsGroFile(f"{input_dir}/input.gro")
    top = GromacsTopFile(f"{input_dir}/input.top",
        periodicBoxVectors=gro.getPeriodicBoxVectors())
    n_atoms = len(gro.getPositions())
    vectors = gro.getUnitCellDimensions().value_in_unit(nanometer)

    # create a system of n_atoms with masses
    system = top.createSystem(nonbondedMethod=PME,nonbondedCutoff=1*nanometer)

    # define custom external force and set initial forces to zero
    force = CustomExternalForce("-fx*x-fy*y-fz*z")
    system.addForce(force)
    force.addPerParticleParameter("fx")
    force.addPerParticleParameter("fy")
    force.addPerParticleParameter("fz")
    for j in range(n_atoms):
        force.addParticle(j, (0, 0, 0))

    # define ensemble, thermostat and integrator
    if ensemble == "nve":
        integrator = VerletIntegrator(ts*picoseconds)
    elif ensemble == "nvt":
        if thermostat == "nose_hoover":
            integrator = NoseHooverIntegrator(temp*kelvin,
                1 / picosecond, ts*picoseconds)
        elif thermostat == "langevin":
            integrator = LangevinMiddleIntegrator(temp*kelvin,
                1 / picoseconds, ts*picoseconds)
    if bias:
        system.addForce(PlumedForce(plumed_script))
    simulation = Simulation(top.topology, system, integrator, platform)
    simulation.context.setPositions(gro.positions)
    simulation.reporters.append(StateDataReporter(f"./{output_dir}/openmm.csv",
        reportInterval=1000, step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, separator=" "))

    f1 = open(f"./{output_dir}/coords.txt", 'w')
    f2 = open(f"./{output_dir}/forces.txt", 'w')
    f3 = open(f"./{output_dir}/velocities.txt", 'w')
    f4 = open(f"./{output_dir}/energies.txt", 'w')

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
        if (i % print_steps) == 0 or i == 0:
            time = simulation.context.getState().getTime()
            velocities = simulation.context.getState(getVelocities=True). \
                getVelocities(asNumpy=True)
            PE = prediction[2][0][0]
            output.gro(n_atoms, vectors, time / picoseconds,
                coords / nanometer, gro.atomNames, output_dir, "output")
            np.savetxt(f1, coords[:n_atoms])
            np.savetxt(f2, forces[:n_atoms])
            np.savetxt(f3, velocities[:n_atoms])
            f4.write(f"{PE}\n")
        simulation.step(1)
    f1.close()
    f2.close()
    f3.close()
    f4.close()

