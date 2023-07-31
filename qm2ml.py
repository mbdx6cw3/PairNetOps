import numpy as np
from itertools import islice

def gau2ml(set_size, input_dir, output_dir):
    energies = np.empty(shape=[set_size])
    errors = []

    # write .txt files
    coord_file = open(f"./{output_dir}/coords.txt", "w")
    energy_file = open(f"./{output_dir}/energies.txt", "w")
    force_file = open(f"./{output_dir}/forces.txt", "w")
    error_file = open(f"./{output_dir}/errors.txt", "w")
    charge_file = open(f"./{output_dir}/charges.txt", "w")

    # get atom count
    with open(f"./nuclear_charges.txt", "r") as nuclear_charge_file:
        n_atom = len(nuclear_charge_file.readlines())

    # set up arrays
    coord = np.empty(shape=[set_size, n_atom, 3])
    force = np.empty(shape=[set_size, n_atom, 3])
    charge = np.empty(shape=[set_size, n_atom])

    # loop over all Gaussian files, extract energies, forces and coordinates
    for i_file in range(set_size):
        normal_term = False
        qm_file = open(f"./{input_dir}/mol_{i_file+1}.out", "r")
        for line in qm_file:
            # extract atomic coordinates
            if "Input orientation:" in line:
                coord_block = list(islice(qm_file, 4+n_atom))[-n_atom:]
            # extract energies
            if "SCF Done:" in line:
                energies[i_file] = (float(line.split()[4]))
            # extract forces
            if "Axes restored to original set" in line:
                force_block = list(islice(qm_file, 4+n_atom))[-n_atom:]
            # extract charges
            if "ESP charges:" in line:
                charge_block = list(islice(qm_file, 1+n_atom))[-n_atom:]
            # assess termination state
            if "Normal termination of Gaussian 09" in line:
                normal_term = True
                break
        # save coordinates to file and
        for i_atom, atom in enumerate(coord_block):
            coord[i_file, i_atom] = atom.strip('\n').split()[-3:]
            print(*coord[i_file, i_atom], file=coord_file)
        # convert to kcal/mol and print to energy.txt file
        print(energies[i_file] * 627.509608, file=energy_file)
        for i_atom, atom in enumerate(force_block):
            # convert to kcal/mol/Angstrom and print to force.txt file
            force[i_file, i_atom] = atom.strip('\n').split()[-3:]
            print(*force[i_file, i_atom]*627.509608/0.529177,
            file=force_file)
        # optional reading of charges
        #for i_atom, atom in enumerate(charge_block):
        #    charge[i_file, i_atom] = atom.strip('\n').split()[-1]
        #    print(charge[i_file, i_atom], file=charge_file)
        if not normal_term:
            errors.append(i_file)
            print(i_file, file=error_file)
        qm_file.close()

    coord_file.close()
    energy_file.close()
    force_file.close()
    error_file.close()

