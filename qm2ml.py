import numpy as np
import random
from itertools import islice

def gau2ml(set_size, step, input_dir, output_dir, perm):
    energies = np.empty(shape=[set_size])
    errors = []

    # write .txt files
    coord_file = open(f"./{output_dir}/coords.txt", "w")
    energy_file = open(f"./{output_dir}/energies.txt", "w")
    force_file = open(f"./{output_dir}/forces.txt", "w")
    error_file = open(f"./{output_dir}/errors.txt", "w")
    #charge_file = open(f"./{output_dir}/charges.txt", "w")

    # get atom count
    with open(f"./nuclear_charges.txt", "r") as nuclear_charge_file:
        n_atom = len(nuclear_charge_file.readlines())

    # read in all symmetry equivalent atoms/groups
    with open(f"./permutations.txt", "r") as perm_file:
        n_perm_grp = int(perm_file.readline())
        n_atm_perm = np.empty(shape=[n_perm_grp], dtype=int)
        for i_perm in range(0,n_perm_grp):
            n_atm_perm[i_perm] = int(perm_file.readline())
            perm_atm = np.empty(shape=[n_perm_grp, n_atm_perm[i_perm]], dtype=int)
            for i_atm in range(0,n_atm_perm[i_perm]):
                perm_atm[i_perm,i_atm] = int(perm_file.readline()) -1
                if perm_atm[i_perm,i_atm] > n_atom:
                    print("permutation atom out of range")
                    exit()
        perm_file.close()

    print(n_perm_grp)
    print(n_atm_perm[0])
    print(perm_atm[0][:])
    #exit()

    # set up arrays
    coord = np.empty(shape=[set_size, n_atom, 3])
    force = np.empty(shape=[set_size, n_atom, 3])
    #charge = np.empty(shape=[set_size, n_atom])

    # loop over all Gaussian files, extract energies, forces and coordinates
    for i_file in range(set_size):
        if ((i_file-1) % step) == 0:
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

            # convert to kcal/mol and print to energy.txt file
            print(energies[i_file] * 627.509608, file=energy_file)

            # read atomic coordinates
            for i_atom, atom in enumerate(coord_block):
                coord[i_file, i_atom] = atom.strip('\n').split()[-3:]

            # read atomic forces
            for i_atom, atom in enumerate(force_block):
                force[i_file, i_atom] = atom.strip('\n').split()[-3:]

            # swap symmetrically equivalent atoms
            if perm:

                # loop over symmetry groups here
                for i_perm in range(n_perm_grp):

                    # perform 10 swap moves
                    for i_swap in range(10):

                        # randomly select groups
                        rand_old = perm_atm[i_perm][random.randint(0,n_atm_perm[0]-1)]
                        rand_new = perm_atm[i_perm][random.randint(0,n_atm_perm[0]-1)]

                        # swap and save new coordinates
                        temp = np.copy(coord[i_file, rand_old])
                        coord[i_file, rand_old] = coord[i_file, rand_new]
                        coord[i_file, rand_new] = temp

                        # swap save new forces
                        temp = np.copy(force[i_file, rand_old])
                        force[i_file, rand_old] = force[i_file, rand_new]
                        force[i_file, rand_new] = temp

            for i_atom, atom in enumerate(coord_block):
                print(*coord[i_file, i_atom], file=coord_file)

            for i_atom, atom in enumerate(force_block):
                print(*force[i_file, i_atom]*627.509608/0.529177, file=force_file)
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

