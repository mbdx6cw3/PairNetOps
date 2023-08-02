# info...
#!/usr/bin/env python
__author__ = ['Christopher D Williams']
__credits__ = ['CDW', 'Jas Kalayan', 'Ismaeel Ramzan',
        'Neil Burton',  'Richard Bryce']
__license__ = 'GPL'
__maintainer__ = 'Christopher D Williams'
__email__ = 'christopher.williams@manchester.ac.uk'
__status__ = 'Development'

def main():
    import qm2ml, analyseQM, query_external, openMM, read_inputs, output,\
        analyseMD
    import os, shutil, sys
    import numpy as np
    from network import Network
    from datetime import datetime
    from keras.models import Model, load_model
    import tensorflow as tf

    # read primary user input
    try:
        input_flag = int(input(""" What would you like to do?
            [1] - Run MD simulation.
            [2] - Analyse MD output.
            [3] - Convert MD output into QM input.
            [4] - Analyse QM output.
            [5] - Convert QM output into ML or MD input.
            [6] - Train or Test an ANN.
            [7] - Query external dataset.
            > """))
    except ValueError:
        print("Invalid Value")
        exit()
    except input_flag > 7:
        print("Invalid Value")
        exit()
    print()

    # determine type of calculation to do
    if input_flag == 1:
        startTime = datetime.now()
        option_flag = int(input("""Run MD simulation.
            [1] - Use an empirical potential.
            [2] - Use a neural network potential.
            > """))

        input_dir1 = "md_input"
        isExist = os.path.exists(input_dir1)
        if not isExist:
            print("Error - no input files detected")
            exit()

        # read in MD input parameters
        md_params = read_inputs.md(f"{input_dir1}/md_params.txt")
        output_dir = "md_output"
        isExist = os.path.exists(output_dir)
        if isExist:
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        shutil.copy2(f"./{input_dir1}/md_params.txt", f"./{output_dir}/")

        # run MD with empirical potential
        if option_flag == 1:
            print("Use an empirical potential.")
            openMM.EP(input_dir1, output_dir, md_params)

        # run MD with a machine-learned potential
        elif option_flag == 2:
            print("Use a neural network potential.")
            input_dir2 = "trained_model"
            isExist = os.path.exists(input_dir2)
            if not isExist:
                print("Error - previously trained model could not be located.")
                exit()
            print("Loading a trained model...")
            prescale = np.loadtxt(f"./{input_dir2}/prescale.txt",
                                  dtype=np.float32).reshape(-1)
            atoms = np.loadtxt(f"./{input_dir2}/atoms.txt",
                                  dtype=np.float32).reshape(-1)
            ann_params = read_inputs.ann(f"./{input_dir2}/ann_params.txt")
            shutil.copy2(f"./{input_dir2}/ann_params.txt", f"./{output_dir}")
            mol = read_inputs.Molecule()
            network = Network(mol)
            model = network.build(len(atoms), ann_params, prescale)
            model.summary()
            model.load_weights(f"./{input_dir2}/best_ever_model")
            openMM.MLP(model, input_dir1, output_dir, md_params, atoms)

        print(datetime.now() - startTime)

    elif input_flag == 2:
        print("Analyse MD output.")

        output_dir = "plots_and_data"
        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)

        option_flag = int(input("""
            [1] - Calculate force S-curve.
            [2] - Calculate force error distribution.
            [3] - Calculate energy correlation.
            [4] - Calculate dihedral angle probability distributions.
            [5] - Calculate 2D free energy surface.
            > """))

        # initiate molecule class for MD dataset
        input_dir1 = "md_output"
        if option_flag == 1 or option_flag == 2 or option_flag == 3 or \
            option_flag == 4:
            while True:
                try:
                    set_size = int(input("Enter the dataset size > "))
                    break
                except ValueError:
                    print("Invalid Value")
            mol1 = read_inputs.Molecule()
            read_inputs.dataset(mol1, input_dir1, set_size, "md")

        # initiate molecule class for QM dataset
        if option_flag == 1 or option_flag == 2 or option_flag == 3:
            input_dir2 = "qm_data"
            mol2 = read_inputs.Molecule()
            read_inputs.dataset(mol2, input_dir2, set_size, "qm")

        if option_flag == 1:
            print("Calculating force S-curve...")
            output.scurve(mol2.forces.flatten(), mol1.forces.flatten(),
                output_dir, "mm_f_scurve")
            np.savetxt(f"./{output_dir}/mm_f_test.dat", np.column_stack((
                mol2.forces.flatten(), mol1.forces.flatten())),
                       delimiter=", ", fmt="%.6f")

        elif option_flag == 2:
            print("Calculating force error distribution...")
            analyseMD.force_MSE_dist(mol2.forces.flatten(),
                mol1.forces.flatten(), output_dir)

        elif option_flag == 3:
            print("Calculating energy correlation with QM...")
            analyseMD.energy_corr(mol2.energies, mol1.energies, output_dir)

        elif option_flag == 4:
            while True:
                try:
                    n_dih = int(input("Enter the number of dihedral angles > "))
                    break
                except ValueError:
                    print("Invalid Value")
                except n_dih > 2:
                    print("Number of dihedral angles can only be 1 or 2")
            CV_list = np.empty(shape=[n_dih, 4], dtype=int)
            for i_dih in range(n_dih):
                atom_indices = input(f"""
                Enter atom indices for dihedral {i_dih+1} separated by spaces:
                e.g. "5 4 6 10"
                Consult mapping.dat for connectivity.
                > """)
                CV_list[i_dih,:] = np.array(atom_indices.split())
            n_bins = int(input("Enter the number of bins > "))
            print("Calculating dihedral angle probability distributions...")
            if n_dih == 1:
                analyseMD.pop1D(mol1, n_bins, CV_list, output_dir, set_size)
            elif n_dih == 2:
                analyseMD.pop2D(mol1, n_bins, CV_list, output_dir, set_size)

        elif option_flag == 5:
            print("Calculating 2D free energy surface...")
            analyseMD.fes2D(input_dir1, output_dir)

    elif input_flag == 3:
        while True:
            try:
                set_size = int(input("Enter the dataset size > "))
                break
            except ValueError:
                print("Invalid Value")
        while True:
            try:
                init = int(input("Enter the initial frame > "))
                break
            except ValueError:
                print("Invalid Value")
        output_dir = "qm_input"
        isExist = os.path.exists(output_dir)
        if isExist:
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        input_dir = "md_output"
        isExist = os.path.exists(input_dir)
        if not isExist:
            print("Error - no input files detected")
            exit()

        mol = read_inputs.Molecule()
        read_inputs.dataset(mol, input_dir, set_size, "md")
        output.write_gau(mol, init, set_size, output_dir)

    elif input_flag == 4:
        print("Analyse QM output.")
        while True:
            try:
                set_size = int(input("Enter the dataset size > "))
                break
            except ValueError:
                print("Invalid Value")
        output_dir = "plots_and_data"
        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)

        input_dir = "qm_data"
        isExist = os.path.exists(input_dir)
        if not isExist:
            print("Error - no input files detected")
            exit()

        # initiate molecule class and parse dataset
        mol = read_inputs.Molecule()
        read_inputs.dataset(mol, input_dir, set_size, "qm")

        option_flag = int(input("""
              [1] - Calculate force and energy probability distributions.
              [2] - Calculate interatomic pairwise force components (q).
              [3] - Calculate energy wrt to geometric variable.
              [4] - Calculate distance matrix RMSD.
              > """))

        if option_flag == 1:
            analyseQM.dist(mol, set_size, output_dir)
        elif option_flag == 2:
            mol.orig_energies = np.copy(mol.energies)
            analyseQM.prescale_e(mol, mol.energies, mol.forces)
            analyseQM.get_pairs(mol, set_size, output_dir)
            recomb_F = analyseQM.get_forces(mol, mol.coords, mol.mat_FE)
            np.savetxt(f"./{output_dir}/recomb_test.dat", np.column_stack((
                mol.forces.flatten(), recomb_F.flatten())), delimiter=" ", fmt="%.6f")
        elif option_flag == 3:
            atom_indices = input("""
                Enter atom indices separated by spaces:
                    e.g. for a distance "0 1"
                    e.g. for an angle "1 2 3 4"
                    e.g. for a dihedral "5 4 6 10"
                    Consult mapping.dat for connectivity.
                > """)
            analyseQM.energy_CV(mol, atom_indices, set_size, output_dir)
        elif option_flag == 4:
            print("Calculating distance matrix RMSD...")
            rmsd_dist = analyseQM.rmsd_dist(mol,set_size)
            print(f"Distance matrix RMSD: {np.mean(rmsd_dist)} Angstrom")

    elif input_flag == 5:

        print("Convert QM output into ML or MD input.")
        option_flag = int(input("""
                     [1] - Convert to ML input.
                     [2] - Convert to MD input (.gro format).
                     > """))
        while True:
            try:
                set_size = int(input("Enter the dataset size > "))
                break
            except ValueError:
                print("Invalid Value")

        if option_flag == 1:
            input_dir = "qm_input"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files detected")
                exit()
            output_dir = "qm_data"
            isExist = os.path.exists(output_dir)
            if not isExist:
                os.makedirs(output_dir)
            qm2ml.gau2ml(set_size, input_dir, output_dir)

        elif option_flag == 2:
            input_dir = "qm_data"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files detected")
                exit()

            # initiate molecule class and parse dataset
            mol = read_inputs.Molecule()
            read_inputs.dataset(mol, input_dir, set_size, "qm")

            output_dir = "md_input"
            isExist = os.path.exists(output_dir)
            if not isExist:
                os.makedirs(output_dir)

            vectors = [2.5, 2.5, 2.5]
            time = 0.0
            mol.coords = mol.coords / 10 # convert to nm
            for item in range(set_size):
                file_name = str(item+1)
                coord = mol.coords[item][:][:]
                print(coord)
                output.gro(mol.n_atom, vectors, time, coord, mol.atom_names,
                    output_dir, file_name)

    elif input_flag == 6:
        startTime = datetime.now()
        option_flag = int(input("""
            [1] - Train a network.
            [2] - Train and test a network.
            [3] - Load and train a network.
            [4] - Load, train and test a network.
            [5] - Load and test a network.
            > """))

        # ensures that tensorflow does not use more cores than requested
        NUMCORES = int(os.getenv("NSLOTS", 1))
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                inter_op_parallelism_threads=NUMCORES,
                allow_soft_placement=True, device_count={'CPU': NUMCORES}))
        tf.compat.v1.keras.backend.set_session(sess)

        # make new directory to store output
        output_dir1 = "plots_and_data"
        isExist = os.path.exists(output_dir1)
        if not isExist:
            os.makedirs(output_dir1)

        # locate dataset
        input_dir2 = "qm_data"
        isExist = os.path.exists(input_dir2)
        if not isExist:
            os.makedirs(input_dir2)

        # initiate molecule and network classes
        mol = read_inputs.Molecule()
        network = Network(mol)
        ann_params = read_inputs.ann("ann_params.txt")
        n_data = ann_params["n_data"]

        # define training and test sets.
        n_train, n_val, n_test = n_data[0], n_data[1], n_data[2]
        set_size = n_train + n_val + n_test
        read_inputs.dataset(mol, input_dir2, set_size, "qm")
        mol.orig_energies = np.copy(mol.energies)

        # set job flags
        if option_flag == 1 or option_flag == 2 or option_flag == 3 or \
                option_flag == 4:
            ann_train = True
            if n_train == 0 or n_val == 0:
                print("""
                ERROR: Cannot train without a training or validation set.
                """)
                exit()
        else:
            ann_train = False
        if option_flag == 2 or option_flag == 4 or option_flag == 5:
            ann_test = True
        else:
            ann_test = False
        if option_flag == 3 or option_flag == 4 or option_flag == 5:
            ann_load = True
        else:
            ann_load = False

        # load previously trained model
        if ann_load:
            input_dir1 = "trained_model"
            isExist = os.path.exists(input_dir1)
            if not isExist:
                print("Error - previously trained model could not be located.")
                exit()
            print("Loading a trained model...")
            prescale = np.loadtxt(f"./{input_dir1}/prescale.txt",
                                  dtype=np.float64).reshape(-1)

            mol.energies = ((prescale[3] - prescale[2]) * (mol.orig_energies
                - prescale[0]) / (prescale[1] - prescale[0]) + prescale[2])
            atoms = np.loadtxt(f"./{input_dir1}/atoms.txt",
                                  dtype=np.float32).reshape(-1)
            model = network.build(len(atoms), ann_params, prescale)
            model.summary()
            model.load_weights(f"./{input_dir1}/best_ever_model")

        else:
            mol.trainval = [*range(0, n_train + n_val, 1)]
            trainval_forces = np.take(mol.forces, mol.trainval, axis=0)
            trainval_energies = np.take(mol.energies, mol.trainval, axis=0)
            prescale = analyseQM.prescale_e(mol, trainval_energies,
                                            trainval_forces)

        # train model
        if ann_train:

            # open new directory to save newly trained model
            output_dir2 = "trained_model"
            isExist = os.path.exists(output_dir2)
            if not isExist:
                os.makedirs(output_dir2)
            shutil.copy2(f"./ann_params.txt", f"./{output_dir2}")

            # get q-values from molecular energies and Cartesian atomic forces
            analyseQM.get_pairs(mol, set_size, output_dir1)

            # build model if not training from scratch
            if not ann_load:
                prescale = analyseQM.prescale_q(mol, prescale)
                print("Building model...")
                model = network.build(len(mol.atoms), ann_params, prescale)
                model.summary()
                np.savetxt(f"./{output_dir2}/atoms.txt",
                           (np.array(mol.atoms)).reshape(-1, 1))
                np.savetxt(f"./{output_dir2}/prescale.txt",
                           (np.array(prescale)).reshape(-1, 1))

            # separate training and validation sets and train network
            print("Training model...")
            mol.train = [*range(0, n_train, 1)]
            mol.val = [*range(n_train, n_train + n_val, 1)]
            network.train(model, mol, ann_params, output_dir1, output_dir2)

            print("Saving model...")
            model.save_weights(f"./{output_dir2}/best_ever_model")

        # test model
        if ann_test:
            mol.test = [*range(n_train + n_val, set_size, 1)]
            if ann_load:
                analyseQM.get_pairs(mol, set_size, output_dir1)

            print("Testing model...")
            network.test(model, mol, output_dir1)

        print(datetime.now() - startTime)

    elif input_flag == 7:
        print("Query external dataset.")
        output_dir = "plots_and_data"
        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)
        # options here for MD22/SPICE/etc
        try:
            inp_vsn = input("""
                Enter the dataset version:
                    1 : original MD17
                    2 : revised MD17
                > """)
        except ValueError:
            print("Invalid Value")
            exit()
        if int(inp_vsn) == 1:
            source = "md17"
        elif int(inp_vsn) == 2:
            source = "rmd17"
        else:
            print("Invalid Value")
            exit()
        try:
            inp_mol = int(input("""
            Enter the molecule:
                 1 : aspirin
                 2 : azobenzene
                 3 : benzene
                 4 : ethanol
                 5 : malonaldehyde
                 6 : naphthalene
                 7 : paracetamol
                 8 : salicylic
                 9 : toluene
                10 : uracil
            > """))
        except ValueError:
            print("Invalid Value")
            exit()
        if inp_mol > 10 or inp_mol < 1:
            print("Invalid Value")
            exit()
        elif inp_mol == 1:
            molecule = "aspirin"
        elif inp_mol == 2:
            molecule = "azobenzene"
            if inp_vsn == 1:
                print("Invalid value - molecule not in MD17 dataset")
                exit()
        elif inp_mol == 3:
            molecule = "benzene"
        elif inp_mol == 4:
            molecule = "ethanol"
        elif inp_mol == 5:
            molecule = "malonaldehyde"
        elif inp_mol == 6:
            molecule = "naphthalene"
        elif inp_mol == 7:
            molecule = "paracetamol"
            if inp_vsn == 1:
                print("Invalid value - molecule not in MD17 dataset")
                exit()
        elif inp_mol == 8:
            molecule = "salicylic"
        elif inp_mol == 9:
            molecule = "toluene"
        elif inp_mol == 10:
            molecule = "uracil"
        sample_freq = int(input("""
            Sample data every n frames:
            > """))
        query_external.geom(sample_freq, molecule, source, output_dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

