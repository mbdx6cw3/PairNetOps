import numpy as np
import calc_geom
import output
from scipy.stats import binned_statistic

def dist(mol, set_size, output_dir):
    n_atom = len(mol.atoms)
    hist, bin = np.histogram(mol.forces.flatten(),200,(-250,250))
    #np.max(mol.forces.flatten())))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    output.lineplot(bin, hist / bin_width / set_size / 3.0 / n_atom, "linear",
                "force (kcal/mol/A)", "probability", "qm_force_dist", output_dir)
    np.savetxt(f"./{output_dir}/qm_force_dist.dat",
        np.column_stack((bin, hist / bin_width / set_size / 3.0 / n_atom)),
        delimiter = " ",fmt="%.6f")
    hist, bin = np.histogram(mol.energies, 50, (np.min(mol.energies),np.max(mol.energies)))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    output.lineplot(bin, hist / bin_width / set_size, "linear", "energy",
        "probability", "qm_energy_dist", output_dir)
    np.savetxt(f"./{output_dir}/qm_energy_dist.dat",
        np.column_stack((bin, hist / bin_width / set_size)), delimiter = " ",
        fmt="%.6f")
    return None


def energy_CV(mol, atom_indices, set_size, output_dir):
    CV_list = np.array(atom_indices.split(), dtype=int)
    CV = np.empty(shape=[set_size])
    for item in range(set_size):
        p = np.zeros([len(CV_list), 3])
        p[0:] = mol.coords[item][CV_list[:]]
        if len(CV_list) == 2:
            x_label = "$r_{ij} / \AA$"
            CV[item] = calc_geom.distance(p)
            #print(item, CV[item], mol.energies[item])
        elif len(CV_list) == 3:
            x_label = "$\u03F4_{ijk}  (degrees)$"
            CV[item] = calc_geom.angle(p)
        elif len(CV_list) == 4:
            x_label = "$\u03C6_{ijkl} (degrees)$"
            CV[item] = calc_geom.dihedral(p)
            #print(item, CV[item], mol.energies[item])
    # plot distribution, scatter and save data
    print("MEAN:", np.mean(CV))
    energy = mol.energies[:,0] - np.min(mol.energies[:,0])
    hist, bin = np.histogram(CV, 50, (min(CV), max(CV)))
    bin = bin[range(1, bin.shape[0])]
    output.lineplot(bin, hist / set_size, "linear", x_label,
        "relative probability", "geom_dist", output_dir)
    np.savetxt(f"./{output_dir}/geom.dat",
        np.column_stack((bin, hist / set_size)), delimiter=" ", fmt="%.6f")
    output.scatterplot(CV, energy, "linear", x_label,
        "QM energy (kcal/mol)", "qm_energy_CV_scatter", output_dir)
    means, edges, counts = binned_statistic(CV, energy, statistic='min',
        bins=72, range=(-180.0, 180.0))
    bin_width = edges[1] - edges[0]
    bin_centers = edges[1:] - bin_width / 2
    output.lineplot(bin_centers, means, "linear", x_label,
        "mean energy (kcal/mol)", "qm_energy_geom", output_dir)
    np.savetxt(f"./{output_dir}/qm_energy_geom.dat",
        np.column_stack((bin_centers, means)), delimiter = " ",
               fmt="%.6f")
    return None

def rmsd_dist(mol, set_size):
    n_atoms = len(mol.atoms)
    _NC2 = int(n_atoms * (n_atoms - 1) / 2)
    r_ij_0 = np.zeros((n_atoms, n_atoms))
    rmsd_dist = np.zeros(set_size)
    # loop over all structures
    for s in range(set_size):
        sum_rmsd_dist = 0
        # loop over all atom pairs
        for i in range(n_atoms):
            for j in range(i):
                r_ij = np.linalg.norm(mol.coords[s][i] - mol.coords[s][j])
                #print(i,j,r_ij)
                if s == 0:
                    r_ij_0[i,j] = r_ij
                else:
                    rij_diff = r_ij - r_ij_0[i,j]
                    sum_rmsd_dist += rij_diff**2
        if s != 0:
            rmsd_dist[s] = np.sqrt(sum_rmsd_dist / n_atoms / n_atoms)
    return rmsd_dist

def prescale_e(mol, energies, forces):
    min_e, max_e = np.min(energies), np.max(energies)
    min_f, max_f = np.min(forces), np.max(forces)
    min_f = np.min(np.abs(forces))
    prescale = [min_e, max_e, min_f, max_f, 0, 0]
    mol.energies = ((max_f-min_f)*(mol.orig_energies-min_e)/(max_e-min_e)+min_f)
    return prescale

def prescale_q(mol, prescale):
    n_atoms = len(mol.atoms)
    n_pairs = int(n_atoms * (n_atoms - 1) / 2)
    input_NRF = mol.mat_NRF.reshape(-1, n_pairs)
    trainval_input_NRF = np.take(input_NRF, mol.trainval, axis=0)
    trainval_output_matFE = np.take(mol.output_matFE, mol.trainval, axis=0)
    prescale[4] = np.max(np.abs(trainval_input_NRF))
    prescale[5] = np.max(np.abs(trainval_output_matFE))
    return prescale

def get_pairs(mol, set_size, output_dir):
    '''Get decomposed energies and forces from the same simultaneous equation'''

    n_atoms = len(mol.atoms)
    _NC2 = int(n_atoms * (n_atoms - 1) / 2)

    # assign arrays
    mol.mat_NRF = np.zeros((set_size, _NC2))
    mol.mat_r = np.zeros((set_size, _NC2))
    mol.mat_bias = np.zeros((set_size, _NC2))
    mol.mat_FE = np.zeros((set_size, _NC2))
    mol.mat_eij = np.zeros((set_size, n_atoms * 3 + 1, _NC2))
    mol.mat_i = np.zeros(_NC2)
    mol.mat_j = np.zeros(_NC2)

    # loop over all structures
    for s in range(set_size):

        mat_Fvals = np.zeros((n_atoms, 3, _NC2))
        _N = -1

        # loop over all atom pairs
        for i in range(n_atoms):
            zi = mol.atoms[i]
            for j in range(i):
                _N += 1
                zj = mol.atoms[j]

                if s == 0:
                    mol.mat_i[_N] = i
                    mol.mat_j[_N] = j

                # calculate interatomic distances, save to distance matrix
                r = np.linalg.norm(mol.coords[s][i] - mol.coords[s][j])
                mol.mat_r[s, _N] = r

                # internuclear repulsion force matrix
                mol.mat_NRF[s, _N] = get_NRF(zi, zj, r)
                bias = 1 / r
                mol.mat_bias[s, _N] = bias
                mol.mat_eij[s, n_atoms * 3, _N] = bias

                # loop over Cartesian axes - don't need this
                for x in range(0, 3):
                    val = ((mol.coords[s][i][x] - mol.coords[s][j][x]) /
                           mol.mat_r[s, _N])

                    mat_Fvals[i, x, _N] = val
                    mat_Fvals[j, x, _N] = -val
                    mol.mat_eij[s, i * 3 + x, _N] = val
                    mol.mat_eij[s, j * 3 + x, _N] = -val

        mat_Fvals2 = mat_Fvals.reshape(n_atoms * 3, _NC2)
        forces2 = mol.forces[s].reshape(n_atoms * 3)

        # calculation normalisation factor
        norm_recip_r = np.sum(mol.mat_bias[s] ** 2) ** 0.5

        # normalisation of pair energy biases to give dimensionless quantities
        mol.mat_bias[s] = mol.mat_bias[s] / norm_recip_r
        # mol.mat_eij not used anywhere because forces are obtained from E
        mol.mat_eij[s, -1] = mol.mat_bias[s]

        mat_bias2 = mol.mat_bias[s].reshape((1, _NC2))

        _E = mol.energies[s].reshape(1)

        mat_FE = np.concatenate((mat_Fvals2, mat_bias2), axis=0)
        _FE = np.concatenate([forces2.flatten(), _E.flatten()])

        # why not just matrix multiply mat_bias2 and _FE here because we don't
        # need the forces anyway?
        # do we even need to scale the energy with the forces?
        decomp_FE = np.matmul(np.linalg.pinv(mat_FE), _FE)
        mol.mat_FE[s] = decomp_FE
        mol.output_matFE = mol.mat_FE.reshape(-1, _NC2)

    # flatten output_matFE instead below?
    output.scatterplot([mol.mat_r.flatten()], [mol.mat_FE.flatten()], "linear",
        "$r_{ij}$ / $\AA$", "q / kcal/mol/$\AA$", "q_rij", output_dir)
    hist, bin = np.histogram(mol.mat_FE.flatten(), 200,
        (np.min(mol.mat_FE.flatten()), np.max(mol.mat_FE.flatten())))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    output.lineplot(bin, hist / bin_width / _NC2 / set_size, "linear",
        "q / kcal/mol/$\AA$", "probability", "q_dist", output_dir)
    np.savetxt(f"./{output_dir}/q_dist.dat",
               np.column_stack((bin, hist / bin_width / _NC2 / set_size)),
               delimiter=" ", fmt="%.6f")
    return None


##### CHECK THIS URGENTLY ######
def get_NRF(zA, zB, r):
    _NRF = r and (zA * zB * np.float64(627.5095 * 0.529177) / (r ** 2))
    return _NRF


def get_forces(mol, all_coords, all_prediction):
    '''Take per-atom pairwise decomposed F and convert them back into
    Cart forces.'''

    n_atoms = len(mol.atoms)
    _NC2 = int(n_atoms * (n_atoms - 1) / 2)
    all_recomb_F = np.zeros((len(all_coords), n_atoms, 3))

    # loop over all structures
    s = -1
    for coords, prediction in zip(all_coords, all_prediction):
        s += 1
        # set projection matrix for this structure
        mat_eij = np.zeros((n_atoms, 3, _NC2))
        _N = -1
        # loop over all pairs
        for i in range(n_atoms):
            for j in range(i):
                _N += 1
                r = np.linalg.norm(coords[i]-coords[j])
                # loop over Cartesian axes
                for x in range(0, 3):
                    eij = ((coords[i][x] - coords[j][x]) / r)
                    mat_eij[i,x,_N] = eij
                    mat_eij[j,x,_N] = -eij

        for i in range(n_atoms):
            recomb_F = np.dot(mat_eij[i], prediction)
            all_recomb_F[s,i] = recomb_F

    return all_recomb_F

