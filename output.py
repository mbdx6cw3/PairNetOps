import matplotlib.pyplot as plt
import numpy as np

def lineplot(x, y, type, x_label, y_label, title, output_dir):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xscale(type)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"./{output_dir}/{title}.png")
    return None

def twolineplot(x1, x2, y1, y2, l1, l2, type, x_label, y_label, title,
                output_dir):
    fig, ax = plt.subplots()
    ax.plot(x1, y1, label = l1)
    ax.plot(x2, y2, label = l2)
    plt.xscale(type)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"./{output_dir}/{title}.png")
    return None

def scatterplot(x, y, type, x_label, y_label, title, output_dir):
    fig, ax = plt.subplots()
    ax.scatter(x, y, marker="o", c="black", s=1)
    plt.xscale(type)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"./{output_dir}/{title}.png")
    return None

def gro(n_atoms, vectors, time, coords, atom_names, output_dir, file_name):
    '''
    For a given set of structure 3D coords and the atom
    associated numbers, output xyz format file called filename.
    Either write a new file (open_type='w') or append to
    an existing file (open_type='a') and chose the entry number
    with i=number
    '''
    if time == 0.0:
        open_type = "w"
    else:
        open_type = "a"
    gro_file = open(f"{output_dir}/{file_name}.gro", open_type)
    gro_file.write(f"output t={time} ps\n")
    gro_file.write(f"{n_atoms}\n")
    for atom in range(n_atoms):
        x = coords[atom][0]
        y = coords[atom][1]
        z = coords[atom][2]
        gro_file.write("{:>8} {:>6} {:4} {:7.3f} {:7.3f} {:7.3f}\n".
                       format("1MOL", atom_names[atom], atom+1, x, y, z))
    gro_file.write("{:10.5f} {:10.5f} {:10.5f}\n".
                   format(vectors[0],vectors[1],vectors[2]))
    gro_file.close()


def write_pdb(coords, resname, resid, atoms, filename, open_type):
    '''
    Write 3D coords in .pdb format. Provide the resname, resid, atoms
    and filename.
    '''
    outfile = open(filename, open_type)
    for i in range(len(atoms)):
        outfile.write('\n'.join(['{:6s}{:5d} {:^4s}{:1s}{:3s}' \
                ' {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}' \
                '{:6.0f}{:6.0f}          {:>2s}{:2s}'.format(
                    'ATOM', i+1, mol.atom_names[i], ' ', resname[0:3],
                    'A', int(str(resid)[0:4]), ' ',
                    coords[i][0], coords[i][1], coords[i][2],
                    1, 1, ' ', ' ')]) + '\n')
    outfile.write('TER\n')
    outfile.close()
    return None

def scurve(baseline, values, output_dir, output_file):
    """
    This function calculates S-curves for MM determined forces.
    Must have run qm2ml.py first to collect QM force dataset.
    :param molecule:
    :param set_size:
    :param n_atom:
    :return:
    """
    RSE = np.sqrt((baseline-values)**2)
    hist, bin_edges = np.histogram(RSE,1000,(-0.2,np.amax(RSE)))
    hist = np.cumsum(hist)
    bin_edges = bin_edges[range(1,bin_edges.shape[0])]
    hist = hist/values.shape[0]*100
    bin_edges[0] = 0.0
    lineplot(bin_edges, hist, "log", "Error ($kcal/mol/\AA$)",
        "% of forces below error", f"{output_file}", output_dir)
    np.savetxt(f"./{output_dir}/{output_file}.dat", np.column_stack((bin_edges,
        hist)), fmt='%.6f', delimiter = " ")
    return None


def heatmap2D(x, y, z, z_max, z_min, output_dir, file, title, cmap):
    fig, ax = plt.subplots()
    # TODO: fix this - why do y and x have to this way round
    c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=z_min, vmax=z_max)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    x_label = "$\u03C6  (degrees)$"
    y_label = "$\u03C8  (degrees)$"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    plt.title(title)
    fig.colorbar(c, ax=ax)
    plt.savefig(f"./{output_dir}/{file}.png")
    return None


def write_gau(mol, set_size, output_dir):

    # read input text section
    gaussian_params = open(f"./gaussian.txt", "r")
    text = gaussian_params.read().strip('\n')

    # create QM input files
    for item in range(set_size):
        qm_file = open(f"./{output_dir}/mol_{item+1}.gjf", "w")
        new_text = text.replace("index", f"{item+1}")
        print(new_text, file=qm_file)
        for atom in range(mol.n_atom):
            print(f"{mol.atom_names[atom]} "
                  f"{mol.coords[item,atom,0]:.8f} " 
                  f"{mol.coords[item,atom,1]:.8f} "
                  f"{mol.coords[item,atom,2]:.8f}",
                  file=qm_file) # convert to Angstroms
        print(file=qm_file)
        qm_file.close()
    return None
