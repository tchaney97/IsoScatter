import numpy as np
import os
import xraydb
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import dask.array as da
from scipy.spatial.distance import pdist, squareform
import xraydb
from concurrent.futures import ThreadPoolExecutor, as_completed
import numexpr as ne

from ptable_dict import ptable, inverse_ptable, aff_dict
from utilities import load_xyz, load_pdb, most_common_element, get_element_f0_dict, get_element_f1_f2_dict, find_nearest_index


def compute_sq_for_q(q_val, rij_matrix, f0_q_elements):
    f0_q = f0_q_elements[:, q_val[1]]  # Atomic scattering factors for this q value
    rij_q = rij_matrix * q_val[0]  # Pre-multiply rij by q_val to avoid repetitive computation
    sinc_rij_q = np.sinc(rij_q / np.pi)  # np.sinc includes division by pi
    return np.sum(np.outer(f0_q, f0_q) * sinc_rij_q)

def sq_with_f0_thread(pos, elements, f0_scales, qs):
    nbins = len(qs)
    sq = np.zeros(nbins)
    rij_matrix = squareform(pdist(pos, metric='euclidean'))
    unique_elements = np.unique(elements)
    f0_dict = {element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in qs]) for element in unique_elements}
    f0_q_elements = np.array([f0_dict[element] for element in elements])
    if f0_scales:
        f0_q_elements *= f0_scales[:, np.newaxis]

    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers to your environment
        futures = {executor.submit(compute_sq_for_q, (q_val, i), rij_matrix, f0_q_elements): i for i, q_val in enumerate(qs)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            sq[futures[future]] = future.result()

    return sq

def sq_with_f0(pos, elements, f0_scales, qs):
    '''
    Calculates the scattering profile using the debye equation with atomic scattering factors.

    Input
      pos = scatterer positions in 3D cartesian coordinates (nx3 array)
      elements = 1D array of string of the element symbols for each scatterer
      f0_scales = 1D array of scaling factors for f0 based on solvent electron density contrast
      qs = list of q values to evaluate scattering intensity at
    '''
    nbins = len(qs)
    sq = np.zeros(nbins)
    rij_matrix = squareform(pdist(pos, metric='euclidean'))
    
    # Identify unique elements and precompute f0 for each element and each q value
    unique_elements = np.unique(elements)
    f0_dict = {element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in qs]) for element in unique_elements}

    # Map precomputed f0 values to the elements array
    f0_q_elements = np.array([f0_dict[element] for element in elements])
    if f0_scales:
        f0_q_elements *= f0_scales[:, np.newaxis]

    for i, q_val in enumerate(tqdm(qs)):
        f0_q = f0_q_elements[:, i]  # Atomic scattering factors for this q value
        
        rij_q = rij_matrix * q_val  # Pre-multiply rij by q_val to avoid repetitive computation
        
        # Compute sin(rij * q_val) / (rij * q_val) for all rij elements (this includes rij and rji)
        # division by pi is to account for np.sinc definition of sin(x*pi)/(x*pi)
        sinc_rij_q = np.sinc(rij_q / np.pi)
        
        # Compute contributions to sq for all pairs of points including self-interaction
        sq[i] += np.sum(np.outer(f0_q, f0_q) * sinc_rij_q)
        
    return sq

def sq_with_numexpr(pos, elements, f0_scales, qs):
    nbins = len(qs)
    sq = np.zeros(nbins)
    rij_matrix = squareform(pdist(pos, metric='euclidean'))
    
    unique_elements = np.unique(elements)
    f0_dict = {element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in qs]) for element in unique_elements}
    f0_q_elements = np.array([f0_dict[element] for element in elements])
    if f0_scales:
        f0_q_elements *= f0_scales[:, np.newaxis]
    
    for i, q_val in enumerate(tqdm(qs)):
        f0_q = f0_q_elements[:, i]
        rij_q = rij_matrix * q_val
        # Calculate sinc function using Numexpr
        sinc_rij_q = ne.evaluate("sin(rij_q) / rij_q")
        
        # Compute outer product using NumPy since Numexpr does not directly support newaxis syntax
        f0_outer = np.outer(f0_q, f0_q)
        # Now use Numexpr to compute the final sum
        sq[i] = ne.evaluate("sum(f0_outer * sinc_rij_q)")
    
    return sq

def plot_convex_hull(coordinates, hull):
    """
    Plots the convex hull and the atomic coordinates of a molecule.

    Parameters:
    - coordinates: np.array, the atomic coordinates of the molecule.
    - hull: ConvexHull object, the convex hull of the molecule.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the atomic coordinates
    ax.scatter(coordinates[:,0], coordinates[:,1], coordinates[:,2], color='r', s=100)

    # Plotting the convex hull
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # loop back to the first vertex
        ax.plot(coordinates[simplex, 0], coordinates[simplex, 1], coordinates[simplex, 2], 'k-')

    # Setting the title
    ax.set_title('Convex Hull Visualization')

    plt.show()

def xyz_simulation(xyz_path, qs, vol_pct, solvent_edens, hull_method=True):
    """
    Calculates the scattering intensity I(q) for a given molecular structure in an xyz file,
    concentration, and background solvent electron density. Molecular volume is calculated via
    convex hull and background solvent essentially subtracts "z" from each atom in molecule. 
    Note, atoms with Z/(volume/#atoms) less than solvent electron density are removed. 
    Complexity is O(n) for number of q-values and O(n^2) for the number of atoms in .xyz file

    Parameters:
    - xyz_path: string, path to xyz file of molecule, NP, etc
    - qs: 1D array of q values which you would like intensity to be calculated
    - vol_pct: float, volume percent of molecule in solution
    - solvent_edens: float, electron density of solvent in e/Å^3
    - plot_hull: boolean 

    Returns:
    -iq_vals: 1D numpy array of absolute intesntiy values for each q in exp_file
    """
    # Extracting the atomic symbols and positions from the xyz file
    with open(xyz_path, 'r') as file:
        lines = file.readlines()
    # Extracting atom data
    atom_data = [line.split() for line in lines[2:] if len(line.split()) == 4]
    symbols, coords = zip(*[(parts[0], np.array(list(map(float, parts[1:])))) for parts in atom_data])

    coords = np.array(coords)
    # Calculate molecular volume
    if hull_method:
        if len(coords) > 3:
            hull = ConvexHull(coords)
            molecular_volume = hull.volume
            #plot hull for verification
            if plot_hull:
                plot_convex_hull(coords, hull)

        else:
            molecular_volume = 0
            if len(coords) == 1:
                print('Insufficient atoms to create hull, approximating atom as sphere with radius 1.5Å')
                molecular_volume = (4/3)*np.pi*1.5**3
            else:
                print('Insufficient atoms to create hull, approximating molecule as cylinder with radius 3Å')
                max_distance = np.max(pdist(coords, metric='euclidean'))
                molecular_volume = max_distance*np.pi*3**2
        
        # Calculate electron density and adjust for solvent contrast
        tot_electrons = sum(ptable[symbol] for symbol in symbols)
        contrast_factor = (solvent_edens * molecular_volume) / len(coords)
        adjusted_electrons = np.array([ptable[symbol] - contrast_factor for symbol in symbols])

        # Filter scatterers based on electron density
        mask = adjusted_electrons > 1
        debye_scatterers = np.array(coords)[mask]
        debye_species = np.array(symbols)[mask]
        f0_scales = adjusted_electrons[mask] / [ptable[s] for s in debye_species]
    if not hull_method:
        f0_scales = False

    # Compute scattering profile
    sq_vals = sq_with_f0_thread(debye_scatterers, debye_species, f0_scales, qs)
    r0 = 2.82e-13 #thomson scattering length of electron (cm)
    correction_fact = (vol_pct*(r0**2))/(molecular_volume*1e-24)

    iq_vals = sq_vals*correction_fact #absolute intensity (cm^-1)
    return iq_vals



