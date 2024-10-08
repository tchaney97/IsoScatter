import numpy as np
import os, psutil
import xraydb
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import ConvexHull
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, shared_memory

from ptable_dict import ptable, inverse_ptable, aff_dict
from utilities import load_xyz, load_pdb, most_common_element, get_element_f0_dict, get_element_f1_f2_dict, find_nearest_index

# Compute histogram of distances for a single atom pair type (e.g., C - S)
# Multiply fi*fj (q) for that pair, vector multiplication by the histogram to calculate I(Q) contribution
def estimate_memory_per_block(num_atoms, num_q_values):
    """
    Estimate the memory required per block.
    This is a rough estimate based on the size of the arrays used in calculations.
    """
    # Estimate memory for positions, distances, and scattering factors
    mem_pos = num_atoms * 3 * 8  # positions: num_atoms x 3 coordinates x 8 bytes (double precision)
    mem_rij = num_atoms * num_atoms * 8  # distance matrix: num_atoms x num_atoms x 8 bytes
    mem_f0_q = num_atoms * num_q_values * 8  # scattering factors: num_atoms x num_q_values x 8 bytes

    total_mem_per_block = mem_pos + mem_rij + mem_f0_q
    return total_mem_per_block

def determine_block_size(pos, qs, max_mem_usage_ratio=0.8, min_block_size=100):
    """
    Determine the optimal block size based on available system memory.
    
    Parameters:
    - pos: array of positions
    - qs: array of q values
    - max_mem_usage_ratio: maximum proportion of total memory to use (e.g., 0.8 for 80%)
    - min_block_size: minimum size for a block to avoid excessive overhead
    
    Returns:
    - block_size: calculated block size
    """
    # Get available system memory
    available_mem = psutil.virtual_memory().available

    # Estimate the memory per block
    num_atoms = len(pos)
    num_q_values = len(qs)
    mem_per_block = estimate_memory_per_block(num_atoms, num_q_values)

    # Calculate the number of atoms that fit within the available memory
    block_size = int(available_mem * max_mem_usage_ratio / mem_per_block)

    # Ensure the block size is not too small
    if block_size < min_block_size:
        block_size = min_block_size

    # Adjust block size to ensure it divides the position data reasonably
    if block_size > num_atoms:
        block_size = num_atoms

    return block_size

def determine_safe_thread_count(task_type='cpu', max_factor=2):
    ''' Evaluate the number of threads available for an io-bound or cpu-bound task.
    '''
    # Get the number of CPU cores
    num_cores = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None

    if task_type == 'cpu':
        # For CPU-bound tasks: use a minimum of 1 and a maximum of num_cores
        thread_count = max(1, num_cores - 1)
    elif task_type == 'io':
        # For I/O-bound tasks: consider using more threads
        thread_count = max(1, num_cores * max_factor)
    else:
        raise ValueError("task_type must be 'cpu' or 'io'")

    return thread_count

def compute_sq_for_q(q_val, rij_matrix, f0_q_elements):
    f0_q = f0_q_elements[:, q_val[1]]  # Atomic scattering factors for this q value
    # Pre-multiply rij by q_val to avoid repetitive computation
    # np.sinc includes division by pi
    return np.sum(np.outer(f0_q, f0_q) * np.sinc(rij_matrix * q_val[0] / np.pi))

def sq_with_f0_thread(pos, elements, f0_scales, qs):
    nbins = len(qs)
    sq = np.zeros(nbins)
    rij_matrix = squareform(pdist(pos, metric='euclidean'))
    unique_elements = np.unique(elements)
    f0_dict = {element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in qs]) for element in unique_elements}
    f0_q_elements = np.array([f0_dict[element] for element in elements])
    if isinstance(f0_scales, np.ndarray):
        f0_q_elements *= f0_scales[:, np.newaxis]

    max_workers = determine_safe_thread_count(task_type='cpu')
    print('Number of Max Workers: ', str(max_workers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers to your environment
        futures = {executor.submit(compute_sq_for_q, (q_val, i), rij_matrix, f0_q_elements): i for i, q_val in enumerate(qs)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            sq[futures[future]] = future.result()

    return sq

def mp_sq_for_q(args):
    i, q_val, pos_block, pos, f0_q_elements_block, f0_q_elements, block_size = args
    
    # # code here to calculate pdist between pos_bloc
    rij_matrix = cdist(pos_block, pos, metric='euclidean')
    
    f0_q_block = f0_q_elements_block[:, i]  # Atomic scattering factors for this q value
    f0_q = f0_q_elements[:, i]  # Atomic scattering factors for this q value

    # Pre-multiply rij by q_val to avoid repetitive computation
    # np.sinc includes division by pi
    sq = np.sum(np.outer(f0_q_block, f0_q) * np.sinc(rij_matrix * q_val / np.pi))

    return i, sq

def sq_with_f0_block(pos, elements, f0_scales, qs, block_size=None):
    nbins = len(qs)
    sq_sum = np.zeros(nbins)

    # Prepare the atomic scattering factors
    unique_elements = np.unique(elements)
    f0_dict = {element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in qs]) for element in unique_elements}
    f0_q_elements = np.array([f0_dict[element] for element in elements])
    
    if isinstance(f0_scales, np.ndarray):
        f0_q_elements *= f0_scales[:, np.newaxis]

    # if len(pos)%block_size == 0:
    #     num_blocks = int(len(pos)/block_size)
    # else:
    #     num_blocks = int(len(pos)//block_size + 1)

    # Dynamically determine block size if not provided
    if block_size is None:
        block_size = determine_block_size(pos, qs)
    print(f"Determined block size: {block_size}")

    # Determine the number of blocks
    num_blocks = (len(pos) + block_size - 1) // block_size

    pos_blocks = np.array_split(pos, num_blocks)
    f0_q_elements_blocks = np.array_split(f0_q_elements, num_blocks)

    # - Dynamically determine the number of max workers        
    max_workers = determine_safe_thread_count(task_type='cpu')
    print('Number of Max Workers: ', str(max_workers))

    # Multi-threading for block processing
    for block_num, pos_block in enumerate(tqdm(pos_blocks)):
        f0_q_elements_block = f0_q_elements_blocks[block_num]
        args = [(i, q_val, pos_block, pos, f0_q_elements_block, f0_q_elements, block_size) for i, q_val in enumerate(qs)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(mp_sq_for_q, arg): i for i, arg in enumerate(args)}
            sq_qval = [future.result() for future in as_completed(futures)]
            # print(sq_qval)
        
        # Reassemble sq based on original order
        sq_qval_sorted = np.zeros(nbins)
        for idx, sq in sq_qval:
            sq_qval_sorted[idx] = sq

        # Add sq values to the sum
        sq_sum += sq_qval_sorted

    return sq_sum

def xyz_simulation(xyz_path, qs, vol_pct, solvent_edens, block_size, hull_method=True):
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
        contrast_factor = (solvent_edens * molecular_volume) / len(coords)
        adjusted_electrons = np.array([ptable[symbol] - contrast_factor for symbol in symbols])

        # Filter scatterers based on electron density
        mask = adjusted_electrons >= 1
        debye_scatterers = np.array(coords)[mask]
        debye_species = np.array(symbols)[mask]
        f0_scales = adjusted_electrons[mask] / [ptable[s] for s in debye_species]
        f0_scales = np.asarray(f0_scales)

    #no rescaling of f0 necessary
    else:
        f0_scales = False
        debye_scatterers = np.array(coords)
        debye_species = np.array(symbols)

    # Compute scattering profile
    if block_size is None or block_size > 0:
        sq_vals = sq_with_f0_block(debye_scatterers, debye_species, f0_scales, qs, block_size)
    else:
        sq_vals = sq_with_f0_thread(debye_scatterers, debye_species, f0_scales, qs)
    
    print(sq_vals[5])
    
    #add correction factors in
    if hull_method:
        r0 = 2.82e-13 #thomson scattering length of electron (cm)
        correction_fact = (vol_pct*(r0**2))/(molecular_volume*1e-24)

        iq_vals = sq_vals*correction_fact #absolute intensity (cm^-1)
    else:
        #Think on what right correction factor is for calculation without hull method?
        iq_vals = sq_vals
    
    return iq_vals

def compute_sq_for_q_condensed(q_val, rij_condensed, f0_q_elements):
    f0_q = f0_q_elements[:, q_val[1]]  # Atomic scattering factors for this q value
    
    # Calculate the sinc for condensed rij values
    sinc_rij_q_condensed = np.sinc(rij_condensed * q_val[0] / np.pi)
    
    # Prepare outer product of f0_q
    f0_outer = np.outer(f0_q, f0_q)
    
    # Convert f0_outer to condensed form as well
    f0_outer_condensed = squareform(f0_outer, force='tovector', checks=False)
    
    # Compute the sum directly on the condensed form
    return np.sum(f0_outer_condensed * sinc_rij_q_condensed)

def sq_with_f0_thread_condensed(pos, elements, f0_scales, qs, num_cpus):
    nbins = len(qs)
    sq = np.zeros(nbins)
    
    # Compute condensed distance matrix
    rij_condensed = pdist(pos, metric='euclidean')
    
    unique_elements = np.unique(elements)
    f0_dict = {element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in qs]) for element in unique_elements}
    f0_q_elements = np.array([f0_dict[element] for element in elements])
    
    if isinstance(f0_scales, np.ndarray):
        f0_q_elements *= f0_scales[:, np.newaxis]

    # Submit tasks using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = {
            executor.submit(compute_sq_for_q_condensed, (q_val, i), rij_condensed, f0_q_elements): i
            for i, q_val in enumerate(qs)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            sq[futures[future]] = future.result()

    return sq

def xyz_simulation_condensed(xyz_path, qs, vol_pct, solvent_edens, num_cpus, hull_method=True):
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
    -iq_vals: 1D numpy array of absolute intensity values for each q in exp_file
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
        else:
            molecular_volume = 0
            if len(coords) == 1:
                print('Insufficient atoms to create hull, approximating atom as sphere with radius 1.5Å')
                molecular_volume = (4/3)*np.pi*1.5**3
            else:
                print('Insufficient atoms to create hull, approximating molecule as cylinder with radius 3Å')
                max_distance = np.max(pdist(coords, metric='euclidean'))
                molecular_volume = max_distance * np.pi * 3**2
        
        # Calculate electron density and adjust for solvent contrast
        contrast_factor = (solvent_edens * molecular_volume) / len(coords)
        adjusted_electrons = np.array([ptable[symbol] - contrast_factor for symbol in symbols])

        # Filter scatterers based on electron density
        mask = adjusted_electrons >= 1
        debye_scatterers = np.array(coords)[mask]
        debye_species = np.array(symbols)[mask]
        f0_scales = adjusted_electrons[mask] / np.array([ptable[s] for s in debye_species])
    else:
        f0_scales = False
        debye_scatterers = np.array(coords)
        debye_species = np.array(symbols)

    # Compute scattering profile using condensed matrix
    sq_vals = sq_with_f0_thread_condensed(debye_scatterers, debye_species, f0_scales, qs, num_cpus)

    # Add correction factors
    if hull_method:
        r0 = 2.82e-13  # Thomson scattering length of electron (cm)
        correction_fact = (vol_pct * (r0**2)) / (molecular_volume * 1e-24)
        iq_vals = sq_vals * correction_fact  # Absolute intensity (cm^-1)
    else:
        iq_vals = sq_vals
    
    return iq_vals

    # # Multiprocessing
    # for block_num, pos_block in enumerate(tqdm(pos_blocks)):
    #     f0_q_elements_block = f0_q_elements_blocks[block_num]
    #     args = [(i, q_val, pos_block, pos, f0_q_elements_block, f0_q_elements) for i, q_val in enumerate(qs)]
    #     with Pool(processes=num_cpus) as pool:
    #         qval_sq = pool.map(mp_sq_for_q, args)
    #     sq_qval_sorted = np.asarray(sorted(qval_sq, key=lambda x: x[0]))
    #     sq = sq_qval_sorted[:,1]
    #     q_vals = sq_qval_sorted[:,0]
    #     if np.any(q_vals!=qs):
    #         print(np.shape(q_vals))
    #         print(np.shape(qs))
    #         raise Exception('q value arrays not matching')
    
    #     # add sq values to the sum
    #     sq_sum += sq

    # return sq_sum

    # Threading
    # with ThreadPoolExecutor(max_workers=num_cpus) as executor:  # Adjust max_workers to your environment
    #     futures = {executor.submit(thread_sq_for_q, (q_val, i), pos_block, f0_q_elements): i for i, q_val in enumerate(qs)}
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         sq[futures[future]] = future.result()

# def thread_sq_for_q(q_val, pos_block, f0_q_elements):
#     rij_matrix = squareform(pdist(pos_block, metric='euclidean'))
#     f0_q = f0_q_elements[:, q_val[1]]  # Atomic scattering factors for this q value
#     # Pre-multiply rij by q_val to avoid repetitive computation
#     # np.sinc includes division by pi
#     return np.sum(np.outer(f0_q, f0_q) * np.sinc(rij_matrix * q_val[0] / np.pi))

# def plot_convex_hull(coordinates, hull):
#     """
#     Plots the convex hull and the atomic coordinates of a molecule.

#     Parameters:
#     - coordinates: np.array, the atomic coordinates of the molecule.
#     - hull: ConvexHull object, the convex hull of the molecule.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plotting the atomic coordinates
#     ax.scatter(coordinates[:,0], coordinates[:,1], coordinates[:,2], color='r', s=100)

#     # Plotting the convex hull
#     for simplex in hull.simplices:
#         simplex = np.append(simplex, simplex[0])  # loop back to the first vertex
#         ax.plot(coordinates[simplex, 0], coordinates[simplex, 1], coordinates[simplex, 2], 'k-')

#     # Setting the title
#     ax.set_title('Convex Hull Visualization')

#     plt.show()

# def sq_with_numexpr(pos, elements, f0_scales, qs):
#     nbins = len(qs)
#     sq = np.zeros(nbins)
#     rij_matrix = squareform(pdist(pos, metric='euclidean'))
    
#     unique_elements = np.unique(elements)
#     f0_dict = {element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in qs]) for element in unique_elements}
#     f0_q_elements = np.array([f0_dict[element] for element in elements])
#     if f0_scales:
#         f0_q_elements *= f0_scales[:, np.newaxis]
    
#     for i, q_val in enumerate(tqdm(qs)):
#         f0_q = f0_q_elements[:, i]
#         rij_q = rij_matrix * q_val
#         # Calculate sinc function using Numexpr
#         sinc_rij_q = ne.evaluate("sin(rij_q) / rij_q")
        
#         # Compute outer product using NumPy since Numexpr does not directly support newaxis syntax
#         f0_outer = np.outer(f0_q, f0_q)
#         # Now use Numexpr to compute the final sum
#         sq[i] = ne.evaluate("sum(f0_outer * sinc_rij_q)")
    
#     return sq

# def sq_with_f0(pos, elements, f0_scales, qs):
#     '''
#     Calculates the scattering profile using the debye equation with atomic scattering factors.

#     Input
#       pos = scatterer positions in 3D cartesian coordinates (nx3 array)
#       elements = 1D array of string of the element symbols for each scatterer
#       f0_scales = 1D array of scaling factors for f0 based on solvent electron density contrast
#       qs = list of q values to evaluate scattering intensity at
#     '''
#     nbins = len(qs)
#     sq = np.zeros(nbins)
#     rij_matrix = squareform(pdist(pos, metric='euclidean'))
    
#     # Identify unique elements and precompute f0 for each element and each q value
#     unique_elements = np.unique(elements)
#     f0_dict = {element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in qs]) for element in unique_elements}

#     # Map precomputed f0 values to the elements array
#     f0_q_elements = np.array([f0_dict[element] for element in elements])
#     if f0_scales:
#         f0_q_elements *= f0_scales[:, np.newaxis]

#     for i, q_val in enumerate(tqdm(qs)):
#         f0_q = f0_q_elements[:, i]  # Atomic scattering factors for this q value
        
#         rij_q = rij_matrix * q_val  # Pre-multiply rij by q_val to avoid repetitive computation
        
#         # Compute sin(rij * q_val) / (rij * q_val) for all rij elements (this includes rij and rji)
#         # division by pi is to account for np.sinc definition of sin(x*pi)/(x*pi)
#         sinc_rij_q = np.sinc(rij_q / np.pi)
        
#         # Compute contributions to sq for all pairs of points including self-interaction
#         sq[i] += np.sum(np.outer(f0_q, f0_q) * sinc_rij_q)
        
#     return sq