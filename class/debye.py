import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
import xraydb
import ast, os
from datetime import datetime
import shutil
import time
import sys
import psutil

class DebyeCalc:
    """
    A class to calculate Debye scattering I(Q) v. q (A^-1) from XYZ coordinate files for elements.
    """

    def __init__(self, 
                 xyzPath,
                 QInput,
                #  adps = None,
                 suppress_plots=True,
                 log_scale=True):
        """
        Initializes the DebyeCalc class with a path to an XYZ file.

        Parameters:
        - xyzPath (str): The path to the XYZ file containing atomic coordinates and elements.
        - QInput (list or str): List of float values to use for the extents of the calculated QRange and the step size,
                                or a path to an .iq file containing experimental data with Q values.
        - adps (list, optional): Anisotropic displacement parameters for each atom. Each element should be a 6-component vector representing the U tensor.                                
        - suppress_plots (bool, optional): Whether to suppress plot display in Jupyter cell. Defaults to True.
        - log_scale (bool, optional): Whether to use logarithmic spacing for QInput. Defaults to False.
        """

        self.r0 = 2.82e-13 # Thomson scattering length of electron (cm)
        self.suppress_plots = suppress_plots
        self.log_scale = log_scale

        # Set the QRange used for the calculation
        self.QInput = QInput
        self.QRange = self.setQRange()  # Initialize QRange using the provided or default values
        # print(f"Generated QRange: {self.QRange}")

        # Load the XYZ File Data
        self.xyzPath = xyzPath
        self.coords, self.elements = self.loadXYZ(xyzPath)

        # Load the anisotropic displacement parameters
        # self.adps = adps

        # Load ptable and atomic_masses
        self.ptable = self.load_tabledata('ptable.txt')
        self.atomic_masses = self.load_tabledata('atomic_masses.txt')

        # Initialize dictionary to store memory usage and calculation time for specific variables
        self.memory_and_time_stats = {}

    def setQRange(self):
        """
        Validates and sets the q-range for the scattering calculation based on user input.

        Parameters:
        - QInput (list or str): List containing [q_min, q_max, step_size or num] or path to an .iq file

        Returns:
        - np.ndarray: Array of Q values
        """
        if isinstance(self.QInput, str):
            try:
                self.QInput = ast.literal_eval(self.QInput)
                print(f"Converted QInput to list: {self.QInput}")
                # Print each element of the list
                for i, elem in enumerate(self.QInput, start=1):
                    print(f"Element {i}: {elem}")
            except (ValueError, SyntaxError):
                if self.QInput.endswith('.iq'):
                    self.QInputFile = self.QInput
                    QRange = self.load_q_from_iq(self.QInput)
                    return QRange
                else:
                    raise ValueError("QInput must be either a list of three numbers or a path to a valid .iq file")

        if isinstance(self.QInput, list):
            if len(self.QInput) != 3:
                raise ValueError("Input list must contain three numbers: [q_min, q_max, step_size or num]")
            q_min, q_max, step_or_num = self.QInput
            print(f"q_min: {q_min}, q_max: {q_max}, step_or_num: {step_or_num}")
            if self.log_scale:
                QRange = np.logspace(np.log10(q_min), np.log10(q_max), num=int(step_or_num))
            else:
                QRange = np.linspace(q_min, q_max, num=int(step_or_num))
            # print(f"Generated QRange: {QRange}")
        else:
            raise ValueError("QInput must be either a list of three numbers or a path to a valid .iq file")
        
        return QRange

    def load_q_from_iq(self, iq_path):
        """
        Load Q values from a given .iq file.

        Parameters:
        - iq_path (str): The path to the .iq file.

        Returns:
        - np.ndarray: Array of Q values extracted from the file.
        """
        try:
            with open(iq_path, 'r') as file:
                lines = file.readlines()
            for i, line in enumerate(lines):
                if line.startswith("Q A-1"):
                    data_start_idx = i + 1
                    break
            q_values = []
            for line in lines[data_start_idx:]:
                q_val = float(line.split()[0])
                q_values.append(q_val)
            return np.array(q_values)
        except Exception as e:
            raise ValueError(f"Error reading Q values from .iq file: {e}")

    def load_tabledata(self, filename):
        """
        Load data from a specified file, assuming the file contains a dictionary definition.

        Parameters:
        - filename (str): The name of the file to load data from.

        Returns:
        - dict: The dictionary of data loaded from the file.
        """
        dir_path = os.getcwd()
        full_path = os.path.join(dir_path, filename)

        try:
            with open(full_path, 'r') as file:
                data = file.read()
            data_dict = data.split('=', 1)[1].strip()
            table = ast.literal_eval(data_dict)
            return table
        except FileNotFoundError:
            print(f"File not found: {full_path}")
            return {}
        except SyntaxError as e:
            print(f"Error reading the data: {e}")
            return {}

    def loadXYZ(self, xyzPath):
        """
        Loads atomic coordinates and elements from an XYZ file.

        Parameters:
        - xyzPath (str): The path to the XYZ file.

        Returns:
        - tuple: A tuple containing numpy arrays of coordinates and elements.
        """
        with open(xyzPath, 'r') as file:
            lines = file.readlines()
        atom_data = [line.split() for line in lines[2:] if len(line.split()) == 4]
        elements, coords = zip(*[(parts[0], np.array(list(map(float, parts[1:])))) for parts in atom_data])

        return np.array(coords), np.array(elements)

    def anisotropic_debye_waller(self, adps, q_vector):
        """
        Calculate the anisotropic Debye-Waller factor for a given q vector.

        Parameters:
        - adps (np.ndarray): An array of anisotropic displacement parameters (Uij) for each atom, in Å^2.
        - q_vector (np.ndarray): The scattering vector.

        Returns:
        - np.ndarray: The Debye-Waller factors for each atom pair.
        """
        dw_factor = np.zeros((len(adps), len(adps)))
        for i in range(len(adps)):
            for j in range(len(adps)):
                if i <= j:
                    U_i = np.array([
                        [adps[i][0], adps[i][3], adps[i][4]],
                        [adps[i][3], adps[i][1], adps[i][5]],
                        [adps[i][4], adps[i][5], adps[i][2]]
                    ])
                    U_j = np.array([
                        [adps[j][0], adps[j][3], adps[j][4]],
                        [adps[j][3], adps[j][1], adps[j][5]],
                        [adps[j][4], adps[j][5], adps[j][2]]
                    ])
                    U_sum = U_i + U_j
                    q_uq = np.dot(q_vector.T, np.dot(U_sum, q_vector))
                    dw_factor[i, j] = np.exp(-q_uq)
                    if i != j:
                        dw_factor[j, i] = dw_factor[i, j]  # Symmetric
        return dw_factor

    def sq_with_f0_contrast(self): #, adps=None):
        """
        Calculates the scattering intensity profile using the Debye equation with contrast-scaled atomic scattering factors.
        Considers f0 scattering factors, solvent contrast, and optionally applies anisotropic displacement parameters (ADPs).

        Parameters:
        - adps (list or np.ndarray, optional): Anisotropic displacement parameters for each atom. Each element should be a 6-component vector representing the U tensor.

        Returns:
        - np.ndarray: Scattering intensities for given q values.
        """
        nbins = len(self.QRange)
        sq_vals = np.zeros(nbins)

        # Measure time and memory usage for rij_matrix
        start_time = time.time()
        rij_matrix = squareform(pdist(self.debye_scatterers, metric='euclidean'))
        end_time = time.time()
        self.memory_and_time_stats['rij_matrix'] = {
            'size_bytes': sys.getsizeof(rij_matrix),
            'time_seconds': end_time - start_time
        }
        
        self.track_memory_usage("After creating rij_matrix")

        # Precompute f0 for each element and each q value
        self.unique_elements = np.unique(self.debye_elements)
        f0_dict = {element: np.array([xraydb.f0(element, q / (4 * np.pi))[0] for q in self.QRange]) for element in self.unique_elements}

        # Map precomputed f0 values to the elements array
        f0_q_elements = np.array([f0_dict[element] for element in self.debye_elements])

        self.track_memory_usage("After creating f0_q_elements")

        # # Calculate the Debye-Waller factors for each q value if ADPs are provided
        # if adps is not None:
        #     debye_waller_factors = []
        #     for q_val in self.QRange:
        #         debye_waller = self.anisotropic_debye_waller(adps, np.array([q_val, q_val, q_val]))
        #         debye_waller_factors.append(debye_waller)
        # else:
        #     debye_waller_factors = [np.ones_like(rij_matrix) for _ in self.QRange]
        # -- This line is killing memory! Need an alternative for when no debye-waller factors are applied.

        # for i, (q_val, debye_waller) in enumerate(zip(self.QRange, debye_waller_factors)):
        
        start_time = time.time()
        for i, (q_val) in enumerate(zip(self.QRange)):
            f0_q = f0_q_elements[:, i]
            rij_q = rij_matrix * q_val

            sinc_rij_q = np.sinc(rij_q / np.pi)

            # # Apply Debye-Waller factor
            # sinc_rij_q *= debye_waller

            # Compute contributions to sq for all pairs of points including self-interaction
            sq_vals[i] += np.sum(np.outer(f0_q, f0_q) * sinc_rij_q)
        end_time = time.time()

        self.memory_and_time_stats['f0_q'] = {
            'size_bytes': sys.getsizeof(f0_q_elements),
            'time_seconds': end_time - start_time
        }

        self.memory_and_time_stats['sinc_rij_q'] = {
            'size_bytes': sys.getsizeof(sinc_rij_q),
            'time_seconds': end_time - start_time
        }

        return sq_vals

    def debyecalc(self): #, adps=None):
        """
        Calculates the scattering intensity I(q) for a given molecular structure in an xyz file,
        using contrast-scaled f0 values. Molecular volume is calculated via convex hull and
        background solvent essentially subtracts "z" from each atom in molecule.
        Atoms with Z/(volume/#atoms) less than solvent electron density are removed.

        Parameters:
        - adps (list or np.ndarray, optional): Anisotropic displacement parameters for each atom. Each element should be a 6-component vector representing the U tensor.

        Returns:
        - tuple: Scattering intensities (sq_vals) calculated for the given q-values.
        """
        # self.adps = adps
        self.debye_scatterers = np.array(self.coords)
        self.debye_elements = np.array(self.elements)
        self.f0_scales = None

        # - Running the S(Q) Contrast Function
        self.track_memory_usage("Before calculation")
        self.start_time = time.time()
        self.sq_vals = self.sq_with_f0_contrast()
        self.end_time = time.time()
        self.track_memory_usage("After calculation")

        self.calc_time = self.end_time - self.start_time

        return

    def plot_scattering(self, save_path):
        """
        Plots the calculated scattering data and saves the plot as a PNG file.

        Parameters:
        - save_path (str): Path to save the plot PNG file.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if hasattr(self, 'sq_vals') and hasattr(self, 'QRange'):
            ax.plot(self.QRange, self.sq_vals, label=self.get_xyz_filename(), marker='.', linestyle='-', markersize=3, color='mediumslateblue')

        ax.set_xlabel('q ($\AA^{-1}$)')
        ax.set_ylabel('S(Q)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title('Scattering Data Comparison')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def get_xyz_filename(self):
        """
        Extracts the filename from the full path of the XYZ file for use in labeling plots.

        Returns:
        - str: The filename of the XYZ file.
        """
        return os.path.splitext(os.path.basename(self.xyzPath))[0]

    def create_output_directory(self):
        """
        Creates a unique output directory based on the XYZ filename and current timestamp.
        If the directory already exists, appends an index suffix to create a unique directory.
        
        Returns:
        - str: Path of the created output directory.
        - str: Formatted timestamp used in the directory name.
        """
        base_name = self.get_xyz_filename()
        timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
        dir_path = f"{base_name}_{timestamp}"
        index = 0

        while os.path.exists(dir_path):
            index += 1
            dir_path = f"{base_name}_{timestamp}_{index:03d}"

        os.makedirs(dir_path)
        return dir_path, timestamp

    def saveData(self, output_dir, timestamp):
        """
        Saves the QRange and calculated scattering values to a file with specified parameters and header information.

        Parameters:
        - output_dir (str): The output directory where the file will be saved.
        - timestamp (str): The timestamp to include in the filename.
        """
        save_name = f"{self.get_xyz_filename()}_{timestamp}.sq"
        full_path = os.path.join(output_dir, save_name)

        date_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = f"Scattering Data Calculated\nDate and Time: {date_time_str}\nXYZ File Path: {self.xyzPath}\n"
        if hasattr(self, 'QInputFile'):
            header += f"Q Input File: {self.QInputFile}\n"
        header += "Q A-1\tS(a.u.)\n"

        q_format = f"{{:.{len(str(self.QRange[0]).split('.')[1])}f}}"
        data_lines = "\n".join([f"{q_format.format(q)}\t{intensity:.6e}" for q, intensity in zip(self.QRange, self.sq_vals)])

        try:
            with open(full_path, 'w') as file:
                file.write(header + data_lines)
            print(f"Data successfully saved to {full_path}")
        except IOError as e:
            print(f"An error occurred while writing the file: {e}")

    def save_metadata_and_statistics(self, output_dir, timestamp):
        """
        Saves metadata and computation statistics to a single file.

        Parameters:
        - output_dir (str): The output directory where the file will be saved.
        - timestamp (str): The timestamp to include in the filename.
        """
        save_name = f"metadata_{self.get_xyz_filename()}_{timestamp}.txt"
        full_path = os.path.join(output_dir, save_name)

        element_counts = {element: np.sum(self.elements == element) for element in np.unique(self.elements)}
        element_counts_str = ', '.join([f"[{element}: {count}]" for element, count in element_counts.items()])

        variables = {
            'coords': self.coords,
            'elements': self.elements,
            'QRange': self.QRange,
            # 'adps': self.adps,
            'debye_scatterers': self.debye_scatterers,
            'debye_elements': self.debye_elements,
            'sq_vals': self.sq_vals
        }

        metadata = {
            "XYZ File Path": self.xyzPath,
            "Date and Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Element Count": element_counts_str,
            "Calculation Time (s)": f"{self.calc_time:.3f} s ({self.calc_time * 1000:.3f} ms)"
        }

        statistics = {
            "Calculation Start Time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "Calculation End Time": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "Total Calculation Time (s)": f"{self.calc_time:.3f} s ({self.calc_time * 1000:.3f} ms)",
            "Number of Atoms": len(self.elements),
            "Q Range": f"{self.QInput[0]} to {self.QInput[1]} with step size {self.QInput[2]}" if isinstance(self.QInput, list) else f"Loaded from {self.QInput}"
        }

        try:
            with open(full_path, 'w') as file:
                file.write("Metadata:\n")
                for key, value in metadata.items():
                    file.write(f"{key}: {value}\n")
                file.write("\nStatistics:\n")
                for key, value in statistics.items():
                    file.write(f"{key}: {value}\n")
                file.write("\nVariable Statistics:\n")
                file.write("Note: These values represent the same memory on different scales.\n")
                file.write(f"{'Variable':<20} {'Data Type':<20} {'Size (bytes)':<20} {'Size (MB)':<20} {'Size (GB)':<20}\n")
                for var_name, var_value in variables.items():
                    size_bytes = sys.getsizeof(var_value)
                    size_mb = size_bytes / (1024 ** 2)
                    size_gb = size_bytes / (1024 ** 3)
                    file.write(f"{var_name:<20} {type(var_value).__name__:<20} {size_bytes:<20} {size_mb:<20.6f} {size_gb:<20.6f}\n")
                # if self.adps is not None:
                #     file.write("\nADPs:\n")
                #     np.savetxt(file, self.adps, fmt='%.6f')
                file.write("\nTemporary Variables and Specific Operations Statistics:\n")
                for var_name, stats in self.memory_and_time_stats.items():
                    size_mb = stats['size_bytes'] / (1024 ** 2)
                    size_gb = stats['size_bytes'] / (1024 ** 3)
                    time_ms = stats['time_seconds'] * 1000
                    file.write(f"{var_name:<30} Size (bytes): {stats['size_bytes']:<20} Size (MB): {size_mb:<20.6f} Size (GB): {size_gb:<20.6f} Time (ms): {time_ms:.3f}\n")
            print(f"Metadata and statistics successfully saved to {full_path}")
        except IOError as e:
            print(f"An error occurred while writing the metadata and statistics file: {e}")

    def copy_xyz_file(self, output_dir):
        """
        Copies the original XYZ file to the output directory.

        Parameters:
        - output_dir (str): The output directory where the file will be copied.
        """
        try:
            shutil.copy(self.xyzPath, output_dir)
            print(f"XYZ file successfully copied to {output_dir}")
        except IOError as e:
            print(f"An error occurred while copying the XYZ file: {e}")

    def convert_sq_to_gr(self, SQ, Q_values, r_values=np.linspace(0.5, 30, 1000)):
        """
        Convert structure factor S(Q) to radial distribution function G(r) using the provided equation.
        
        Parameters:
            SQ (array): Structure factor values.
            Q_values (array): Values of Q.
            r_values (array, optional): Values of r.
        
        Returns:
            GR (array): Radial distribution function values.
        """
        GR = np.zeros_like(r_values)
        for i, r in enumerate(r_values):
            integrand = Q_values * (SQ - 1) * np.sin(Q_values * r)
            GR[i] = np.trapz(integrand, Q_values) * 2 / np.pi
        return GR

    def gaussian_filter_gr(self, GR, sigma=4):
        """
        Apply a Gaussian filter to the G(r) values.
        
        Parameters:
            GR (array): Radial distribution function values.
            sigma (float, optional): Standard deviation for Gaussian kernel.
        
        Returns:
            smoothed_GR (array): Smoothed G(r) values.
        """
        smoothed_GR = gaussian_filter1d(GR, sigma)
        return smoothed_GR

    def plot_gr(self, r_values, GR, smoothed_GR, save_path, overlay=False):
        """
        Plot r vs. G(r) and save the plot as a PNG file.
        
        Parameters:
            r_values (array): Values of r.
            GR (array): Radial distribution function values.
            smoothed_GR (array): Smoothed G(r) values.
            save_path (str): Path to save the plot PNG file.
            overlay (bool, optional): If True, overlay both G(r) and smoothed G(r) on the same plot.
        """
        plt.figure(figsize=(8, 6))
        if overlay:
            plt.plot(r_values, GR, label='G(r)')
        plt.plot(r_values, smoothed_GR, label='Smoothed G(r)')
        plt.xlabel('r (Å)')
        plt.ylabel('G(r) (Å$^{-2}$)')
        plt.title('Pair Distribution Function')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def save_gr(self, r_values, GR, smoothed_GR, output_dir, timestamp, sigma):
        """
        Save the r and G(r) values to a .gr file.
        
        Parameters:
            r_values (array): Values of r.
            GR (array): Radial distribution function values.
            smoothed_GR (array): Smoothed G(r) values.
            output_dir (str): The output directory where the file will be saved.
            timestamp (str): The timestamp to include in the filename.
            sigma (float): The sigma value used for Gaussian filtering.
        """
        save_name = f"{self.get_xyz_filename()}_{timestamp}.gr"
        full_path = os.path.join(output_dir, save_name)

        date_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = f"#L r(Å)\tG_unfiltered(Å$^{-2}$)\tG(Å$^{-2}$)\nDate and Time: {date_time_str}\nXYZ File Path: {self.xyzPath}\n"
        if hasattr(self, 'QInputFile'):
            header += f"Q Input File: {self.QInputFile}\n"
        header += f"Sigma: {sigma}\n"

        data_lines = "\n".join([f"{r:.3f}\t{gr:.6e}\t{smoothed_gr:.6e}" for r, gr, smoothed_gr in zip(r_values, GR, smoothed_GR)])

        try:
            with open(full_path, 'w') as file:
                file.write(header + data_lines)
            print(f"G(r) data successfully saved to {full_path}")
        except IOError as e:
            print(f"An error occurred while writing the file: {e}")

    def plot_scattering_and_gr(self, r_values, smoothed_GR, save_path):
        """
        Plot S(Q) and smoothed G(r) in subplots and save the figure as a PNG file.

        Parameters:
            r_values (array): Values of r.
            smoothed_GR (array): Smoothed G(r) values.
            save_path (str): Path to save the plot PNG file.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(self.get_xyz_filename())

        # Plot S(Q)
        ax1.plot(self.QRange, self.sq_vals, marker='.', linestyle='-', markersize=3, color='mediumslateblue')
        ax1.set_xlabel('q ($\AA^{-1}$)')
        ax1.set_ylabel('S(Q)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title('Scattering Data Comparison')

        # Plot G(r)
        ax2.plot(r_values, smoothed_GR, label='Smoothed G(r)')
        ax2.set_xlabel('r (Å)')
        ax2.set_ylabel('G(r) (Å$^{-2}$)')
        ax2.set_title('Pair Distribution Function')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def track_memory_usage(self, message):
        """
        Track and print the current memory usage with a custom message.

        Parameters:
        - message (str): Custom message to print along with the memory usage.
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"{message}: RSS={mem_info.rss / (1024 ** 2):.2f} MB, VMS={mem_info.vms / (1024 ** 2):.2f} MB")

    def run(self):
        """
        Executes the computation, generates the output folder, and saves all the output files.
        """
        self.debyecalc()
        output_dir, timestamp = self.create_output_directory()
        self.copy_xyz_file(output_dir)
        self.saveData(output_dir, timestamp)
        self.save_metadata_and_statistics(output_dir, timestamp)
        self.plot_scattering(save_path=os.path.join(output_dir, f"fig_{self.get_xyz_filename()}_{timestamp}.png"))
        
        # New G(r) related operations
        r_values = np.linspace(0.5, 30, 1000)
        GR = self.convert_sq_to_gr(self.sq_vals, self.QRange, r_values)
        sigma = 4
        smoothed_GR = self.gaussian_filter_gr(GR, sigma=sigma)
        self.save_gr(r_values, GR, smoothed_GR, output_dir, timestamp, sigma)
        self.plot_gr(r_values, GR, smoothed_GR, save_path=os.path.join(output_dir, f"fig_gr_{self.get_xyz_filename()}_{timestamp}.png"), overlay=True)
        self.plot_gr(r_values, GR, smoothed_GR, save_path=os.path.join(output_dir, f"fig_smoothed_gr_{self.get_xyz_filename()}_{timestamp}.png"))
        self.plot_scattering_and_gr(r_values, smoothed_GR, save_path=os.path.join(output_dir, f"fig_scattering_and_gr_{self.get_xyz_filename()}_{timestamp}.png"))

def main():
    parser = argparse.ArgumentParser(description='Debye Calculation Script')
    parser.add_argument('xyzPath', type=str, help='Path to the .xyz file')
    parser.add_argument('QInput', type=str, help='Q input values: either a string representing a list of three floats [qmin, qmax, qstep] or a path to a .iq file')
    # parser.add_argument('--adps', type=str, help='Path to the ADPs file', default=None)
    parser.add_argument('--suppress_plots', action='store_true', help='Suppress plot display')
    parser.add_argument('--log_scale', action='store_true', help='Use logarithmic scale for QInput')

    args = parser.parse_args()

    # Check if QInput is a single path to an .iq file or a string representing a list of three float values
    try:
        QInput = ast.literal_eval(args.QInput)
        if not isinstance(QInput, list) or len(QInput) != 3:
            raise ValueError
        # QInput = [float(q) for q in QInput]
    except (ValueError, SyntaxError):
        if args.QInput.endswith('.iq'):
            QInput = args.QInput
        else:
            raise ValueError("QInput must be either a string representing a list of three float values [qmin, qmax, qstep] or a path to a .iq file")

    # adps = None
    # if args.adps:
    #     try:
    #         adps = np.loadtxt(args.adps)
    #     except Exception as e:
    #         print(f"Error reading ADPs file: {e}")
    #         adps = None

    debye_calc = DebyeCalc(args.xyzPath, QInput, args.suppress_plots, args.log_scale)
    debye_calc.run()

if __name__ == "__main__":
    main()