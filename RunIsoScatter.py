import numpy as np
import glob
import os
import argparse
import time

from utilities import parse_config_file, save_config_to_txt
from IsoScatter import xyz_simulation

def main(config):
    # Input Parameters
    input_folder = config.get('input_folder', None)
    input_filepath = config.get('input_filepath', None)
    filetype = config.get('filetype', 'xyz')
    gen_name = config.get('gen_name')
    q_min = float(config.get('q_voxel_size', 0.01))
    q_max = float(config.get('q_voxel_size', 0.01))
    num_qs = float(config.get('q_voxel_size', 0.01))
    logspace_q = config.get('log_q', False)
    energy = float(config.get('energy', 1))
    output_dir = config.get('output_dir', os.getcwd())
    hull_method = config.get('hull_method', False)
    vol_pct = float(config.get('vol_pct', 0))
    solvent_edens = float(config.get('solvent_edens', 0))

    if input_folder:
        input_paths = glob.glob(f'{input_folder}/*{filetype}')
    elif input_filepath:
        input_paths = [input_filepath]
    else:
        raise Exception('Either input_folder or input_path must be specified')
    
    if logspace_q:
        q_min = np.log10(q_min)
        q_max = np.log10(q_max)
        qs = np.logspace(q_min, q_max, num_qs)
    else:
        qs = np.linspace(q_min, q_max, num_qs)
        
    for i, input_path in enumerate(input_paths):
        iq, q_vals = xyz_simulation(input_path,
                                    qs, 
                                    vol_pct, 
                                    solvent_edens,
                                    energy, 
                                    hull_method=hull_method)
        if i==0:
            iq_avg = iq
        else:
            iq_avg += iq
    
    iq_avg /= len(input_paths)

    # Save
    save_path = f'{output_dir}/{gen_name}_output_files'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    np.save(f'{save_path}/{gen_name}_iq.npy', iq_avg)
    np.save(f'{save_path}/{gen_name}_q_vals.npy', q_vals)

    #add some plotting later?

    save_config_to_txt(config, f'{save_path}/{gen_name}_config.txt')

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="Process a configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    config_path = args.config
    config = parse_config_file(config_path)
    main(config)
    end = time.time()
    runtime = end-start
    print(f'\nTotal Time: {str(runtime)}')