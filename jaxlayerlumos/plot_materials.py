import numpy as np
import os
import matplotlib.pyplot as plt
# import yaml
import csv
import pandas as pd


# path_figures = constants_src.path_figures
path_materials = 'materials'

def load_csv(str_file_csv):
    if str_file_csv is None:
        return None

    # Read the file content
    with open(os.path.join(path_materials, str_file_csv), 'r') as f:
        lines = f.readlines()

    # Initialize lists for n and k data
    df_n = []
    df_k = []
    current_section = None

    for line in lines:
        line = line.strip()
        if line.lower() == 'wl,n':
            current_section = 'n'
            continue
        elif line.lower() == 'wl,k':
            current_section = 'k'
            continue
        elif line == '':
            continue  # Skip empty lines
        else:
            if current_section == 'n':
                df_n.append(line)
            elif current_section == 'k':
                df_k.append(line)

    # Process n data
    if df_n:
        df_n = pd.DataFrame([line.split(',') for line in df_n], columns=['wl', 'n'])
        df_n['wl'] = pd.to_numeric(df_n['wl'], errors='coerce')
        df_n['n'] = pd.to_numeric(df_n['n'], errors='coerce')
        df_n.dropna(inplace=True)
    else:
        print(f"No 'n' data found in {str_file_csv}. Skipping.")
        return None

    # Process k data
    if df_k:
        df_k = pd.DataFrame([line.split(',') for line in df_k], columns=['wl', 'k'])
        df_k['wl'] = pd.to_numeric(df_k['wl'], errors='coerce')
        df_k['k'] = pd.to_numeric(df_k['k'], errors='coerce')
        df_k.dropna(inplace=True)
    else:
        print(f"No 'k' data found in {str_file_csv}. Skipping.")
        return None

    # Merge n and k data on wavelength
    df = pd.merge(df_n, df_k, on='wl', how='inner')
    df.sort_values('wl', inplace=True)

    if df.empty:
        print(f"No overlapping 'wl' data in {str_file_csv}. Skipping.")
        return None

    # Convert wavelengths from micrometers (μm) to nanometers (nm)
    wavelengths_nm = df['wl'].values * 1000

    # Prepare the data_w_n_k array: [wavelength_nm, n, k, zeros]
    zeros = np.zeros(len(wavelengths_nm))
    data_w_n_k = np.column_stack((wavelengths_nm, df['n'].values, df['k'].values, zeros))

    return data_w_n_k

def load_file(str_file):
    # if '.yml' in str_file:
        # data_w_n_k = load_yml(str_file)
    # el
    if '.csv' in str_file:
        data_w_n_k = load_csv(str_file)
    else:
        raise ValueError

    return data_w_n_k
