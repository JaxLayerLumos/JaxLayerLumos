import matplotlib.pyplot as plt
import numpy as onp

from plot_R_frequency import read_csv_file

def plot_R_thickness():
    filename = "./data/LF.csv"
    #filename = "./data/CHF.csv"
    prefix = filename.split("/")[-1].replace(".csv", "")
    all_materials, d_stack, freq_range, inc_angles, polarization, R_db_list = read_csv_file(filename)
    total_thickness = onp.sum(d_stack*1e3, axis=1)
    R_db_flat = onp.array([r[0] for r in R_db_list])
    plt.plot(total_thickness, R_db_flat, marker='o', linestyle = 'None')

    plt.xlabel('Total Thickness (m)')
    plt.ylabel('Reflection (R_db)')
    plt.title('Total Thickness vs Reflection (R_db)')
    #plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    plot_R_thickness()



