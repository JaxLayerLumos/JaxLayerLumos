from tmm.tmm_core import coh_tmm, unpolarized_RT, ellips, position_resolved, find_in_structure_with_inf
from numpy import pi, linspace, inf, array
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

material_nk_data = array([[200, 2.1+0.1j],
                            [300, 2.4+0.3j],
                            [400, 2.3+0.4j],
                            [500, 2.2+0.4j],
                            [750, 2.2+0.5j]])
material_nk_fn = interp1d(material_nk_data[:,0].real,
                            material_nk_data[:,1], kind='quadratic')



d_list = [inf, 300, inf] #in nm
lambda_list = linspace(200, 750, 400) #in nm
T_list = []
for lambda_vac in lambda_list:
    # print(material_nk_fn(lambda_vac))
    n_list = [1, material_nk_fn(lambda_vac), 1]
    # print(n_list)
    # print(d_list)
    T_list.append(coh_tmm('s', n_list, d_list, 0, lambda_vac)['T'])
plt.figure()
plt.plot(lambda_list, T_list)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Fraction of power transmitted')
plt.title('Transmission at normal incidence')
# plt.show()