# Benchmarking

You should install the packages required for benchmarking our software against existing methods.
Before installing the packages, you should install PyTorch first.
In particular, you can install it using the following command.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For details, you can refer to [the official instruction of PyTorch](https://pytorch.org).
Then, the required packages can be installed by the following command.

```bash
pip install .[benchmarking]
```

## Comparisons to Ansys Optics

Simulation results of JaxLayerLumos are compared to the results of [stackrt](https://optics.ansys.com/hc/en-us/articles/360034406254-stackrt-Script-command), which is included in [Ansys Optics](https://www.ansys.com/products/optics).
Our results are matched to the Ansys Optics results with sufficiently small errors.

<p align="center">
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_100.0nm_angle_0.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_100.0nm_angle_45.0_deg.png" width="400" />
<br>
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_100.0nm_angle_75.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_100.0nm_angle_89.0_deg.png" width="400" />
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_10.0nm_Al_11.0nm_Ag_12.0nm_angle_0.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_10.0nm_Al_11.0nm_Ag_12.0nm_angle_45.0_deg.png" width="400" />
<br>
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_10.0nm_Al_11.0nm_Ag_12.0nm_angle_75.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_10.0nm_Al_11.0nm_Ag_12.0nm_angle_89.0_deg.png" width="400" />
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_TiO2_20.0nm_Ag_5.0nm_TiO2_30.0nm_angle_0.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_TiO2_20.0nm_Ag_5.0nm_TiO2_30.0nm_angle_45.0_deg.png" width="400" />
<br>
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_TiO2_20.0nm_Ag_5.0nm_TiO2_30.0nm_angle_75.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_TiO2_20.0nm_Ag_5.0nm_TiO2_30.0nm_angle_89.0_deg.png" width="400" />
</p>
