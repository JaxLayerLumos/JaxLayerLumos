from jaxlayerlumos import utils_materials


if __name__ == "__main__":
    all_materials = utils_materials.get_all_materials()

    with open("../markdowns/MATERIALS.md", "w") as file_materials:
        file_materials.write("# Supported Materials")
        file_materials.write("\n")
        file_materials.write("\n")

        str_all_material = ", ".join(all_materials)
        file_materials.write(f"{len(all_materials)} materials: {str_all_material}")
        file_materials.write("\n")
        file_materials.write("\n")

        for material in all_materials:
            file_materials.write(f"- {material}")
            file_materials.write("\n")
            file_materials.write("\n")
            file_materials.write(
                f"<p align='center'><img src='../assets/refractive_indices/{material}.png' width='400' /></p>"
            )
            file_materials.write("\n")
            file_materials.write("\n")
