import csv


def read_and_save(str_file, file_python, str_variable):
    print(str_file, str_variable)
    illuminant = []

    with open(str_file) as file_csv:
        reader = csv.reader(file_csv, delimiter=",")

        for row in reader:
            assert len(row) == 2

            illuminant.append([row[0], row[1]])

    assert len(illuminant) == 531
    file_python.write(f"{str_variable} = [\n")

    for row in illuminant:
        file_python.write("    [")
        file_python.write(", ".join(row))
        file_python.write("],\n")

    file_python.write("]\n")


if __name__ == "__main__":
    file_python = open("../jaxlayerlumos/colors/illuminants.py", mode="w")

    read_and_save("csv/CIE_std_illum_D65.csv", file_python, "illuminant_d65")

    file_python.close()
