import csv


def read_and_save(str_file, file_python, str_variable):
    print(str_file, str_variable)
    cmfs = []

    with open(str_file) as file_csv:
        reader = csv.reader(file_csv, delimiter=",")

        for row in reader:
            assert len(row) == 4

            cmfs.append([row[0], row[1], row[2], row[3]])

    assert len(cmfs) == 471
    file_python.write(f"{str_variable} = [\n")

    for row in cmfs:
        file_python.write("    [")
        file_python.write(", ".join(row))
        file_python.write("],\n")

    file_python.write("]\n")


if __name__ == "__main__":
    file_python = open("../jaxlayerlumos/colors/color_matching_functions.py", mode="w")

    read_and_save("csv/ciexyz31_1.csv", file_python, "cmfs_cie1931")

    file_python.close()
