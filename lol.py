import csv
from natsort import natsorted

def read_csv_file(file_path) -> list:
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        row_list = []
        for row in csv_reader:
            row_list.append(row)
        return row_list

def write_matrix_to_csv(matrix, file_path):
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(matrix)

# Example usage
csv_file_path = 'img-class-sorted.csv'
matrix = read_csv_file(csv_file_path)

matrix = [[f"card_{151 + i}.png"] + sublist[1:] for i, sublist in enumerate(matrix)]

write_matrix_to_csv(matrix, "img-class-sorted-renamed.csv")
