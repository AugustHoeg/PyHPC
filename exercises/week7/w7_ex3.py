import sys
from pyarrow import csv
import pyarrow.parquet as pq


def save_parquet(filename="2023_01"):

    # Load .csv using PyArrow
    table = csv.read_csv(filename + ".csv")

    # Save parquet file
    pq.write_table(table, filename + ".parquet")


if __name__ == "__main__":

    filename = sys.argv[1]

    save_parquet(filename)
    print(f"Saved parquet file: {filename}.parquet")