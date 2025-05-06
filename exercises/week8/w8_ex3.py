import sys
from pyarrow import csv
import pyarrow.parquet as pq

# This file is wrong!!! See section 8.2.3 in Fast Python for how to save a chunked parquet file

def save_parquet_chunked(filename="2023_01", chunk_size=100000):

    # Load .csv using PyArrow
    table = csv.read_csv(filename + ".csv")

    # Save parquet file
    pq.write_table(table, filename + ".parquet", row_group_size=chunk_size)


if __name__ == "__main__":

    filename, chunk_size = sys.argv[1], int(sys.argv[2])

    save_parquet_chunked(filename, chunk_size)
    print(f"Saved parquet file: {filename}.parquet")