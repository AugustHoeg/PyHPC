import sys
from time import perf_counter as time
from pyarrow import csv
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt


def total_precip_chunked(dfc):
    total = 0.0
    for i, df in enumerate(dfc):
        total += df['value'][df['parameterId'] == 'precip_past10min'].sum()
    return total


def total_precip(df):
    total = 0.0
    for i in range(len(df)):
        row = df.iloc[i]
        if row['parameterId'] == 'precip_past10min':
            total += row['value']
    return total


def total_precip_v2(df):
    start = time()
    total = df.apply(lambda row: row['value'] if row['parameterId'] == 'precip_past10min' else 0, axis=1).sum()
    print(f"total_precip using df.apply took {time() - start} sec.")
    return total


def total_precip_v3(df):
    start = time()
    total = df['value'][df['parameterId'] == 'precip_past10min'].sum()
    print(f"total_precip using vectorization took {time() - start} sec.")
    return total


if __name__ == "__main__":

    filename = sys.argv[1]

    chunk_sizes = [1000, 10000, 100000, 1000000]
    times = []

    for chunk_size in chunk_sizes:

        dfc = pd.read_csv(filename, chunksize=chunk_size)  # Read "chuck_size" rows at a time.
        start = time()
        total = total_precip_chunked(dfc)
        run_time = time() - start
        print(f"Total precip {total} w. chunk size {chunk_size} took {run_time} sec.")
        times.append(run_time)

    # Bar plot
    plt.figure()
    plt.bar(chunk_sizes, times)
    #plt.yscale('log')
    plt.ylabel("Run time [sec]")
    plt.xlabel("Chunk size [rows]")
    plt.savefig("runtime_vs_chunksize.pdf", dpi=300)

    #summarize_columns(df)

