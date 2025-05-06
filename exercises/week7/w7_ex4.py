import sys
from time import perf_counter as time
from pyarrow import csv
import pyarrow.parquet as pq
import pandas as pd

from w7_ex1 import summarize_columns

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

    table = csv.read_csv(filename + ".csv")
    df = table.to_pandas()

    summarize_columns(df)

    #total = total_precip(df)  # Very slow, order of minutes
    #total = total_precip_v2(df)  # faster, 51 sec
    total = total_precip_v3(df)  # Much faster, 0.4 sec
