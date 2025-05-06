from time import perf_counter as time
import pandas as pd
from pyarrow import csv
import pyarrow.parquet as pq

def summarize_columns(df):
    print(pd.DataFrame([
        (
            c,
            df[c].dtype,
            len(df[c].unique()),
            df[c].memory_usage(deep=True) // (1024**2)
        ) for c in df.columns
    ], columns=['name', 'dtype', 'unique', 'size (MB)']))
    print('Total size:', df.memory_usage(deep=True).sum() / 1024**2, 'MB')


def test_load_speeds(filename="2023_01"):

    start = time()
    df = pd.read_csv(filename + ".csv.zip")
    print(f"Read csv.zip took {time() - start} sec.")

    start = time()
    df = pd.read_csv(filename + ".csv")
    print(f"Read csv raw took {time() - start} sec.")

    start = time()
    table = csv.read_csv(filename + ".csv")
    print(f"Read csv PyArrow took {time() - start} sec.")
    #pq.write_table(table, filename + ".parquet")

    start = time()
    table = pq.read_table(filename + ".parquet")  # Read with PyArrow Parquet
    print(f"Read csv PyArrow Parquet took {time() - start} sec.")

    return df, table

def df_memsize(df):

    # Two faster options:
    #mem = df.info(memory_usage='deep')
    #mem = df.memory_usage(deep=True).sum() // 1024

    mem = 0
    for c in df.columns:
        mem += df[c].memory_usage(deep=True) // 1024
    return mem

def reduce_dmi_df(df):

    start = time()
    # Optimize category values
    df['parameterId'] = df['parameterId'].astype('category')
    df['stationId'] = df['stationId'].astype('category')

    # Optimize datetime values
    df['created'] = pd.to_datetime(df['created'], format='ISO8601')
    df['observed'] = pd.to_datetime(df['observed'], format='ISO8601')
    print(f"Conversion took {time() - start} sec.")

    return df

if __name__ == "__main__":

    run_load_speed_test = False

    if run_load_speed_test:
        df, table = test_load_speeds("2023_01")
        print("Memory usage", df_memsize(df))
        summarize_columns(df)

    #df = pd.read_csv("2023_01.csv.zip")

    table = csv.read_csv("2023_01.csv")  # Read with PyArrow
    df = table.to_pandas()  # Convert back to pandas

    summarize_columns(df)

    df = reduce_dmi_df(df)

    summarize_columns(df)