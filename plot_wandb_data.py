import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_memory_utilization():
    plt.rcParams.update({'font.family': 'Times'})

    # Replace 'your_file.csv' with the actual path to your CSV file
    # csv_path = 'results/wandb_process_memory.csv'
    csv_path = 'results/wandb_VoDaSuRe_RAM.csv'

    # Read the CSV file, handle quoted values properly
    df = pd.read_csv(csv_path, quotechar='"')

    # Extract the relevant columns
    time_col = 'Relative Time (Process)'
    util_col = 'RRDBNet3D_000_bs16 - system/proc.memory.rssMB'

    # Convert columns to appropriate types
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df[util_col] = pd.to_numeric(df[util_col], errors='coerce')
    print("Average RAM usage: ", df[util_col].mean(), "MB")

    # Drop any rows with NaNs just in case
    df.dropna(subset=[time_col, util_col], inplace=True)

    N = len(df)
    sliding_window_size = 200  # Adjust this size as needed
    sliding_window_avg = np.convolve(df[util_col], np.ones(sliding_window_size) / sliding_window_size, mode='valid')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_col] / 60 ** 2, df[util_col])
    plt.plot(df[time_col][:N - sliding_window_size + 1] / 60 ** 2, sliding_window_avg)
    plt.xlabel('Relative process time [hours]', fontsize=16)
    plt.ylabel('RAM usage [MB]', fontsize=16)
    plt.title('RAM usage', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(['RAM usage', f'Sliding Window Avg'], loc='best', fontsize=14)

    # Save the plot
    plt.savefig('figures/VoDaSuRe_memory_utilization_over_time.pdf', dpi=300, bbox_inches='tight')

def plot_gpu_utilization():
    plt.rcParams.update({'font.family': 'Times'})

    # Replace 'your_file.csv' with the actual path to your CSV file
    # csv_path = 'results/wandb_export_2025-06-18T16_35_25.325+02_00.csv'
    csv_path = 'results/wandb_VoDaSuRe_GPU.csv'

    # Read the CSV file, handle quoted values properly
    df = pd.read_csv(csv_path, quotechar='"')

    # Extract the relevant columns
    time_col = 'Relative Time (Process)'
    util_col = 'RRDBNet3D_000_bs16 - system/gpu.0.gpu'

    # Convert columns to appropriate types
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df[util_col] = pd.to_numeric(df[util_col], errors='coerce')
    print("Average GPU usage: ", df[util_col].mean(), "%")

    # Drop any rows with NaNs just in case
    df.dropna(subset=[time_col, util_col], inplace=True)

    N = len(df)
    sliding_window_size = 200  # Adjust this size as needed
    sliding_window_avg = np.convolve(df[util_col], np.ones(sliding_window_size)/sliding_window_size, mode='valid')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_col] / 60**2, df[util_col])
    plt.plot(df[time_col][:N - sliding_window_size + 1] / 60**2, sliding_window_avg)
    plt.xlabel('Relative process time [hours]', fontsize=16)
    plt.ylabel('GPU utilization (%)', fontsize=16)
    plt.title('GPU utilization', fontsize=18)
    plt.ylim(0, 100)  # Set y-axis limits to 0-100%
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(['GPU utilization', f'Sliding Window Avg'], loc='best', fontsize=14)

    # Save the plot
    plt.savefig('figures/VoDaSuRe_gpu_utilization_over_time.pdf', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    plot_memory_utilization()
    plot_gpu_utilization()
    print("Plots generated successfully.")