import pynvml
import csv

# Initialize NVML
pynvml.nvmlInit()

# Prepare CSV file for writing
csv_filename = "gpu_max_memory_clock_frequencies.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["GPU_Index", "GPU_Name", "Max_Memory_Clock_MHz", "Graphics_Clock_MHz"])

    # Get the number of GPUs available
    device_count = pynvml.nvmlDeviceGetCount()

    # Loop through each GPU and get only the graphics clocks for the highest memory clock
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

        # Get supported memory clocks and select the highest one
        memory_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
        max_memory_clock = max(memory_clocks)

        # Get the supported graphics clocks for the highest memory clock
        graphics_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, max_memory_clock)
        
        # Write each graphics clock for the max memory clock to the CSV
        for graphics_clock in graphics_clocks:
            writer.writerow([i, gpu_name, max_memory_clock, graphics_clock])

print(f"Supported GPU frequencies for max memory clock saved to {csv_filename}")

# Test setting the GPU clock and reset all clocks if successful
try:
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        # Set to test clock to check permissions
        max_memory_clock = max(pynvml.nvmlDeviceGetSupportedMemoryClocks(handle))
        test_graphics_clock = min(pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, max_memory_clock))
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, test_graphics_clock, max_memory_clock)
        print(f"Successfully set GPU {i} to graphics clock {test_graphics_clock} MHz and memory clock {max_memory_clock} MHz.")
        
        # Reset all clocks to default after test
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        print(f"Successfully reset GPU {i} clock settings to default for both memory and graphics.")

except pynvml.NVMLError_NoPermission:
    print("Error: Insufficient permissions to modify GPU clock settings.")
    print("To run this script with the required permissions, try the following:")
    print("1. Run the script as root using 'sudo'.")
    print("2. Ensure your NVIDIA driver and NVML library are up to date.")
    print("3. If running inside Docker, start the container with elevated privileges using '--privileged'.")

# Shutdown NVML
pynvml.nvmlShutdown()
