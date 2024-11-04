import psutil

# Define the process names for Aspen Plus executables
process_names = ["apmain", "AspenPlus"]

# Function to forcefully terminate Aspen Plus processes
def terminate_aspen_processes():
    for process_name in process_names:
        for process in psutil.process_iter(['name', 'pid']):
            if process.info['name'] and process_name.lower() in process.info['name'].lower():
                try:
                    process.terminate()  # Attempt to terminate the process
                    process.wait(timeout=5)  # Wait briefly for termination
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # Ignore if the process does not exist or access is denied

# Run the termination function
if __name__ == "__main__":
    terminate_aspen_processes()
