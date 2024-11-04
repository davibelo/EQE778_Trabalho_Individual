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
                    print(f"Terminated process {process.info['name']} (PID: {process.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"Failed to terminate process {process.info['name']} (PID: {process.info['pid']})")

# Run the termination function
if __name__ == "__main__":
    terminate_aspen_processes()
