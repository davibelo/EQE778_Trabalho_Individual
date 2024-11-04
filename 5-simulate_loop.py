import os
import subprocess
import psutil
import time
from datetime import datetime, timedelta

# Define the process names for Aspen Plus executables
process_names = ["apmain", "AspenPlus"]

# Function to log messages with timestamp
def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")

# Function to forcefully terminate Aspen Plus processes
def terminate_aspen_processes():
    write_log("Starting forceful termination of Aspen Plus processes.")
    for process_name in process_names:
        found = False
        for process in psutil.process_iter(['name', 'pid']):
            if process.info['name'] and process_name.lower() in process.info['name'].lower():
                found = True
                try:
                    write_log(f"Found process {process.info['name']} with ID {process.info['pid']}. Terminating.")
                    process.terminate()
                    process.wait(timeout=5)  # Wait briefly for termination
                    write_log(f"Process ID {process.info['pid']} ({process.info['name']}) has been terminated.")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    write_log(f"Failed to terminate process ID {process.info['pid']} ({process.info['name']}): {e}")
        if not found:
            write_log(f"No processes found for {process_name}.")
    write_log("Aspen Plus process termination check completed.")

# Function to run other scripts in the same folder
def run_scripts():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    write_log("Running 2-generate_remain_files.py")
    subprocess.run(["python", os.path.join(current_directory, "2-generate_remain_files.py")])
    write_log("Running 3-simulate.py")
    subprocess.run(["python", os.path.join(current_directory, "3-simulate.py")])

# Main loop to run every minute for one hour, then restart
start_time = datetime.now()
end_time = start_time + timedelta(minutes=1)

while datetime.now() < end_time:
    terminate_aspen_processes()
    run_scripts()
    time.sleep(60)  # Wait for one minute before repeating

# After one hour, restart the script by calling itself
write_log("Restarting script after one hour.")
subprocess.Popen(["python", __file__])
