import subprocess
import time
import psutil
import os

# Path to Python interpreter in the virtual environment
python_path = os.path.join(".venv", "Scripts", "python.exe")

def run_script(script_name):
    return subprocess.Popen([python_path, script_name])

def terminate_process_and_children(process):
    try:
        parent = psutil.Process(process.pid)
        # Attempt to terminate all child processes
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
        parent.wait(timeout=5)
    except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
        # Force kill if termination fails
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()

while True:
    print("Running 2-terminate_aspen.py")
    run_script("2-terminate_aspen.py").wait()

    time.sleep(5)  # Brief pause to ensure processes are fully terminated

    print("Running 3-generate_remain_files.py")
    run_script("3-generate_remain_files.py").wait()

    print("Running 4-simulate.py")
    simulate_process = run_script("4-simulate.py")

    time.sleep(900)

    print("Stopping 4-simulate.py and its subprocesses")
    terminate_process_and_children(simulate_process)

    print("Restarting sequence...\n")
    time.sleep(10)
