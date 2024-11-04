import subprocess
import time
import os
import signal

def run_script(script_name):
    return subprocess.Popen(["python", script_name])

def terminate_process(process):
    if process.poll() is None:  # Check if process is still running
        process.terminate()
        try:
            process.wait(timeout=5)  # Give it a few seconds to terminate gracefully
        except subprocess.TimeoutExpired:
            process.kill()  # Force kill if not terminated

while True:
    print("Running 2-terminate_aspen.py")
    run_script("2-terminate_aspen.py").wait()
    
    print("Running 3-generate_remain_files.py")
    run_script("3-generate_remain_files.py").wait()
    
    print("Running 4-simulate.py")
    simulate_process = run_script("4-simulate.py")
    
    # Wait for 1 minute
    time.sleep(60)
    
    print("Stopping 4-simulate.py")
    terminate_process(simulate_process)
    
    print("Restarting sequence...\n")
