import os
import shutil
import csv
import concurrent.futures
from multiprocessing import Manager, freeze_support, Process
import win32com.client as win32
import time

INPUT_FILE_TEMPLATE = 'remain_sim_points_part_{}.csv'
RESULT_FILE_TEMPLATE = 'simulation_results_part_{}.csv'
ERROR_FILE_TEMPLATE = 'error_sim_points_part_{}.csv'  # Error file template
SIMULATION_FILE = r'UTAA_run\UTAA_revK.bkp'
NUM_INSTANCES = 4

aspen_Path = os.path.abspath(SIMULATION_FILE)

def log_worker(log_queue, log_file_path):
    """Worker that listens to log_queue and writes error messages to the log file."""
    buffer = []
    flush_interval = 1  # seconds
    last_flush_time = time.time()

    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        while True:
            try:
                # Timeout allows us to periodically flush the buffer
                message = log_queue.get(timeout=flush_interval)
                if message == "STOP":
                    break
                buffer.append(message)
                print(message)  # Limited console output for monitoring
            except:
                pass  # Timeout reached; proceed to flush buffer

            # Periodic buffer flush
            current_time = time.time()
            if current_time - last_flush_time >= flush_interval or len(buffer) > 100:
                if buffer:
                    log_file.write('\n'.join(buffer) + '\n')
                    log_file.flush()
                    buffer.clear()
                    last_flush_time = current_time

def log_message_factory(log_queue):
    """Creates a logging function that adds only error messages to the log queue."""
    def log_message(message, is_error=False):
        if is_error:
            log_queue.put(message)
        else:
            print(message)  # Print successful runs to the console only
    return log_message

def load_points_from_csv(filename):
    """Loads points to be simulated from a CSV file."""
    points = []
    with open(filename, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header row
        for row in reader:
            points.append(tuple(map(float, row)))
    return points

def save_result_to_csv(result, result_file):
    """Appends a single result row to the specified results CSV file."""
    with open(result_file, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        if os.path.getsize(result_file) == 0:
            writer.writerow(['feedNH3', 'feedH2S', 'feedH20', 'QN1', 'QN2', 'QC', 'SF', 'H2S_ppm', 'NH3_ppm'])
        writer.writerow(result)

def save_error_to_csv(point, error_file):
    """Logs an error point to the specified error CSV file."""
    with open(error_file, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        if os.path.getsize(error_file) == 0:
            writer.writerow(['feedNH3', 'feedH2S', 'feedH20', 'QN1', 'QN2', 'QC', 'SF'])
        writer.writerow(point)

def start_aspen(instance_id, log_message):
    """Starts Aspen using the instance-specific copy of the simulation file and returns the Application object."""
    instance_file = f'{aspen_Path}_copy_{instance_id}.bkp'
    if not os.path.exists(instance_file):
        shutil.copyfile(aspen_Path, instance_file)

    try:
        Application = win32.Dispatch('Apwn.Document')
        Application.InitFromArchive2(instance_file)
        Application.visible = 0
        log_message(f"Aspen started successfully for instance {instance_id}.")
        return Application
    except Exception as e:
        log_message(f"Failed to start Aspen for instance {instance_id}: {e}", is_error=True)
        return None

def simulate(x, Application, log_message, lock, result_file, error_file):
    feedNH3, feedH2S, feedH20, QN1, QN2, QC, SF = x

    if not Application:
        Application = start_aspen(log_message)
        if not Application:
            log_message(f"Failed to initialize Aspen for inputs {x}. Skipping simulation.", is_error=True)
            return None

    try:
        # Set values in Aspen
        Application.Tree.FindNode(r"\Data\Streams\CARGA3\Input\FLOW\MIXED\NH3").Value = feedNH3
        Application.Tree.FindNode(r"\Data\Streams\CARGA3\Input\FLOW\MIXED\H2S").Value = feedH2S
        Application.Tree.FindNode(r"\Data\Streams\CARGA3\Input\FLOW\MIXED\H2O").Value = feedH20
        Application.Tree.FindNode(r"\Data\Blocks\T1\Input\QN").Value = QN1
        Application.Tree.FindNode(r"\Data\Blocks\T2\Input\QN").Value = QN2
        Application.Tree.FindNode(r"\Data\Blocks\T2\Input\Q1").Value = QC
        Application.Tree.FindNode(r"\Data\Blocks\SPLIT1\Input\FRAC\AGUAPR5A").Value = max(0, SF)

        # Run simulation
        Application.Engine.Run2()

        # Collect results
        cH2S = Application.Tree.FindNode(r"\Data\Streams\AGUAR1\Output\MASSFRAC\MIXED\H2S").Value
        cNH3 = Application.Tree.FindNode(r"\Data\Streams\AGUAR1\Output\MASSFRAC\MIXED\NH3").Value
        cH2S_ppm = cH2S * 1E6
        cNH3_ppm = cNH3 * 1E6
        y = cH2S_ppm, cNH3_ppm

        # Log result and save to results file
        log_message(f"Simulation result: {x} -> H2S: {cH2S_ppm}, NH3: {cNH3_ppm}")
        with lock:
            save_result_to_csv(x + y, result_file)

        return x + y

    except Exception as e:
        error_message = str(e)
        log_message(f"Error simulating {x}: {error_message}", is_error=True)        
        if 'NoneType' in error_message:
            save_error_to_csv(x, error_file)
        return None

def run_parallel_simulations(batch_id, input_file, log_queue, lock):
    log_message = log_message_factory(log_queue)
    result_file = RESULT_FILE_TEMPLATE.format(batch_id)
    error_file = ERROR_FILE_TEMPLATE.format(batch_id)
    Application = start_aspen(batch_id, log_message)
    results = []

    # Load points from the specific input file
    points = load_points_from_csv(input_file)

    for point in points:
        result = simulate(point, Application, log_message, lock, result_file, error_file)
        if result:
            results.append(result)

    if Application:
        Application.Close()
    return results

if __name__ == '__main__':
    freeze_support()

    # Define the path to the log file
    log_file_path = f"{os.path.splitext(__file__)[0]}.log"

    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # Create a manager and lock for synchronizing file access
    manager = Manager()
    lock = manager.Lock()

    # Set up a global logging queue and process
    log_queue = manager.Queue()
    log_process = Process(target=log_worker, args=(log_queue, log_file_path))
    log_process.start()

    # Run simulations in parallel for each input file
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_INSTANCES) as executor:
        futures = [
            executor.submit(run_parallel_simulations, i + 1, INPUT_FILE_TEMPLATE.format(i + 1), log_queue, lock)
            for i in range(NUM_INSTANCES)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure exceptions in workers are raised

    # Stop the logging process
    log_queue.put("STOP")
    log_process.join()

    print('Simulation results saved in multiple result files.')
