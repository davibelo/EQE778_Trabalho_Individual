import os
import shutil
import csv
import concurrent.futures
from multiprocessing import Manager, freeze_support
import win32com.client as win32

INPUT_FILE_TEMPLATE = 'simulation_points_part_{}.csv'
RESULT_FILE_TEMPLATE = 'simulation_results_part_{}.csv'
SIMULATION_FILE = r'UTAA_run\UTAA_revK.bkp'
NUM_INSTANCES = 8

aspen_Path = os.path.abspath(SIMULATION_FILE)

def start_aspen(instance_id, log_message):
    """Starts Aspen using the instance-specific copy of the simulation file and returns the Application object."""
    instance_file = f'{aspen_Path}_copy_{instance_id}.bkp'
    if not os.path.exists(instance_file):
        shutil.copyfile(aspen_Path, instance_file)  # Create a copy for the instance

    try:
        Application = win32.Dispatch('Apwn.Document')
        Application.InitFromArchive2(instance_file)
        Application.visible = 0
        log_message(f"Aspen started successfully for instance {instance_id}.")
        return Application
    except Exception as e:
        log_message(f"Failed to start Aspen for instance {instance_id}: {e}")
        return None

def simulate(x, Application, log_message, lock, result_file, input_file):
    feedNH3, feedH2S, feedH20, QN1, QN2, QC, SF = x

    if not Application:
        Application = start_aspen(log_message)
        if not Application:
            log_message(f"Failed to initialize Aspen for inputs {x}. Skipping simulation.")
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

        # Remove point from input file
        with lock:
            remove_point_from_csv(input_file, x)
        return x + y

    except Exception as e:
        log_message(f"Error simulating {x}: {e}")
        return None

def log_message_factory(log_file_path):
    def log_message(message):
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + '\n')
            print(message)
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
    with open(result_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if os.path.getsize(result_file) == 0:  # Write header only if file is empty
            writer.writerow(['feedNH3', 'feedH2S', 'feedH20', 'QN1', 'QN2', 'QC', 'SF', 'H2S_ppm', 'NH3_ppm'])
        writer.writerow(result)

def remove_point_from_csv(filename, point):
    """Removes a specific point from the input CSV file."""
    with open(filename, mode='r') as csv_file:
        rows = list(csv.reader(csv_file))

    # Rewrite the CSV without the completed point
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(rows[0])  # Header
        for row in rows[1:]:
            if tuple(map(float, row)) != point:
                writer.writerow(row)

def run_parallel_simulations(batch_id, input_file, lock):
    log_file_path = f'{os.path.splitext(__file__)[0]}_batch_{batch_id}.log'
    result_file = RESULT_FILE_TEMPLATE.format(batch_id)
    log_message = log_message_factory(log_file_path)
    Application = start_aspen(batch_id, log_message)
    results = []

    # Load points from the specific input file
    points = load_points_from_csv(input_file)

    for point in points:
        result = simulate(point, Application, log_message, lock, result_file, input_file)
        if result:
            results.append(result)

    if Application:
        Application.Close()
    return results

if __name__ == '__main__':
    # Required for Windows to handle multiprocessing correctly
    freeze_support()
    
    # Create a manager and lock for synchronizing file access
    manager = Manager()
    lock = manager.Lock()

    # Run simulations in parallel for each input file
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_INSTANCES) as executor:
        futures = [
            executor.submit(run_parallel_simulations, i + 1, INPUT_FILE_TEMPLATE.format(i + 1), lock)
            for i in range(NUM_INSTANCES)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure exceptions in workers are raised

    print('Simulation results saved in multiple result files.')
