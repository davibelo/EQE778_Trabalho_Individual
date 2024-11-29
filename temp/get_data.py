import os
import csv
import time
import numpy as np
import win32com.client as win32
import concurrent.futures

RESULT_FILE = 'simulation_results.csv'
SIMULATION_FILE = r'UTAA_run\UTAA_revK.bkp'
TEST = 0
RETRY_LIMIT = 1  # Maximum number of retries for each simulation
RETRY_DELAY = 5  # Delay between retries in seconds

if TEST == 1:
    # Define ranges for testing purposes
    ranges = {
        'feedNH3': [0.001, 0.05, 0.05],
        'feedH2S': [0.001, 0.05, 0.05],
        'QN1':     [500000, 600000, 100000],
        'QN2':     [800000, 909000, 100000],
        'SF':      [0.5, 0.6, 0.1]
    }
else:
    # Define ranges for production
    ranges = {
        'feedNH3': [0.0000001, 0.10, 0.005],
        'feedH2S': [0.0000001, 0.10, 0.005],
        'QN1':     [450000,  600000, 10000],
        'QN2':     [700000, 1200000, 10000],
        'SF':      [0.0000001, 1, 0.05]
    }

aspen_Path = os.path.abspath(SIMULATION_FILE)

def start_aspen(log_message):
    """Starts Aspen and returns the Application object, with retries."""
    for attempt in range(RETRY_LIMIT):
        try:
            Application = win32.Dispatch('Apwn.Document')
            Application.InitFromArchive2(aspen_Path)
            Application.visible = 0
            log_message(f"Aspen started successfully on attempt {attempt + 1}.")
            return Application
        except Exception as e:
            log_message(f"Failed to start Aspen on attempt {attempt + 1}: {e}")
            time.sleep(RETRY_DELAY)
    return None

def simulate(x, Application, log_message):
    feedNH3, feedH2S, feedH20, QN1, QN2, QC, SF = x
    for attempt in range(RETRY_LIMIT):
        if not Application:
            Application = start_aspen(log_message)
            if not Application:
                log_message(f"Failed to initialize Aspen after {RETRY_LIMIT} attempts for inputs {x}. Skipping simulation.")
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
            message = f'Simulation with feedNH3: {feedNH3}, feedH2S: {feedH2S}, feedH20: {feedH20}, QN1: {QN1}, QN2: {QN2}, QC: {QC}, SF: {SF} -> H2S: {cH2S_ppm}, NH3: {cNH3_ppm}'
            log_message(message)
            return x + y

        except Exception as e:
            log_message(f"Error simulating {x} on attempt {attempt + 1}: {e}")
            Application = None  # Set Application to None to trigger restart in next attempt
            time.sleep(RETRY_DELAY)
    return None

def log_message_factory(log_file_path):
    def log_message(message):
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + '\n')
            print(message)
    return log_message

def generate_points(ranges):
    feedNH3_range = np.arange(ranges['feedNH3'][0], ranges['feedNH3'][1] + ranges['feedNH3'][2], ranges['feedNH3'][2])
    feedH2S_range = np.arange(ranges['feedH2S'][0], ranges['feedH2S'][1] + ranges['feedH2S'][2], ranges['feedH2S'][2])
    QN1_range = np.arange(ranges['QN1'][0], ranges['QN1'][1] + ranges['QN1'][2], ranges['QN1'][2])
    QN2_range = np.arange(ranges['QN2'][0], ranges['QN2'][1] + ranges['QN2'][2], ranges['QN2'][2])
    SF_range = np.arange(ranges['SF'][0], ranges['SF'][1] + ranges['SF'][2], ranges['SF'][2])

    feedNH3, feedH2S, QN1, QN2, SF = np.meshgrid(feedNH3_range, feedH2S_range, QN1_range, QN2_range, SF_range, indexing='ij')
    feedNH3, feedH2S, QN1, QN2, SF = map(np.ravel, (feedNH3, feedH2S, QN1, QN2, SF))

    points = []
    for nh3, h2s, qn1, qn2, sf in zip(feedNH3, feedH2S, QN1, QN2, SF):
        feedH20 = 1 - nh3 - h2s
        if feedH20 >= 0:
            points.append((float(nh3), float(h2s), float(feedH20), float(qn1), float(qn2), 3.0, float(sf)))
    return points

def run_parallel_simulations(batch_id, points):
    log_file_path = f'{os.path.splitext(__file__)[0]}_batch_{batch_id}.log'
    log_message = log_message_factory(log_file_path)
    Application = start_aspen(log_message)
    results = []

    for point in points:
        result = simulate(point, Application, log_message)
        if result:
            results.append(result)

    if Application:
        Application.Close()
    return results

if __name__ == '__main__':
    # Generate input points
    input_points = generate_points(ranges)
    print(f'Number of points to simulate: {len(input_points)}')
    results = []

    # Divide input points into 4 batches for parallel processing
    num_instances = 4
    batch_size = len(input_points) // num_instances
    batches = [input_points[i:i + batch_size] for i in range(0, len(input_points), batch_size)]

    # Run simulations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_instances) as executor:
        for batch_id, local_results in enumerate(executor.map(run_parallel_simulations, range(num_instances), batches)):
            results.extend(local_results)

    # Save results to CSV file
    csv_file_path = os.path.join(os.getcwd(), RESULT_FILE)    
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['feedNH3', 'feedH2S', 'feedH20', 'QN1', 'QN2', 'QC', 'SF', 'H2S_ppm', 'NH3_ppm'])
        for result in results:
            writer.writerow(result)

    print(f'Simulation results saved to {csv_file_path}')