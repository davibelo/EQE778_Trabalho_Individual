import os
import csv
import numpy as np

POINTS_FILE_TEMPLATE = 'simulation_points_part_{}_initial.csv'
NUM_PARTS = 4
TEST = 0

def generate_points(ranges):
    def generate_range(start, stop, step):
        num_points = int((stop - start) / step) + 1
        return np.linspace(start, stop, num=num_points, endpoint=(stop - start) % step == 0)

    feedNH3_range = generate_range(ranges['feedNH3'][0], ranges['feedNH3'][1], ranges['feedNH3'][2])
    feedH2S_range = generate_range(ranges['feedH2S'][0], ranges['feedH2S'][1], ranges['feedH2S'][2])
    QN1_range = generate_range(ranges['QN1'][0], ranges['QN1'][1], ranges['QN1'][2])
    QN2_range = generate_range(ranges['QN2'][0], ranges['QN2'][1], ranges['QN2'][2])
    SF_range = generate_range(ranges['SF'][0], ranges['SF'][1], ranges['SF'][2])

    feedNH3, feedH2S, QN1, QN2, SF = np.meshgrid(feedNH3_range, feedH2S_range, QN1_range, QN2_range, SF_range, indexing='ij')
    feedNH3, feedH2S, QN1, QN2, SF = map(np.ravel, (feedNH3, feedH2S, QN1, QN2, SF))

    points = []
    for nh3, h2s, qn1, qn2, sf in zip(feedNH3, feedH2S, QN1, QN2, SF):
        feedH20 = 1 - nh3 - h2s
        if feedH20 >= 0:
            points.append((float(nh3), float(h2s), float(feedH20), float(qn1), float(qn2), 3.0, float(sf)))
    return points

def save_points_to_csv(points, filename):
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['feedNH3', 'feedH2S', 'feedH20', 'QN1', 'QN2', 'QC', 'SF'])
        for point in points:
            writer.writerow(point)

def save_points_in_parts(points, num_parts):
    total_points = len(points)
    points_per_part = total_points // num_parts
    print(f'Total number of points: {total_points}')
    print(f'Number of points per file: {points_per_part}')

    for i in range(num_parts):
        start_index = i * points_per_part
        end_index = (i + 1) * points_per_part if i < num_parts - 1 else len(points)
        part_points = points[start_index:end_index]
        filename = POINTS_FILE_TEMPLATE.format(i + 1)
        save_points_to_csv(part_points, filename)
        print(f'Part {i + 1} saved to {filename}')

if __name__ == '__main__':
    # Define ranges
    if TEST == 1:
        # Testing ranges
        ranges = {
            'feedNH3': [0.001, 0.01, 0.001],
            'feedH2S': [0.001, 0.01, 0.001],
            'QN1':     [500000, 600000, 100000],
            'QN2':     [800000, 900000, 100000],
            'SF':      [0.5, 0.6, 0.1]
        }
    else:
        # Production ranges
        ranges = {
            'feedNH3': [0.001, 0.01, 0.001],
            'feedH2S': [0.001, 0.01, 0.001],
            'QN1':     [450000,  600000, 10000],
            'QN2':     [700000, 1200000, 10000],
            'SF':      [0.0, 1, 0.1]
        }

    # Generate and save points
    input_points = generate_points(ranges)
    save_points_in_parts(input_points, num_parts=NUM_PARTS)
    print(f'Grid of points divided into {NUM_PARTS} files for multithreading.')
