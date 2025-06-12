# split_csv.py

import sys
import csv
import random

def split_csv(input_file, test_percentage):
    with open(input_file, 'r', newline='') as f:
        rows = list(csv.reader(f))

    total = len(rows)
    test_size = int(total * test_percentage / 100)
    
    random.shuffle(rows)
    test_rows = rows[:test_size]
    train_rows = rows[test_size:]

    with open('test.csv', 'w', newline='') as test_f:
        writer = csv.writer(test_f)
        writer.writerows(test_rows)

    with open('train.csv', 'w', newline='') as train_f:
        writer = csv.writer(train_f)
        writer.writerows(train_rows)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_csv.py name.csv test_percentage")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        percentage = float(sys.argv[2])
        if not 0 <= percentage <= 100:
            raise ValueError
    except ValueError:
        print("Error: test_percentage must be a number between 0 and 100.")
        sys.exit(1)

    split_csv(filename, percentage)
