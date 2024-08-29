import re
import matplotlib.pyplot as plt
import argparse

# Function to read log data from a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to parse log data
def parse_data(log_data):
    pattern = re.compile(r"Steps: (\d+) Loss: ([-\d\.]+)\| DTW: ([\d\.]+)\| PCK: ([\d\.]+) LR: ([\d\.]+)")
    steps, loss, dtw, pck, lr = [], [], [], [], []
    
    for match in re.finditer(pattern, log_data):
        steps.append(int(match.group(1)))
        loss.append(float(match.group(2)))
        dtw.append(float(match.group(3)))
        pck.append(float(match.group(4)))
        lr.append(float(match.group(5)))

    return steps, loss, dtw, pck, lr

def plot_validation(steps, loss, dtw, pck, lr):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(steps, loss, marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(steps, dtw, marker='o')
    plt.title('DTW')
    plt.xlabel('Steps')
    plt.ylabel('DTW')

    plt.subplot(2, 2, 3)
    plt.plot(steps, pck, marker='o')
    plt.title('PCK')
    plt.xlabel('Steps')
    plt.ylabel('PCK')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser("Plot Validation")
    ap.add_argument("path", type=str,
                    help="path validation file")
    
    args = ap.parse_args()

    validation_data = read_file(args.path)
    steps, loss, dtw, pck, lr = parse_data(validation_data)

    plot_validation(steps, loss, dtw, pck, lr)
