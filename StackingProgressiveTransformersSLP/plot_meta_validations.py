import re
import matplotlib.pyplot as plt
import argparse

# Function to read log data from a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to parse log data
def parse_data(log_data):
    pattern = re.compile(r"Steps (\d+) Train Loss ([\d\.]+)\| Val Loss ([\d\.]+)\| Train PCK ([\d\.]+)\| Val PCK ([\d\.]+)\| Val DTW ([\d\.]+)")
    steps, train_loss, val_loss, train_pck, val_pck, val_dtw = [], [], [], [], [], []
    
    for match in re.finditer(pattern, log_data):
        steps.append(int(match.group(1)))
        train_loss.append(float(match.group(2)))
        val_loss.append(float(match.group(3)))
        train_pck.append(float(match.group(4)))
        val_pck.append(float(match.group(5)))
        val_dtw.append(float(match.group(6)))

    return steps, train_loss, val_loss, train_pck, val_pck, val_dtw

def plot_validation(steps, train_loss, val_loss, train_pck, val_pck, val_dtw):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(steps, train_loss, marker='o')
    plt.plot(steps, val_loss, marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(steps, train_pck, marker='o')
    plt.plot(steps, val_pck, marker='o')
    plt.title('PCK')
    plt.xlabel('Steps')
    plt.ylabel('PCK')

    # plt.subplot(2, 2, 3)
    # plt.plot(steps, , marker='o')
    # plt.title('PCK')
    # plt.xlabel('Steps')
    # plt.ylabel('PCK')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser("Plot Validation")
    ap.add_argument("path", type=str,
                    help="path validation file")
    
    args = ap.parse_args()

    validation_data = read_file(args.path)
    steps, train_loss, val_loss, train_pck, val_pck, val_dtw = parse_data(validation_data)

    plot_validation(steps, train_loss, val_loss, train_pck, val_pck, val_dtw)