
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
def parse_training_loss(path, stop_epoch=10000):
    
    data = read_file(path)
    training_losses = re.findall(r'total training loss (\d+\.\d+)', data)
    training_losses = [float(loss)/(7096/8) for loss in training_losses] # Division for normalization, missed during training. It's simply dividing by the number times when epoch_loss was added.
    return training_losses[:stop_epoch]

def parse_validations(path, stop_step=100000):
    pattern = re.compile(r"Steps: (\d+) Loss: ([-\d\.]+)\| DTW: ([\d\.]+)\| PCK: ([\d\.]+) LR: ([\d\.]+)")
    steps, loss, dtw, pck, lr = [], [], [], [], []
    
    data = read_file(path)
    matches = re.findall(pattern, data)
    for match in matches:
        steps.append(int(match[0]))
        loss.append(float(match[1]))
        dtw.append(float(match[2]))
        pck.append(float(match[3]))
        lr.append(float(match[4]))

        if steps[-1] >= stop_step:
            break

    return steps, loss, dtw, pck, lr

def parse_meta_validations(path):
    pattern = re.compile(r"Steps (\d+) Train Loss ([\d\.]+)\| Val Loss ([\d\.]+)\| Train PCK ([\d\.]+)\| Val PCK ([\d\.]+)\| Val DTW ([\d\.]+)")
    steps, train_loss, val_loss, train_pck, val_pck, val_dtw = [], [], [], [], [], []

    data = read_file(path)
    matches = re.findall(pattern, data)
    for match in matches:
        steps.append(int(match[0]))
        train_loss.append(float(match[1]))
        val_loss.append(float(match[2]))
        train_pck.append(float(match[3]))
        val_pck.append(float(match[4]))
        val_dtw.append(float(match[5]))

    return steps, train_loss, val_loss, train_pck, val_pck, val_dtw

def create_figures():
    titles = ['Training Loss', 'Validation Loss', 'Validation DTW', 'Validation PCK']
    figures = []
    
    for title in titles:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.set_title(title)
        figures.append((fig, ax))
    
    return figures

def plot_data(figures, model_name, training_losses, validations):
    steps, val_losses, val_dtws, val_pcks, _ = validations
    data_to_plot = [training_losses, val_losses, val_dtws, val_pcks]

    # for i, ax in enumerate([figures[0][1], figures[1][1], figures[2][1], figures[3][1]]):
    for i, ax in enumerate([figures[0][1], figures[0][1], figures[2][1], figures[3][1]]):
        if i == 0:  # Training losses
            ax.plot(range(1, len(training_losses) + 1), training_losses, label=model_name)
            ax.set_xlabel('epoch')
        else:  # Validation metrics
            ax.plot(steps, data_to_plot[i], label=model_name)
            ax.set_xlabel('step')

def plot_meta_data(figures, model_name, validations):
    steps, train_losses, val_losses, train_pck, val_pcks, val_dtws = validations
    data_to_plot = [train_losses, val_losses, val_dtws, val_pcks]

    model_name = model_name.split('_')
    type = 'T2G2P' if model_name[-1] != 'T2P' else 'T2P'
    model_name = f'{model_name[0]}_{model_name[1]}_{model_name[3]}_{type}'
    for i, ax in enumerate([figures[0][1], figures[1][1], figures[2][1], figures[3][1]]):
        ax.plot(steps, data_to_plot[i], label=model_name)
        ax.set_xlabel('step')


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Plot Results")
    parser.add_argument("models_list", type=str,
                        help="List of model results to plot")
    parser.add_argument('--meta_models', type=bool, default=False)
    args = parser.parse_args()

    models_list = np.loadtxt(args.models_list, dtype=str)

    if models_list.ndim == 0:
        models_list = [models_list.item()]
    else:
        models_list = list(models_list)


    figures = create_figures()

    for model_path in models_list:
        if args.meta_models:
            validations = parse_meta_validations(f'{model_path}/validations.txt')
            plot_meta_data(figures, os.path.basename(model_path), validations)
        else:
            training_losses = parse_training_loss(f'{model_path}/train.log')
            validations = parse_validations(f'{model_path}/validations.txt')

            plot_data(figures, os.path.basename(model_path), training_losses, validations)
    

    for fig, ax in figures:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.).set_title('Model')
        ax.set_xlim(left=1)

        fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right boundary of the plot
        fig.subplots_adjust(right=0.7)  # Create space on the right side for the legend

    plt.show()


