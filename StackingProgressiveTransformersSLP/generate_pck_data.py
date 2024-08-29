import os
import argparse
import torch
import numpy as np
from torchtext.data import Dataset

from constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from model import build_model
from batch import Batch
from helpers import load_config, load_checkpoint, get_latest_checkpoint
from model import Model
from data import load_data, make_data_iter
from plot_videos import alter_DTW_timing, plot_video
from helpers import  load_config, log_cfg, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, get_latest_checkpoint, calculate_pck, calculate_dtw

from generate_meta_inputs import load_datasets
from torch.utils.data import DataLoader, TensorDataset

from dtw import dtw

def add_gaussian_noise(keypoints, mean=0, std=1):

    noise = np.random.normal(mean, std, keypoints.shape)
    
    mask = np.random.random(keypoints.shape) < 0.5
    noisy_keypoints = keypoints + (noise * mask)

    noisy_keypoints = keypoints + noise
    return noisy_keypoints

def generate_dataset(data, stop_after = 1000):

    inputs = []
    pcks = []

    valid_iter = make_data_iter(
            dataset=data, batch_size=1, batch_type='sentence',
            shuffle=False, train=False)
    

    stds = np.concatenate([
        np.linspace(0, 0.5, 20),
        np.linspace(0, 5, 20)
    ])

    for i, valid_batch in enumerate(iter(valid_iter)):

        if (i+1) % 100 == 0:
            print(f'Batch #{i+1}')

        if (i+1) % stop_after == 0:
            break

        target = valid_batch.trg[:, :, :-1].squeeze()
        # target = valid_batch.trg

        for std in stds:
            noisy = add_gaussian_noise(target, mean=0, std=std)
            pck = calculate_pck(noisy.numpy(), target.numpy(), alpha=0.2)

            inputs.append(torch.cat((noisy, target), dim=1))
            pcks.extend(pck)


        simulated = torch.rand(target.shape) * 1000
        pck_sim = calculate_pck(simulated.numpy(), target.numpy(), alpha=0.2)

        inputs.append(torch.cat((simulated, target), dim=1))
        pcks.extend(pck_sim)

    
    inputs = torch.cat(inputs, dim=0)
    inputs = torch.tensor(inputs, dtype=torch.float32)

    pcks = torch.tensor(pcks, dtype=torch.float32).unsqueeze(1)

    print('Generated', inputs.shape)
    print(pcks.shape)

    dataset = TensorDataset(inputs, pcks)

    return dataset

if __name__ == '__main__':
    text_cfg = load_config('Meta/text_data.yaml')
    train_data, dev_data, test_data, _, _ = load_data(cfg=text_cfg)
    
    path = os.path.abspath(__file__)
    save_dir = os.path.join(os.path.dirname(path), 'PCK_data')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_dataset = generate_dataset(train_data, stop_after=2000)
    print(f'Generated {len(train_dataset)} of train data')
    
    dev_dataset = generate_dataset(dev_data, stop_after=100)
    print(f'Generated {len(dev_dataset)} of dev data')

    test_dataset = generate_dataset(test_data, stop_after=100)
    print(f'Generated {len(test_dataset)} of test data')

    data = {'train': train_dataset, 'dev': dev_dataset, 'test': test_dataset}
    torch.save(data, f'{save_dir}/data.pt')
