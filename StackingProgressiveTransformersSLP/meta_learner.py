import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset


from model import build_model
from batch import Batch
from helpers import  load_config, log_cfg, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, get_latest_checkpoint, calculate_pck, calculate_dtw
from model import Model
from prediction import validate_on_data
from data import load_data, make_data_iter
from constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from torch.utils.data import DataLoader, TensorDataset, Subset
from plot_videos import alter_DTW_timing, plot_video
from dtw import dtw
import re

class MetaDataIterator:
    def __init__(self, stacked_tensor, target_tensor, frames, file_paths):

        self.stacked_tensor = stacked_tensor.squeeze(0)
        self.target_tensor = target_tensor.squeeze(0)   
        self.frames = frames
        self.file_paths = file_paths
        self.start_indices = [sum(frames[:i]) for i in range(len(frames))]
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.frames):
            raise StopIteration
        
        start_idx = self.start_indices[self.current_index]
        end_idx = start_idx + self.frames[self.current_index]

        sentence = self.stacked_tensor[start_idx:end_idx]
        target = self.target_tensor[start_idx:end_idx]
        file_path = self.file_paths[self.current_index]
        
        self.current_index += 1
        return sentence, target, file_path

class ChunkIterator:
    def __init__(self, chunk_dir, type, chunk_size=1000):
        self.chunk_dir = chunk_dir
        self.chunk_size = chunk_size 
        self.chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(f'{type}_meta_data.pt')])
        
        self.chunk_counter = 0
        self.total_counter = 0
        self.current_meta_iterator = None

    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.current_meta_iterator is None or self.current_meta_iterator.current_index >= len(self.current_meta_iterator.frames):
            if self.chunk_counter >= len(self.chunk_files):
                raise StopIteration
            
            chunk_file = self.chunk_files[self.chunk_counter]
            chunk_data = torch.load(f'{self.chunk_dir}/{chunk_file}')

            self.current_meta_iterator = MetaDataIterator(
                chunk_data['stacked_pred'], chunk_data['targets'], chunk_data['frames'], chunk_data['file_paths']
            )

            self.chunk_counter += 1 
        
        self.total_counter += 1
        return next(self.current_meta_iterator)
        

class MetaModel(nn.Module):
    def __init__(self, config):
        super(MetaModel, self).__init__()

        layers = []
        for layer_config in config['model']['layers']:
            if layer_config['type'] == 'Linear':
                layers.append(nn.Linear(layer_config['in_features'], layer_config['out_features']))
                if 'activation' in layer_config:
                    if layer_config['activation'] == 'ReLU':
                        layers.append(nn.ReLU())
                if 'dropout' in layer_config:
                    layers.append(nn.Dropout(layer_config['dropout']))
        
        self.network = nn.Sequential(*layers)
        self.name = config['model']['dir'].split('/')[-1]


    def forward(self, x):
        return self.network(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        # print(f'Model saved to {path}')

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f'Model loaded from {path}')
    
    def evaluate(self, iterator: ChunkIterator, device, verbose=1,  dataset_name=None, save_to=None):
        # Euclidean norm is the cost function, difference of coordinates
        euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

        self.eval()
        predictions = []
        targets = []
        loss = nn.MSELoss()
        dtws = []
        total_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for sentence, target, file_path in iterator:
                final_sentence = torch.tensor([]).to(device)
                for frame in sentence:
                    out = self.forward(frame)
                    final_sentence = torch.cat((final_sentence, out.unsqueeze(0)), dim=0)
                predictions.append(final_sentence)

                total_loss += loss(final_sentence, target)
                targets.append(target)
        
                d, cost_matrix, acc_cost_matrix, path = dtw(target.cpu().numpy(), final_sentence.cpu().numpy(), dist=euclidean_norm)

                # Normalise the dtw cost by sequence length
                d = d/acc_cost_matrix.shape[0]

                dtws.append(d)

        
        pcks = []
        for b in range(len(predictions)):
            pck = calculate_pck(predictions[b].cpu().numpy(), targets[b].cpu().numpy())
            pcks.append(np.mean(pck))

        pck_val, dtw_val, loss = np.mean(pcks), np.mean(dtws), total_loss/len(targets)

        report = f'Loss {loss.item()}| PCK {pck_val}| DTW {dtw_val}'
        if verbose: 
            print(report)
        if save_to:
            with open(f'{save_to}/{dataset_name}_result.txt', 'w') as f:
                f.write(report + '\n')

        return predictions, targets, (pck_val, dtw_val, loss)
                
        
class MetaModelTrainer:
    def __init__(self, data_dir, meta_model, optimizer, criterion, model_dir, device, batch_size = 32, epochs = 100):
        self.data_dir = data_dir

        self.meta_model = meta_model 
        self.optimizer = optimizer 
        self.criterion = criterion
        self.batch_size = batch_size 
        self.epochs = epochs

        self.device = device
        self.meta_model.to(self.device)
        print('device', self.device)

        self.model_dir = model_dir

    def validate(self):
        self.meta_model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        pcks = []
        with torch.no_grad():
            dev_dataloader = ChunkIterator(data_dir, type='dev')
            for batch_idx, (meta_inputs, targets, file_path) in enumerate(dev_dataloader):
                meta_inputs, targets = meta_inputs.to(self.device), targets.to(self.device)
                meta_outputs = self.meta_model(meta_inputs)
                loss = self.criterion(meta_outputs, targets)

                pck = np.mean(calculate_pck(meta_outputs.detach().cpu().numpy(), targets.detach().cpu().numpy()))
                pcks.append(pck)

                val_loss += loss.item()
                
        avg_val_loss = val_loss / batch_idx
        pck_score = np.mean(pcks)
        return avg_val_loss, pck_score

    def train(self):
        print('device', self.device)
        with open(f'{model_dir}/validations.txt', 'w') as log:
            steps = 0
            validate_each = 100
            best_pck = 0.0
            for epoch in range(self.epochs):
                self.meta_model.train()

                epoch_loss = 0.0
                pcks = []
                print(f'Epoch {epoch}')
                train_dataloader = ChunkIterator(self.data_dir, type='train')
                for meta_inputs, targets, file_path in train_dataloader:
                    meta_inputs, targets = meta_inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    meta_outputs = self.meta_model(meta_inputs)
                    loss = self.criterion(meta_outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    steps += 1

                    pck = np.mean(calculate_pck(meta_outputs.detach().cpu().numpy(), targets.detach().cpu().numpy()))
                    pcks.append(pck)

                    epoch_loss += loss.item()

                    if (steps+1) % validate_each == 0:

                        train_loss = epoch_loss / validate_each
                        train_pck = np.mean(pcks)

                        # val_loss, val_pck = self.validate()
                        validation = self.meta_model.evaluate(ChunkIterator(data_dir, type='dev'), device=self.device, verbose=0)
                        self.meta_model.train()

                        _, _, (val_pck, val_dtw, val_loss) = validation

                        if val_pck > best_pck:
                            # print('New best PCK!')
                            meta_model.save(f'{self.model_dir}/best.pth')
                            best_pck = val_pck

                        log_msg = f'Steps {steps+1} Train Loss {train_loss:.4f}| Val Loss {val_loss:.4f}| Train PCK {train_pck:.4f}| Val PCK {val_pck:.4f}| Val DTW {val_dtw:.4f}'
                        print(log_msg)

                        log.write(f'{log_msg}\n')
                        log.flush()  # Ensure the data is written to the file immediately

                        pcks = []
                        epoch_loss = 0.0

                    
        meta_model.save(f'{self.model_dir}/step_{steps}.pth')

        # Validate
        datasets_to_run = {'test', 'dev'}
        for set_name in datasets_to_run:
            self.meta_model.evaluate(ChunkIterator(self.data_dir, type=set_name), device=self.device, verbose=0, save_to=self.model_dir, dataset_name=set_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Meta Model")

    parser.add_argument("mode", choices=["train", "test"],
                    help="train a model or test")

    parser.add_argument('cfg', type=str,
                        help='Config file (yaml)')
    
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    
    for l in cfg['model']['layers']:
        print(l)

    data_dir = cfg['data']['dir']
    
    meta_model = MetaModel(config=cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta_model.to(device)

    model_dir = cfg['model']['dir']
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if args.mode == 'test':
        meta_model.load(f'{model_dir}/best.pth')

    
    trainer = MetaModelTrainer(data_dir=data_dir,
                                meta_model=meta_model,
                                optimizer=optim.Adam(meta_model.parameters(), lr=0.001),
                                criterion = nn.MSELoss(),
                                model_dir=model_dir,
                                device = device,
                                batch_size=32,
                                epochs=1)
    
    if args.mode == 'test':
        print('test dataset')
        output_joints, references, _ = meta_model.evaluate(ChunkIterator(data_dir, type='test'), device=device, save_to=model_dir, dataset_name='test', verbose=1)
        print('dev dataset')
        meta_model.evaluate(ChunkIterator(data_dir, type='dev'), device=device, save_to=model_dir, dataset_name='dev', verbose=1)

        videos_dir = os.path.join(model_dir, 'test_videos')
        if not os.path.exists(videos_dir):
            os.mkdir(videos_dir)

        display = 20

        file_paths = []
        for ( _, _, file_path) in ChunkIterator(data_dir, type='test'):
            file_paths.append(file_path)
            if len(file_path) == display:
                break

        for i in range(display):
            seq = output_joints[i]
            ref_seq = references[i]
            file_path = file_paths[i]

            print(f'{file_path}', seq.shape)

            #add false counter to be compatible with plot_video()
            seq = torch.cat((seq.cpu(), torch.zeros(seq.size(0), 1)), dim=1)
            ref_seq = torch.cat((ref_seq.cpu(), torch.zeros(ref_seq.size(0), 1)), dim=1)

            plot_video(joints=seq.cpu().numpy(),
                            file_path=videos_dir,
                            video_name=f'{i}.mp4',
                            references=ref_seq.cpu().numpy(),
                            skip_frames=2,
                            sequence_ID=file_path)
        
        print(f'Saved {display} videos')
    else:
        trainer.train()


