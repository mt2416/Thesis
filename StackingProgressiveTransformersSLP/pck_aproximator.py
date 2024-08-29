import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


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
from generate_pck_data import generate_dataset
import random


class PCKAproximator(nn.Module):
    def __init__(self):
        super(PCKAproximator, self).__init__()

        # self.input_size = 150 # 150 keypoints (without counter)

        # self.pred_fc1 = nn.Linear(150, 256)
        # # self.bn1 = nn.BatchNorm1d(256)
        # self.pred_fc2 = nn.Linear(256, 256)

        # self.gt_fc1 = nn.Linear(150, 256)
        # # self.bn2 = nn.BatchNorm1d(256)
        # self.gt_fc2 = nn.Linear(256, 256)

        # self.fc1 = nn.Linear(256*2, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 1)

        # self.sigmoid = nn.Sigmoid()

        # self.dropout = nn.Dropout(p=0.5)


        self.input_size = 150 # 150 keypoints (without counter)

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(2 * self.input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.5)

    
    def forward(self, pred_skel, gt_skel):

        x = torch.stack((pred_skel, gt_skel), dim=1)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))  
        
        x = x.view(x.size(0), -1) # Flatten

        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        
        out = self.sigmoid(self.fc3(x))

        return out
        
        
class PCKTrainer:
    def __init__(self, approximator, model_dir, datasets=None):
        super(PCKTrainer, self).__init__()

        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.approximator = approximator
        self.optimizer = optim.Adam(self.approximator.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.approximator.to(self.device)

        self.datasets = datasets
        if datasets != None:
            self.train_loader = DataLoader(datasets['train'], batch_size=128, shuffle=True)
            self.val_loader = DataLoader(datasets['test'], batch_size=128, shuffle=False) #!!!
            self.test_loader = DataLoader(datasets['test'], batch_size=128, shuffle=False) 

    def train(self, epochs, validate_every=1000):
        if self.datasets is None:
            raise Exception('datasets are not specified')

        self.approximator.train()

        steps = 0
        for epoch in range(epochs):
            train_loss = 0.0 
            for input_batch, target_batch in self.train_loader:
                input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.approximator(input_batch[:, :150], input_batch[:, 150:])
                loss = self.criterion(outputs, target_batch)
                loss.backward()
                self.optimizer.step()

                steps += 1

                train_loss += loss.item()

                if(steps+1) % validate_every == 0:
                    val_loss = self.validate(type='val')
                    msg = f'Steps {steps+1} Train Loss {train_loss/validate_every:.6f}| Val Loss {val_loss:.6f}|'
                    print(f'Epoch [{epoch+1}/{epochs}] {msg}')

                    train_loss = 0.0
                    self.approximator.train()
                if (steps+1) % 5000 == 0:
                    save_dir = f'{self.model_dir}/model_{steps+1}.pth'
                    torch.save(self.approximator, save_dir)
                    print(f'model saved {save_dir}')

            print('Test')
            print('TEST loss', self.validate(type='test'))

    def train_buffer_(self, data_buffer, n_batches=10):
        self.approximator.train()
        for param in self.approximator.parameters():
            param.requires_grad = True 

        for _ in range(n_batches):
            batch = random.sample(data_buffer, 3)
            inputs = torch.cat([item[0] for item in batch], dim=0).to(self.device)
            gt_pcks = torch.cat([item[1] for item in batch], dim=0).to(self.device)
            


            self.optimizer.zero_grad()
            outputs = self.approximator(inputs[:, :150], inputs[:, 150:])
            loss = self.criterion(outputs, gt_pcks)
            loss.backward()
            self.optimizer.step()

    def validate(self, type='val'):
        self.approximator.eval()
        val_loss = 0.0

        data_loader = self.val_loader
        if type == 'test':
            data_loader = self.test_loader

        i=0
        with torch.no_grad():
            for input_batch, target_batch in data_loader:
                input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
                outputs = self.approximator(input_batch[:, :150], input_batch[:, 150:])
                loss = self.criterion(outputs, target_batch)

                val_loss += loss.item()
        val_loss /= len(self.val_loader)
        return val_loss

    def test(self):
        return self.validate(type='test')
    

class NegPCKLoss(nn.Module):
    def __init__(self, fine_tuning=False):
        super(NegPCKLoss, self).__init__()

        self.approximator = torch.load("C:\\dis\\ProgressiveTransformersSLP\\PCK_data\\model_10000.pth")
        # self.approximator = PCKAproximator()
        self.approximator.eval()

        # self.fine_tuning = fine_tuning
        self.fine_tuning = False

        for param in self.approximator.parameters():
            param.requires_grad = False

        if self.fine_tuning:
            self.trainer = PCKTrainer(self.approximator, model_dir="C:\\dis\\ProgressiveTransformersSLP\\PCK_data")
            self.data_buffer = []
            self.buffer_size = 4000
            self.train_interval = 3000
            self.call_out = 0

        self.mse = nn.MSELoss()

        self.smoothness = 10
        self.threshold = 0.9
        self.alpha = 0.2
        self.grads = []

    def forward(self, pred_skel, gt_skel):
        batch_size, seq_len, keypoints_num = pred_skel.shape

        # if self.fine_tuning and pred_skel.requires_grad:
        #     # Store new data for later training
        #     batch_pck = []
        #     for b in range(batch_size):
        #         batch_input, batch_target = pred_skel[b].clone().detach(), gt_skel[b].clone().detach()

        #         if keypoints_num > 151:
        #             batch_input, batch_target = batch_input[:, :151], batch_target[:, :151]

        #         # remove counter 
        #         batch_input, batch_target = batch_input[:, :-1], batch_target[:, :-1]

        #         pck = calculate_pck(batch_input.cpu().numpy(), batch_target.cpu().numpy())

        #         data_entry = ((torch.cat((batch_input, batch_target), dim=-1)),
        #                     torch.tensor(pck, dtype=torch.float32).unsqueeze(1))

        #         batch_pck.extend(pck)
                
        #         self.data_buffer.append(data_entry)

        #         if len(self.data_buffer) >= self.buffer_size:
        #             self.data_buffer.pop(0)

        #         self.call_out += 1
        #          # Periodically train the approximator model
        #         if self.call_out % 250 == 0:
        #             print('Batch PCK mean', torch.tensor(batch_pck).mean())
        #         if self.call_out % self.train_interval == 0 and torch.tensor(batch_pck).mean() < 0.5:
        #         # if self.call_out % self.train_interval == 0:
        #             print('fine tuning')
        #             self.trainer.train_buffer_(self.data_buffer, n_batches=128)
                    
        #             self.approximator.eval()
        #             for param in self.approximator.parameters():
        #                 param.requires_grad = False


        if keypoints_num > 151:

            reshaped_pred_skel = pred_skel[:, :, :-1].view(batch_size, seq_len, keypoints_num//150, 150).permute(2, 0, 1, 3)
            reshaped_gt_skel = gt_skel[:, :, :-1].view(batch_size, seq_len, keypoints_num//150, 150).permute(2, 0, 1, 3)


            output_accum = []
            for i in range(keypoints_num//150):
                out = self.approximator(reshaped_pred_skel[i].view(batch_size*seq_len, -1), reshaped_gt_skel[i].view(batch_size*seq_len, -1))
                output_accum.append(out)
            
            outputs = torch.mean(torch.stack(output_accum))

        else:
            outputs = self.approximator(pred_skel[:, :, :-1].view(batch_size*seq_len, -1), gt_skel[:, :, :-1].view(batch_size*seq_len, -1))


        loss = -torch.log(torch.mean(outputs) + 1e-8)
        return loss
    
    def save_grad(self, grad):
        self.grads.append(grad)

    def soft_forward(self, pred_skel, gt_skel):

        distance = torch.norm(gt_skel - pred_skel, dim=-1)

        correct = torch.sigmoid(distance)

        loss = 1-correct.mean()
        return loss
    
if __name__ == '__main__':

    path = os.path.abspath(__file__)
    model_dir = os.path.join(os.path.dirname(path), 'PCK_data')

    model = PCKAproximator()
    datasets = torch.load(f'{model_dir}/data.pt')
    
    trainer = PCKTrainer(model, model_dir=model_dir, datasets=datasets)
    print('Val', trainer.validate())
    trainer.train(epochs=1)

    model.to(torch.device('cpu'))
    test = torch.load("PCK_data\\test.pt", map_location='cpu')

    batch_size, seq_len, _ =  test['target'].shape

    print(test['input'].shape)

    out0 = model(test['input'][:, :, :-1].view(batch_size*seq_len, -1), test['target'][:, :, :-1].view(batch_size*seq_len, -1))

    input = test['input'].squeeze().detach()[ :, :-1]
    target = test['target'].squeeze().detach()[ :, :-1]

    pck = calculate_pck(input.numpy(), target.numpy())
    print('PCK', pck)
    pck = torch.tensor(pck)

    out = model(input, target)
    print('OUT', out.view(-1, input.shape[0]))

    print('Mean', torch.mean(pck).item(), ' - ', torch.mean(out).item())

    # # plot_video(joints=test['input'][:, :-1].squeeze().cpu().detach().numpy(),
    # plot_video(joints=input.numpy(),
    #                 file_path="C:\\dis\\ProgressiveTransformersSLP\\PCK_data",
    #                 video_name=f'2.mp4',
    #                 references=target.numpy(),
    #                 skip_frames=2,
    #                 sequence_ID='')

