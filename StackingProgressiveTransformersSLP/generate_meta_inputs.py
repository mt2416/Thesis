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

def process_batch(base_models, gloss_batch, text_batch):
    target = None
    file_path = None
    stacked_output = None

    for modality in ['gloss', 'text']:
        if len(base_models[modality]) == 0:
            continue

        output, current_target, current_file_path = produce_base_output(base_models[modality], locals()[f'{modality}_batch'])
        
        if stacked_output is None:
            stacked_output = output
        else:
            stacked_output = torch.cat((stacked_output, output), dim=-1)

        if target is None:
            target = current_target
            file_path = current_file_path
        elif not torch.equal(target, current_target) or file_path != current_file_path:
            raise ValueError('Modality mismatch')
    
    return stacked_output, target, file_path


def generate_data(base_models, data, ds_type, save_to=None, stop_after=10000, chunk_size = None):

    valid_gloss_iter = make_data_iter(
        dataset=data['gloss'], batch_size=1, batch_type='sentence',
        shuffle=False, train=False)
    valid_text_iter = make_data_iter(
        dataset=data['text'], batch_size=1, batch_type='sentence',
        shuffle=False, train=False)
    
    targets = []
    all_stacked_output = []
    file_paths = []
    counter = 0
    chunk_counter = 0
    
    for gloss_batch, text_batch in iter(zip(valid_gloss_iter, valid_text_iter)):
        
        if (counter+1) % 100 == 0:
            print(f'processed {counter+1} samples')

        if (counter+1) % stop_after == 0:
            break
        
        try:
            stacked_output, target, file_path = process_batch(base_models, gloss_batch=gloss_batch, text_batch=text_batch)
        except Exception as e:
            print(f'error batch #{counter}: {e}')
            continue

        all_stacked_output.append(stacked_output)
        targets.append(target)
        file_paths.append(file_path)
        counter += 1

        if chunk_size is not None and (counter+1) % chunk_size == 0:
            save_chunk(all_stacked_output, targets, file_paths, chunk_counter, ds_type, save_to)
            chunk_counter += 1
            all_stacked_output, targets, file_paths = [], [], []
    
    if chunk_size is not None and all_stacked_output:
        save_chunk(all_stacked_output, targets, file_paths, chunk_counter, ds_type, save_to)
    
    print('Total size', counter)
    print(f'{chunk_counter+1} chunks saved')

    if not chunk_size:
        frames = [tar.shape[1] for tar in targets]
        return torch.cat(all_stacked_output, dim=1), torch.cat(targets, dim=1), frames, file_paths
    
def save_meta_inputs(meta_data, type, save_to=None):
    path = os.path.abspath(__file__)

    if not save_to:
        save_to = 'meta_data'

    save_dir = os.path.join(os.path.dirname(path), save_to)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    torch.save(meta_data, f'{save_dir}/{type}_meta_data.pt')

def save_chunk(stacked_output, targets, file_paths, chunk_counter, dataset_type, save_to=None):
    frames = [tar.shape[1] for tar in targets]
    stacked_output = torch.cat(stacked_output, dim=1)
    targets = torch.cat(targets, dim=1)

    if len(frames) != len(file_paths):
        raise Exception('frames != filepaths')

    meta_data = {
        'stacked_pred': stacked_output,
        'targets': targets,
        'frames': frames,
        'file_paths': file_paths
    }
    save_meta_inputs(meta_data, type=f'chunk_{chunk_counter}_{dataset_type}', save_to=save_to)

    print(f'Saved chunk {chunk_counter} - {stacked_output.shape}')

def produce_base_output(base_models, valid_batch):

    outputs = []
    targets = []
    file_paths = []
    for model in base_models:

        batch = Batch(torch_batch=valid_batch,
                                pad_index = model.src_vocab.stoi[PAD_TOKEN],
                            model = model)
        target = batch.trg

        skel_out, _ = model.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask)
        
        if model.future_prediction != 0:
            target = target[:, :, :target.shape[2] // (model.future_prediction)]
            skel_out = skel_out[:, :, :skel_out.shape[2] // model.future_prediction]

        outputs.append(skel_out)
        targets.append(target)
        file_paths.append(batch.file_paths)

    stacked_output = outputs[0] 
    for m in range(1, len(base_models)):
        stacked_output = torch.cat((stacked_output, outputs[m]), dim=2)

    return stacked_output, targets[0], file_paths[0][0]

def load_base_models(cfgs, src_vocabs: dict, trg_vocab):
    base_models = {'text': [], 'gloss': []}
    for cfg in cfgs:

        is_gloss_model = cfg["data"]["src"] == "gloss"

        # Load the model directory and checkpoint
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir,post_fix="_best")
        print(f'ckpt {ckpt} ')

        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))
        
        # Load model state from disk
        model_checkpoint = load_checkpoint(ckpt, use_cuda=True)

        # Set gloss or text vocab
        if is_gloss_model:
            src_vocab = src_vocabs['gloss']

        # Build model and load parameters into it
        model = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
        model.load_state_dict(model_checkpoint["model_state"])
        model.cuda()
        model.eval()

        if is_gloss_model:
            base_models['gloss'].append(model)
        else:
            base_models['text'].append(model)
    
    return base_models 

def load_datasets(text_data_path='Meta/text_data.yaml',
                   gloss_data_path='Meta/gloss_data.yaml'):
    
    text_cfg = load_config(text_data_path)
    gloss_cfg = load_config(gloss_data_path)

    train_text_data, dev_text_data, test_text_data, src_text_vocab, trg_vocab = load_data(cfg=text_cfg)
    train_gloss_data, dev_gloss_data, test_gloss_data, src_gloss_vocab, trg_vocab = load_data(cfg=gloss_cfg)

    train_data = {'text': train_text_data, 'gloss': train_gloss_data}
    dev_data = {'text': dev_text_data, 'gloss': dev_gloss_data}
    test_data = {'text': test_text_data, 'gloss': test_gloss_data}
    src_vocabs = {'text': src_text_vocab, 'gloss': src_gloss_vocab}

    return train_data, dev_data, test_data, src_vocabs, trg_vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Meta input data generator")

    parser.add_argument("configs", type=str,
                        help="File for the model's configurations.")
    parser.add_argument("--save_to", type=str,
                        help='Save dir')
    
    args = parser.parse_args()

    configs_path = np.loadtxt(args.configs, dtype=str)

    cfgs = []
    # Load configs
    for yaml in configs_path:
        cfgs.append(load_config(yaml))
    
    train_data, dev_data, test_data, src_vocabs, trg_vocab = load_datasets(text_data_path='Meta/text_data.yaml',
                                                    gloss_data_path='Meta/gloss_data.yaml')
    
    is_mismatch = len(train_data['text']) != len(train_data['gloss']) or \
                         len(dev_data['text'])   != len(dev_data['gloss'])   or \
                         len(test_data['text'])  != len(test_data['gloss'])
    
    if is_mismatch:
        raise Exception('gloss, text dataset sizes dont match')
    
    base_models = load_base_models(cfgs, src_vocabs=src_vocabs, trg_vocab=trg_vocab)

    save_to=None
    if args.save_to:
        save_to = args.save_to

    print(base_models)
    print('Generating train data...')
    generate_data(base_models, train_data, ds_type='train', chunk_size=1000, save_to=save_to)

    print('Generating dev data...')
    generate_data(base_models, dev_data, ds_type='dev', chunk_size=1000, save_to=save_to)

    print('Generating test data...')
    generate_data(base_models, test_data, ds_type='test', chunk_size=1000, save_to=save_to)


