import torch 
import torch.nn.functional as F
import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from network import UNet
from datasets import Noise2VoidDataset


# 掩码条件下的l2 loss

def pixel_mse_loss(predictions, targets, mask):
    return F.mse_loss(predictions * mask, targets * mask)
  

   
def _load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# patch_size 转换为二元组
def _parse_patch_size(value):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError('patch_size must be int or a list/tuple of length 2')



def train():
    parser = argparse.ArgumentParser(description='Noise2Void training')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    cfg = _load_config(args.config)

    in_channels = cfg.get('in_channels', 1)
    out_channels = cfg.get('out_channels', 1)
    depth = cfg.get('depth', 5)
    padding = cfg.get('padding', True)
    batch_norm = cfg.get('batch_norm', True)
    up_mode = cfg.get('up_mode', 'upconv')
    interpolation_mode = cfg.get('interpolation_mode', 'nearest')

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        padding=padding,
        batch_norm=batch_norm,
        up_mode=up_mode,
        interpolation_mode=interpolation_mode,
    )

    if torch.cuda.is_available() and cfg.get('device', '').startswith('cuda'):
        device = torch.device(cfg.get('device', 'cuda:0'))
    else:
        device = torch.device('cpu')
    model.to(device)

    # summary(model, (1, 100, 100))    

    epochs = cfg.get('epochs', 500)
    batch_size = cfg.get('batch_size', 16)
    patch_size = _parse_patch_size(cfg.get('patch_size', 64))
    dataset_path = cfg.get('dataset_path')
    if not dataset_path:
        raise ValueError('dataset_path must be set in config.yaml')

    lr = cfg.get('lr', 1e-3)
    scheduler_tmax = cfg.get('scheduler_tmax', 20)
    scheduler_eta_min = cfg.get('scheduler_eta_min', 1e-5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_tmax, eta_min=scheduler_eta_min
    )

    full_dataset = Noise2VoidDataset(dataset_path, patch_size=patch_size)
    val_split = float(cfg.get('val_split', 0.1))
    val_len = int(len(full_dataset) * val_split)
    train_len = len(full_dataset) - val_len
    if val_len > 0:
        generator = torch.Generator().manual_seed(cfg.get('split_seed', 42))
        train_dataset, val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=generator
        )
    else:
        train_dataset, val_dataset = full_dataset, full_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get('num_workers', 4),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get('val_batch_size', 1),
        shuffle=False,
        num_workers=cfg.get('val_num_workers', 0),
    )


    print(f'train images : {len(train_loader)}')
    print(f'val images : {len(val_loader)}')

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(epochs):   
        train_loss = 0.0
        val_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        # train        
        model.train()
        for i, batch in loop:
            sources, targets, mask = batch[0], batch[1], batch[2]
            sources, targets, mask = sources.to(device), targets.to(device), mask.to(device)

            pred = model(sources)

            loss = pixel_mse_loss(pred, targets, mask)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()   

            # progress bar
            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        

        # validation
        model.eval()
        with torch.no_grad():

            loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)

            for i, batch in loop:
                sources, targets, mask = batch[0], batch[1], batch[2]
                sources, targets, mask = sources.to(device), targets.to(device), mask.to(device)

                pred = model(sources)

                loss = pixel_mse_loss(pred, targets, mask)
                
                val_loss += loss.item()       
                
                # progress bar
                loop.set_description(f'validation')                    


        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)        

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f'Epoch: {epoch+1}\t train_loss: {train_loss}\t val_loss: {val_loss}')

        # scheduler.step(val_loss)
        scheduler.step()

        if best_val_loss > val_loss:
            print('=' * 100)
            print(f'val_loss is improved from {best_val_loss:.4f} to {val_loss:.4f}\t saved current weight')
            print('=' * 100)
            best_val_loss = val_loss
            
            # save weight
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(model.state_dict(), './checkpoints/weight.pth')
            

    os.makedirs('./results', exist_ok=True)
    f = open('./results/train_loss.txt', 'w')
    train_loss_history = list(map(str, train_loss_history))
    for i,v in enumerate(train_loss_history):
        f.write(v+'\n')
    f.close()

    f = open('./results/val_loss.txt', 'w')
    val_loss_history = list(map(str, val_loss_history))
    for i,v in enumerate(val_loss_history):
        f.write(v+'\n')
    f.close()

    print('Finished')


if __name__ == '__main__':
    train()
