import os
import time
import torch
import random
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataloader import H3R_Datasets
from models.h3r_head import HeatMapLandmarker
from utils.utils import AverageMeter, compute_nme
from models.heatmapmodel import lmks2heatmap, binary_heatmap_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                transforms.RandomErasing(p = 0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value = 'random')
                               ])


transform_valid = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
                                     ])



def train_one_epoch(traindataloader, model, optimizer, epoch, args = None):
    model.train()
    losses = AverageMeter()
    start_time = time.time()

    for step, data in enumerate(traindataloader):

        img, lmksGT = data

        img = img.to(device)  # B, 3, 128, 128
        
        lmksGT = lmksGT.view(lmksGT.shape[0], -1, 2)  # B, 106, 2
        lmksGT = lmksGT * 128


        heatGT = lmks2heatmap(lmksGT, args.random_round, args.random_round_with_gaussian)
        heatGT = heatGT.to(device)

        heatPRED, lmksPRED = model(img.to(device))

        loss = binary_heatmap_loss(heatPRED, heatGT, 100)  
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())

        rate = (step + 1) / len(traindataloader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.5f}".format(int(rate * 100), a, b, loss.item() * 100), end="")
    print('time::{:.3f}'.format(time.time() - start_time))

    return losses.avg


def validate(valdataloader, model, optimizer, epoch, args):
    if not os.path.isdir(args.snapshot):
        os.makedirs(args.snapshot)

    logFilepath  = os.path.join(args.snapshot, args.log_file)

    logFile  = open(logFilepath, 'a')

    model.eval()
    losses = AverageMeter()

    nme = []

    with torch.no_grad():

        for step, data in enumerate(valdataloader):

            img, lmksGT = data

            img = img.to(device)  # B, 3, 256, 256
        
            lmksGT = lmksGT.view(lmksGT.shape[0], -1, 2)  # B, 106, 2
            lmksGT = lmksGT * 128


            heatGT = lmks2heatmap(lmksGT, args.random_round, args.random_round_with_gaussian)
            heatGT = heatGT.to(device)
            
            
            heatPRED, lmksPRED = model(img.to(device))

        
            loss = binary_heatmap_loss(heatPRED, heatGT, 100)

            # loss = mse_loss(heatPRED, heatGT)

            # Loss
            nme_batch = list(compute_nme(lmksPRED, lmksGT))


            nme += nme_batch
            losses.update(loss.item())
            rate = (step + 1) / len(valdataloader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtest loss: {:^3.0f}%[{}->{}]{:.5f},{:.3f}".format(int(rate * 100), a, b, loss.item(), np.mean(nme)), end="")
        print()


    
    message = f" Epoch:{epoch}. Lr:{optimizer.param_groups[0]['lr']}. Loss :{losses.avg}. NME :{np.mean(nme)}"
    print(message)
    logFile.write(message + "\n")

    return losses.avg, np.mean(nme)


def main(args):
    model = HeatMapLandmarker(pretrained = True)

    if args.resume != '':
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
        print('load weights from' + args.resume)
    
    model.to(device)

    train_dataset = H3R_Datasets(args.dataroot, transform, img_root = os.path.realpath('./data'), img_size = 128)
    test_dataset = H3R_Datasets(args.test_dataroot, transform_valid, img_root = os.path.realpath('./data'), img_size = 128)
    val_dataset = H3R_Datasets(args.val_dataroot, transform_valid, img_root = os.path.realpath('./data'), img_size = 128)

    traindataloader = DataLoader(
        train_dataset,
        batch_size = args.train_batchsize,
        shuffle = True,
        num_workers = 8,)
    
    testdataloader = DataLoader(
        test_dataset,
        batch_size = args.val_batchsize,
        shuffle = False,
        num_workers = 8,)

    
    validdataloader = DataLoader(
        val_dataset,
        batch_size = args.val_batchsize,
        shuffle = False,
        num_workers = 8,
        drop_last = True)
    
    optimizer = torch.optim.Adam(
        [{
            'params': model.parameters()
        }],
        lr = args.lr)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size ,gamma = args.gamma)
    
    best_nme = 100
    for epoch in range(70):
        train_one_epoch(traindataloader, model, optimizer, epoch, args)
        _, nme = validate(testdataloader, model, optimizer, epoch, args)
        _, nme_valid = validate(validdataloader, model, optimizer, epoch, args)
        
        scheduler.step()
        
        if nme < best_nme:
            best_nme = nme
            torch.save(model.state_dict(), 'H3R.pth')


def parse_args():
    parser = argparse.ArgumentParser(description = 'H3R')

    parser.add_argument(
        '--snapshot',
        default = './checkpoint/',
        type = str,
        metavar = 'PATH')

    parser.add_argument(
        '--log_file', default = "log.txt", type=str)

    # --dataset
    parser.add_argument(
        '--dataroot',
        default = './data/train_data/list.txt',
        type = str,
        metavar = 'PATH')
    
    parser.add_argument(
        '--test_dataroot',
        default = './data/test_data/list.txt',
        type = str,
        metavar = 'PATH')
        
    parser.add_argument(
        '--val_dataroot',
        default = './data/valid_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default = 16, type=int)
    parser.add_argument('--val_batchsize', default = 8, type=int)
    parser.add_argument('--lr', default = 0.0001, type=float)
    parser.add_argument('--step_size', default = 30, type=float)
    parser.add_argument('--gamma', default = 0.1, type=float)
    parser.add_argument('--resume', default = "", type=str)

    parser.add_argument('--random_round', default = 0, type = int)

    parser.add_argument('--random_round_with_gaussian', default = 1, type=int)
    parser.add_argument('--mode', default = 'train', type = str)



    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)


    






        


                                
                                


