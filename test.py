import os
import torch
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataloader import H3R_Datasets
from models.h3r_head import HeatMapLandmarker
from utils.utils import compute_nme


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_valid = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
                                     ])


def validate(valdataloader, model, args):

    model.eval()

    nme = []

    with torch.no_grad():

        for step, data in enumerate(valdataloader):

            img, lmksGT = data

            img = img.to(device)  # B, 3, 256, 256
        
            lmksGT = lmksGT.view(lmksGT.shape[0], -1, 2)  # B, 106, 2
            lmksGT = lmksGT * 128
            
            heatPRED, lmksPRED = model(img.to(device))

            nme_batch = list(compute_nme(lmksPRED, lmksGT))

            nme += nme_batch
            rate = (step + 1) / len(valdataloader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtest loss: {:^3.0f}%[{}->{}],{:.3f}".format(int(rate * 100), a, b, np.mean(nme)), end="")
        print()


    print('NME:{:.5f}'.format(np.mean(nme)))

def main(args):
    model = HeatMapLandmarker(pretrained = True)

    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint)
    print('load weights from' + args.resume)
    
    model.to(device)

    val_dataset = H3R_Datasets(args.val_dataroot, transform_valid, img_root = os.path.realpath('./data'), img_size = 128)

    
    validdataloader = DataLoader(
        val_dataset,
        batch_size = args.val_batchsize,
        shuffle = False,
        num_workers = 8,
        drop_last = True)
    
    validate(validdataloader, model, args)
        

def parse_args():
    parser = argparse.ArgumentParser(description = 'H3R')
   
    parser.add_argument(
        '--val_dataroot',
        default = './data/valid_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--val_batchsize', default = 8, type=int)
    parser.add_argument('--resume', default = "weights/H3R.pth", type=str)

    parser.add_argument('--random_round', default = 0, type = int)

    parser.add_argument('--random_round_with_gaussian', default = 1, type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)