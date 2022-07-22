import cv2
import torch
import argparse
from torchvision import transforms
from models.h3r_head import HeatMapLandmarker



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_valid = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
                                     ])

def get_landmarks(model, image_path):

    model.eval()

    img_raw = cv2.imread(image_path)
    img_raw = cv2.resize(img_raw, (128, 128))
    img = transform_valid(img_raw)
    img = img.unsqueeze(0)

    img = img.to(device)
    
    _, lmksPRED = model(img.to(device))

    return lmksPRED[0], img_raw

def parse_args():
    parser = argparse.ArgumentParser(description = 'H3R')
   
    parser.add_argument('--resume', default = "weights/H3R.pth", type=str)

    parser.add_argument('--random_round', default = 0, type = int)

    parser.add_argument('--random_round_with_gaussian', default = 1, type=int)

    args = parser.parse_args()
    return args


def main(args):
    model = HeatMapLandmarker(pretrained = True)

    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint)
    print('load weights from' + args.resume)
    
    model.to(device)

    image_path = 'test_samples/test.jpg'
    lmsPRED, img_raw = get_landmarks(model, image_path)

    for landmark in lmsPRED:
        cv2.circle(img_raw, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), 1)
    
    cv2.imwrite('test_samples/result.jpg', img_raw)

if __name__ == "__main__":
    args = parse_args()
    main(args)