import argparse
import sys
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from time import sleep
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
import re

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import pandas as pd

from PIL import Image

class SimpleCelebA(Dataset):
    def __init__(self, root, split='train', target_type='attr', transform=None, split_=True):
        super().__init__()
        self.root = root
        self.transform = transform
        
        # Read the data files
        attr_data = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv'))
        partition_data = pd.read_csv(os.path.join(root, 'list_eval_partition.csv'))
        
        # Use only the first n-10000 data points
        n = len(attr_data)
        target_n = 10000
        attr_data = attr_data.tail(target_n)
        partition_data = partition_data.tail(target_n)
        
        # Filter based on the provided split
        split_dict = {'train': 0, 'valid': 1, 'test': 2}
        if split_:
            partition_data = partition_data[partition_data['partition'] == split_dict[split]]
        
        # Merge datasets on image_id and filter attributes
        self.data = pd.merge(partition_data, attr_data, on='image_id')
        
        # Convert attributes from -1 to 0
        self.data[target_type] = (self.data[target_type] == 1).astype(int)
        self.attr = torch.FloatTensor(self.data[target_type].values)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root, 'img_align_celeba/img_align_celeba', self.data.iloc[idx, 0])
        image = Image.open(img_name)
        
        attrs = self.attr[idx, :]
        if self.transform:
            image = self.transform(image)
            
        return image, attrs

attrs = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young '.split()
ti = attrs.index("Smiling")
si = attrs.index("Male")



def set_args():
    parser = argparse.ArgumentParser(description="Arguments for training the model.")
    parser.add_argument(
        "--form",
        type=str,
        default="logistic",
        help="Which regularization to use",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../model_results_sim_dp-celeba-fairsurrogates-r2",
        help="Directory to save models.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed to use for random initialization."
    )

    args = parser.parse_args()


    return args


def calc_loss_batch(model, data, Pmale, Pfem):
    inputs, labels = data
    inputs, labels, sens_attr = inputs.to(device), labels[:,ti].float().to(device), labels[:,si].bool().to(device)
    outputs = model(inputs).reshape(-1)
    acc  = ((outputs > 0)*(labels.reshape(-1)) + (outputs <= 0)*(1-labels.reshape(-1))).sum()
    unfairness = torch.sigmoid(outputs[sens_attr]).sum()/Pmale - torch.sigmoid(outputs[~sens_attr]).sum()/Pfem
    return acc, unfairness


def get_metrics(model, dataloader, Pmale, Pfem):
    accuracy = 0
    unfairness = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            (acc, unfair) = calc_loss_batch(model, data, Pmale, Pfem)
            accuracy += acc
            unfairness += unfair
        n = dataloader.dataset.data.shape[0]
        accuracy /= n
        unfairness /= n
    return accuracy.cpu().item(), unfairness.cpu().item()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create directory for saving models
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Usage
    root_dir = '../celeba'
    testset  = SimpleCelebA(root=root_dir, split='train',  target_type=attrs, transform=preprocess, split_=False)

    testloader  = torch.utils.data.DataLoader(testset,  batch_size=32, shuffle=False, num_workers=2)

    model = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2') #pretrained = True if you want
    model.fc = nn.Linear(2048, 1, bias=True)

    ti = attrs.index("Smiling")
    si = attrs.index("Male")

    (Pmale, Pfem) = (testset.attr[:,si].float().mean(), 1 - testset.attr[:,si].float().mean())
    results_df = pd.DataFrame()
    for filename in os.listdir(args.model_dir):
        if filename.endswith(".pth"):
            # Extract parameters from filename using regex
            regex = rf"model_{args.form}_([\d.]+)_(\d+).pth"
            match = re.search(regex, filename)
            if match is not None:
                lambda_reg, seed = map(float, match.groups())
                model_path = os.path.join(args.model_dir, filename)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                acc, unfairness = get_metrics(model, testloader, Pmale, Pfem)
                results = pd.DataFrame(data={'lambda_reg': [lambda_reg], 'Accuracy': [acc], 'unfairness': [unfairness]})
                results_df = pd.concat([results_df, results], ignore_index=True)
    results_df.to_csv("./celeba_dp_surrogates-r3.csv", index=False)
    

if __name__ == "__main__":
    args = set_args()
    main(args)
