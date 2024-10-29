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
import torchvision.models as models

import os

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
    def __init__(self, root, split='train', target_type='attr', transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        
        # Read the data files
        attr_data = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv'))
        partition_data = pd.read_csv(os.path.join(root, 'list_eval_partition.csv'))
        
        # Use only the first n-10000 data points
        n = len(attr_data)
        target_n = n - 10000
        attr_data = attr_data.head(target_n)
        partition_data = partition_data.head(target_n)
        
        # Filter based on the provided split
        split_dict = {'train': 0, 'valid': 1, 'test': 2}
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


def set_args():
    parser = argparse.ArgumentParser(description="Arguments for training the model.")
    parser.add_argument(
        "--form",
        type=str,
        default="logistic",
        help="Which regularization to use",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.0,
        help="Lambda parameter",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../models",
        help="Directory to save models.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed to use for random initialization."
    )

    args = parser.parse_args()


    return args


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    lam_fair = args.lambda_reg
    form = args.form

    # Create directory for saving models
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Usage
    root_dir = '../celeba'
    trainset = SimpleCelebA(root=root_dir, split='train', target_type=attrs, transform=preprocess)
    validset = SimpleCelebA(root=root_dir, split='valid', target_type=attrs, transform=preprocess)
    testset  = SimpleCelebA(root=root_dir, split='test',  target_type=attrs, transform=preprocess)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=32, shuffle=False, num_workers=2)

    # model = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2') #pretrained = True if you want
    # Instantiate the model architecture
    model = models.wide_resnet50_2()

    # Load the weights
    model.load_state_dict(torch.load('./fairsurrogates/wide_resnet50_2.pth'))
    model.fc = nn.Linear(2048, 1, bias=True)

    ti = attrs.index("Smiling")
    si = attrs.index("Male")
    
    Pmale = [(trainset.attr[:,si].bool() & trainset.attr[:,ti].bool()).float().mean(), (trainset.attr[:,si].bool() & ~trainset.attr[:,ti].bool()).float().mean()]
    Pfem = [(~trainset.attr[:,si].bool() & trainset.attr[:,ti].bool()).float().mean(), (~trainset.attr[:,si].bool() & ~trainset.attr[:,ti].bool()).float().mean()]


    ploss = nn.BCEWithLogitsLoss()
    if form == "logistic":
        def floss(outputs, sens_attr, y):
            return -lam_fair/64 * (F.logsigmoid(outputs[sens_attr & y.bool()]).sum()/Pmale[0] + F.logsigmoid(-outputs[~sens_attr & y.bool()]).sum()/Pfem[0]) +\
            -lam_fair/64 * (F.logsigmoid(outputs[sens_attr & ~y.bool()]).sum()/Pmale[1] + F.logsigmoid(-outputs[~sens_attr & ~y.bool()]).sum()/Pfem[1]) 
    elif form == "linear":
        def floss(outputs, sens_attr, y):
            return lam_fair/64 * torch.abs(-outputs[sens_attr & y.bool()].sum()/Pmale[0] + outputs[~sens_attr & y.bool()].sum()/Pfem[0]) +\
            lam_fair/64 * torch.abs(-outputs[sens_attr & ~y.bool()].sum()/Pmale[1] + outputs[~sens_attr & ~y.bool()].sum()/Pfem[1])
    elif form == "weighting":
        def floss(outputs, sens_attr, y):
            return -lam_fair/64 * (F.logsigmoid(outputs[sens_attr & y.bool()]).sum()/Pmale[0] - F.logsigmoid(outputs[~sens_attr & y.bool()]).sum()/Pfem[0]) +\
            -lam_fair/64 * (F.logsigmoid(outputs[sens_attr & ~y.bool()]).sum()/Pmale[1] - F.logsigmoid(outputs[~sens_attr & ~y.bool()]).sum()/Pfem[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    model.to(device)
    torch.cuda.empty_cache()

    def calc_loss(data):
        inputs, labels = data
        inputs, labels, sens_attr = inputs.to(device), labels[:,ti].float().to(device), labels[:,si].bool().to(device)
        labels_bool = labels.bool()
        optimizer.zero_grad()
        outputs = model(inputs).reshape(-1)
        loss = ploss(outputs, labels) + floss(outputs, sens_attr, labels_bool)
        loss.backward()
        preds = (outputs >= 0).float()
        # unfairness_y1 = 0.5*torch.abs((torch.sigmoid(outputs[y.bool() & s])).float().mean() - (torch.sigmoid(outputs[y.bool() & ~s])).float().mean())
        # unfairness_y0 = 0.5*torch.abs((torch.sigmoid(outputs[~y.bool() & s])).float().mean() - (torch.sigmoid(outputs[~y.bool() & ~s])).float().mean())

        unfairness = torch.tensor([preds[ sens_attr & labels_bool].sum(), preds[ sens_attr & labels_bool].shape[0],
                                preds[~sens_attr & labels_bool].sum(), preds[~sens_attr & labels_bool].shape[0]]) #msmiling, m, fsmiling, f
        return ((labels == preds).float().mean(), loss, unfairness)


    print_every = 200
    valid_batches = 32
    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.0
        running_acc = 0.0
        running_unfair = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            (acc, loss, unfair) = calc_loss(data)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += acc.item()
            running_unfair += unfair

            if i % print_every == (print_every - 1):    # print every 200 mini-batches
                print('[%d, %5d]' % (epoch + 1, i + 1))
                valid_loss = 0.0
                valid_acc = 0.0
                valid_iter = iter(validloader)
                for vi in range(valid_batches):
                    (new_valid_acc, new_valid_loss, _) = calc_loss(next(valid_iter))
                    valid_loss += new_valid_loss
                    valid_acc += new_valid_acc.item()
                scheduler.step(valid_loss)
                print('Training Accuracy: %.3f, Validation Accuracy: %.3f, Unfairness: %.3f' % (running_acc / print_every,
                                                            valid_acc/valid_batches,
                                                            running_unfair[2]/running_unfair[3] -
                                                            running_unfair[0]/running_unfair[1]))
                print('Training Loss: %.3f, Validation Loss: %.3f' % (running_loss / print_every, valid_loss/valid_batches))
                print('')
                running_loss = 0.0
                running_acc = 0.0
                running_unfair = 0.0
                sleep(1)

    print('Finished Training')

    torch.save(model.state_dict(), f"{args.model_dir}/model_{form}_{lam_fair}_{args.seed}_eo.pth")


if __name__ == "__main__":
    args = set_args()
    main(args)