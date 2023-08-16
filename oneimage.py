import argparse
import os
import shutil
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import model as models
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

# from utils.datasets import Get_Dataset
parser = argparse.ArgumentParser(description='OneImage')
parser.add_argument('--experiment', default='pa100k', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--resume', default='checkpoint/41.pth', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--image_path', default='sample_pa100k/000001.jpg', type=str, required=False, help='(default=%(default)s)')

def default_loader(path):
    return Image.open(path).convert('RGB')
class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform = None, loader = default_loader):
        images = []
        labels = open(label).readlines()
        for line in labels:
            items = line.split()
            img_name = items.pop(0)
            if os.path.isfile(os.path.join(root, img_name)):
                cur_label = tuple([int(v) for v in items])
                images.append((img_name, cur_label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)

attr_nums = {}

attr_nums['pa100k'] = 26
attr_nums['rap'] = 51
attr_nums['peta'] = 35


description = {}
description = {'pa100k': ['Female',
                          'AgeOver60',
                          'Age18-60',
                          'AgeLess18',
                          'Front',
                          'Side',
                          'Back',
                          'Hat',
                          'Glasses',
                          'HandBag',
                          'ShoulderBag',
                          'Backpack',
                          'HoldObjectsInFront',
                          'ShortSleeve',
                          'LongSleeve',
                          'UpperStride',
                          'UpperLogo',
                          'UpperPlaid',
                          'UpperSplice',
                          'LowerStripe',
                          'LowerPattern',
                          'LongCoat',
                          'Trousers',
                          'Shorts',
                          'Skirt&Dress',
                          'boots'], 'peta': ['Age16-30',
                                             'Age31-45',
                                             'Age46-60',
                                             'AgeAbove61',
                                             'Backpack',
                                             'CarryingOther',
                                             'Casual lower',
                                             'Casual upper',
                                             'Formal lower',
                                             'Formal upper',
                                             'Hat',
                                             'Jacket',
                                             'Jeans',
                                             'Leather Shoes',
                                             'Logo',
                                             'Long hair',
                                             'Male',
                                             'Messenger Bag',
                                             'Muffler',
                                             'No accessory',
                                             'No carrying',
                                             'Plaid',
                                             'PlasticBags',
                                             'Sandals',
                                             'Shoes',
                                             'Shorts',
                                             'Short Sleeve',
                                             'Skirt',
                                             'Sneaker',
                                             'Stripes',
                                             'Sunglasses',
                                             'Trousers',
                                             'Tshirt',
                                             'UpperOther',
                                             'V-Neck'], 'rap': ['Female',
                                                                'AgeLess16',
                                                                'Age17-30',
                                                                'Age31-45',
                                                                'BodyFat',
                                                                'BodyNormal',
                                                                'BodyThin',
                                                                'Customer',
                                                                'Clerk',
                                                                'BaldHead',
                                                                'LongHair',
                                                                'BlackHair',
                                                                'Hat',
                                                                'Glasses',
                                                                'Muffler',
                                                                'Shirt',
                                                                'Sweater',
                                                                'Vest',
                                                                'TShirt',
                                                                'Cotton',
                                                                'Jacket',
                                                                'Suit-Up',
                                                                'Tight',
                                                                'ShortSleeve',
                                                                'LongTrousers',
                                                                'Skirt',
                                                                'ShortSkirt',
                                                                'Dress',
                                                                'Jeans',
                                                                'TightTrousers',
                                                                'LeatherShoes',
                                                                'SportShoes',
                                                                'Boots',
                                                                'ClothShoes',
                                                                'CasualShoes',
                                                                'Backpack',
                                                                'SSBag',
                                                                'HandBag',
                                                                'Box',
                                                                'PlasticBag',
                                                                'PaperBag',
                                                                'HandTrunk',
                                                                'OtherAttchment',
                                                                'Calling',
                                                                'Talking',
                                                                'Gathering',
                                                                'Holding',
                                                                'Pusing',
                                                                'Pulling',
                                                                'CarryingbyArm',
                                                                'CarryingbyHand']}

def Get_Label(experiment):
    if experiment == 'pa100k':
        return attr_nums['pa100k'], description['pa100k']
    elif experiment == 'rap':
        return attr_nums['rap'], description['rap']
    elif experiment == 'peta':
        return attr_nums['peta'], description['peta']
def main():
    # global args
    args = parser.parse_args()

    attr_num, description = Get_Label(args.experiment)

    image_path = args.image_path
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)  # 将input_data从3维变为4维

    model = models.__dict__['convnext_base'](attr_num=attr_num)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    # best_accu = checkpoint['best_accu']
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        output = model(image)

        # maximum voting
        if type(output) == type(()) or type(output) == type([]):
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])

        output = torch.sigmoid(output.data).cpu().numpy()
        output = np.where(output > 0.5, 1, 0)
        predicted_attributes = []
        for i, _ in enumerate(output[0]):
            if _ == 1:
                predicted_attributes.append(description[i])

    print('Result：')
    for attribute in predicted_attributes:
        print(attribute)
        # print(f'{attribute[0]}: {attribute[1] * 100:.2f}%')

if __name__ == '__main__':
    main()
