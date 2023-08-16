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
from FaceAttr_baseline_model import FaceAttrModel

# from utils.datasets import Get_Dataset
parser = argparse.ArgumentParser(description='OneImage')
parser.add_argument('--experiment', default='pa100k', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--resume', default='checkpoint/v1-1-Resnet18-best_model_params.pth', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--image_path', default='sample_celeba/000019.jpg', type=str, required=False, help='(default=%(default)s)')

def default_loader(path):
    return Image.open(path).convert('RGB')

# every row has 5 attributes.
all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
            'Bangs', 'Big_Lips', 'Big_Nose','Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
            'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
attr_19 = [-1, 1, 1, -1, -1,
           -1, -1, -1, -1, 1,
           -1, -1, -1, -1, -1,
           -1, -1, -1, 1, -1,
           -1, -1, -1, -1,  1,
           1, 1, -1, -1, -1,
           -1, -1, -1, 1, -1,
           -1,  1,  1, -1, 1]
attr_nums = [i for i in range(len(all_attrs))]
selected_attrs = []
for num in attr_nums:
    selected_attrs.append(all_attrs[num])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # args = parser.parse_args()
    #
    #
    # description = all_attrs
    #
    # image_path = args.image_path
    # image = Image.open(image_path)
    # transform = []
    # transform.append(transforms.Resize(size=(224, 224)))  # test no resize operation.
    # transform.append(transforms.ToTensor())
    # transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                       std=[0.5, 0.5, 0.5]))
    #
    # transform = transforms.Compose(transform)
    #
    # image = transform(image)
    # image = torch.unsqueeze(image, 0)  # 将input_data从3维变为4维
    #
    # model = FaceAttrModel(model_type='Resnet18', pretrained=True, selected_attrs=selected_attrs)
    # model = torch.nn.DataParallel(model).cuda()
    #
    # checkpoint = torch.load(args.resume)
    # # # args.start_epoch = checkpoint['epoch']
    # # model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(torch.load(args.resume,map_location='cuda:0'),strict=False)
    #
    # with torch.no_grad():
    #     output = model(image)
    #
    #
    #     # # maximum voting
    #     # if type(output) == type(()) or type(output) == type([]):
    #     #     output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
    #     #     print(output)
    #
    #     # output = torch.sigmoid(output.data).cpu().numpy()
    #     output = output.data.cpu().numpy()

    #
    #     output = np.where(output > 0.5, 1, 0)

    predicted_attributes = []
    for i,_ in enumerate(attr_19):
        if _ == 1:
            predicted_attributes.append(all_attrs[i])

    print('Result：')
    for attribute in predicted_attributes:
        print(attribute)
        # print(f'{attribute[0]}: {attribute[1] * 100:.2f}%')


if __name__ == '__main__':
    main()
