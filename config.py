import torch
import numpy as np

# ---------------- all data on the same device ----#
DEVICE_ID = 0

# ---------------- experiment mark -----------------#
exp_version = "v3"


# -----------------dataset spilit---------------------#
train_end_index = 162770 + 1
validate_end_index = 182637 + 1
test_end_index = 202599 + 1 


# for test
#train_end_index = 128 + 1
#validate_end_index = 256 + 1
#test_end_index = 320 + 1

# ------------- Path setting --------------------- #

log_dir = "./log"
# You should download the celeba dataset in the root dir.

# the dataset local path.
# image_dir = "../CelebA/Img/img_align_celeba/" 
# attr_path = "../CelebA/Anno/list_attr_celeba.txt"

#the dataset path run on server.
image_dir = "/media/hulu/home1/data/CelebA/Img/img_align_celeba/" 
attr_path = "/media/hulu/home1/data/CelebA/Anno/list_attr_celeba.txt"
P = None
# P=np.load("/home/hulu/home2/ln/paper-5/attri-analysis-ln-relationship/attribute_map.npy") #relationship_matrix
scatter_dir = "/home/hulu/home2/scatter_norm_coeff/"
# ----------- model/train/test configuration ---- #
"""
epoches = 50  # 50

batch_size = 32

learning_rate = 0.001

model_type = "Resnet101"  # 34 50 101 152

optim_type = "SGD"

momentum = 0.9

pretrained = True
"""

# ------------- loss type----------------------------- #
# loss_type = "BCE_loss"  #  focal_loss
# loss_type = "focal_loss"

# Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
focal_loss_alpha = 0.8
focal_loss_gamma = 2
size_average = False
# -------------- Attribute configuration --------- #

# every row has 5 attributes.
all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
            'Bangs', 'Big_Lips', 'Big_Nose','Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
            'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' 
]


"""
all_attrs = ['Attractive','Bald', 'Bangs', 'Black_Hair','Blond_Hair','Brown_Hair', 'Chubby', 'Double_Chin', 
            'Eyeglasses', 'Goatee', 'Gray_Hair','Male', 'Mustache', 'Pale_Skin', 'Receding_Hairline',  
            'Smiling', 'Wearing_Hat'
]
"""
relationship = [[4, 20], [5, 24], [9, 24], [9, 39], [10, 24], [12, 39], [13, 20], [14, 7], [14, 20], [16, 20], [17, 20], [17, 24], [22, 20], [23, 24], [26, 24], [26, 39], [29, 2], [29, 18], [29, 19], [29, 24], [29, 31], [29, 36], [29, 39], [30, 20], [37, 24], [38, 20]]
in_attr =  [4, 5, 9, 10, 12, 13, 14, 16, 17, 22, 23, 26, 29, 30, 37, 38]
out_attr = [2, 7, 18, 19, 20, 24, 31, 36, 39]
#attr_loss_weight = [1 for i in range(len(all_attrs))]
#attr_loss_weight[0] = 10  # attractive 
#attr_loss_weight[5] = 5  # brown_hair

#for num in in_attr:
    #selected_attrs.append(all_attrs[num])


# To be optimized
attr_nums = [i for i in range(len(all_attrs))] 
attr_loss_weight = [1 for i in range(len(all_attrs))]
attr_loss_weight[0] = 10  # attractive 
attr_loss_weight[5] = 5  # brown_hair

selected_attrs = [] 
for num in attr_nums:
    selected_attrs.append(all_attrs[num])
#print(selected_attrs)


# To solve the sample imbalance called rescaling, If the threshold > m+ /(m+ + m-), treat it as a positive sample. 
attr_threshold = [0.5 for i in range(len(all_attrs))]  

""" Cause worse accuracy result.
sample_csv = pd.read_csv('sample_num.csv')
attr_threshold = (sample_csv['positive sample']/(sample_csv['positive sample'] + sample_csv['negative sample'])).tolist()
"""

# -------------- Tensorboard --------------------- #
use_tensorboard = False
