from __future__ import print_function
from __future__ import division
import torch
from torchvision import  transforms
import torch.optim as optim

import pandas as pd 

import copy
import time
import json

from CelebA import get_loader
import torch.nn.functional as F
import utils
from FaceAttr_baseline_model import FaceAttrModel
from Module.focal_loss import FocalLoss
import config as cfg

import argparse
from utils import seed_everything
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='FaceAtrr')
parser.add_argument('--model_type', choices=[
                    'Resnet101','Resnet152','Resnet50',
                    'gc_resnet101','gc_resnet50',
                    'se_resnet101','se_resnet50', 
                    'sk_resnet101', 'sk_resnet50',
                    'sge_resnet101','sge_resnet50', 
                    "shuffle_netv2", 'densenet121',
                    "cbam_resnet101","cbam_resnet50","Resnet18"], 
                    default='Resnet18')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--epochs', default=1, type=int, help='epochs')
parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
parser.add_argument('--learning_rate', default=1e-2, type=float, help='learning_rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--optim_type', choices=['SGD','Adam'], default='SGD')
parser.add_argument('--pretrained', action='store_true', default=True)
parser.add_argument("--loss_type", choices=['BCE_loss', 'focal_loss'], default='BCE_loss')
parser.add_argument("--exp_version",type=str, default="vtest")
#parser.add_argument("--load_model_path", default="resnet101-5d3b4d8f.pth", type=str)
parser.add_argument("--load_model_path", default="Resnet18.pth", type=str)
parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
model_type = args.model_type
optim_type = args.optim_type
momentum = args.momentum
pretrained = args.pretrained
loss_type = args.loss_type
exp_version = args.exp_version
model_path = args.load_model_path
resume_path = args.resume
checkpoint_dir = Path(args.checkpoint)
checkpoint_dir.mkdir(parents=True, exist_ok=True)




class Solver(object):
    
    def __init__(self, epoches, batch_size, learning_rate, model_type, 
        optim_type, momentum, pretrained, loss_type, exp_version):

        self.epoches = epoches 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.selected_attrs = cfg.selected_attrs
        self.momentum = momentum
        #self.device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_dir = cfg.image_dir
        #self.scatter_dir = cfg.scatter_dir
        self.attr_path = cfg.attr_path
        self.pretrained = pretrained
        self.model_type = model_type
        self.build_model(model_type, pretrained)
        #self.create_optim(optim_type)
        self.train_loader = None
        self.validate_loader = None
        self.test_loader = None
        self.log_dir = cfg.log_dir
        #self.use_tensorboard = cfg.use_tensorboard
        self.attr_loss_weight = torch.tensor(cfg.attr_loss_weight).to(self.device)
        self.attr_threshold = cfg.attr_threshold
        self.model_save_path = None
        self.LOADED = False
        self.start_time = 0
        self.loss_type = loss_type
        self.exp_version = exp_version
        self.P = cfg.P
        self.in_attr = cfg.in_attr
        self.out_attr = cfg.out_attr
        self.relationship = cfg.relationship
        torch.cuda.set_device(cfg.DEVICE_ID)

    def build_model(self, model_type, pretrained):
        """Here should change the model's structure""" 
        self.model = FaceAttrModel(model_type, pretrained, self.selected_attrs).to(self.device).to(self.device)
        #print(self.model)

    def set_transform(self, mode):
        transform = []
        if mode == 'train':
            transform.append(transforms.RandomHorizontalFlip())
            transform.append(transforms.RandomRotation(degrees=30))  # 旋转30度
            # transform.append(RandomBrightness())
            # transform.append(RandomContrast())
            # transform.append(RandomHue())
            # transform.append(RandomSaturation())
        # the advising transforms way in imagenet
        # the input image should be resized as 224 * 224 for resnet.
        transform.append(transforms.Resize(size=(224, 224))) # test no resize operation.
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]))
        
        transform = transforms.Compose(transform)
        self.transform = transform


    # self define loss function
    def BCE_loss(self, input_, target):
        # cost_matrix = [1 for i in range(len(self.selected_attrs))]
        loss = F.binary_cross_entropy(input_.to(self.device),  
                                    target.type(torch.FloatTensor).to(self.device), 
                                    weight=self.attr_loss_weight.type(torch.FloatTensor).to(self.device))
        return loss

    def focal_loss(self, inputs, targets):
        focal_loss_func = FocalLoss()
        focal_loss_func.to(self.device)
        return focal_loss_func(inputs, targets)

    def load_model_dict(self, model_state_dict_path):
        self.model_save_path = model_state_dict_path
        self.model.load_state_dict(torch.load(model_state_dict_path,map_location='cuda:0'),strict=False)
        #self.model.load_state_dict(torch.load(model_state_dict_path,map_location='cuda:0'))
        #self.model.load_state_dict(torch.load(model_state_dict_path),strict=False)
        print("The model has loaded !")

    def save_model_dict(self, model_state_dict_path):
        torch.save(self.model.state_dict(), model_state_dict_path)
        print("The model has saved!")

    def train(self, epoch):
        """
        Return: the average trainging loss value of this epoch
        """
        self.model.train()

        self.set_transform("train")

        # to avoid loading dataset repeatedly
        if self.train_loader == None:
            self.train_loader = get_loader(image_dir = self.image_dir, attr_path = self.attr_path, 
                                            selected_attrs = self.selected_attrs, mode="train", 
                                            batch_size=self.batch_size, transform=self.transform)
            print("train_dataset size: {}".format(len(self.train_loader.dataset)))

        temp_loss = 0.0
            
        for batch_idx, samples in enumerate(self.train_loader):


            images, labels = samples
            labels = torch.stack(labels).t() 
            images= images.to(self.device)
            #scatter= scatter.to(self.device)
            outputs = self.model(images)
            self.optim_.zero_grad()
            if self.loss_type == "BCE_loss":
                total_loss = self.BCE_loss(outputs, labels)  

            elif self.loss_type == "focal_loss":
                total_loss = self.focal_loss(outputs, labels)

            total_loss.backward()
            self.optim_.step()
            temp_loss += total_loss.item()
            
            if batch_idx % 50 == 0:
                print("Epoch: {}/{}, training batch_idx : {}/{}, time: {}, loss: {}".format(epoch, self.epoches, 
                                batch_idx, int(len(self.train_loader.dataset)/self.batch_size), 
                                utils.timeSince(self.start_time), total_loss.item()))
            self.scheduler.step()

        return temp_loss/(batch_idx + 1)
        
    def evaluate(self, mode):
        """
        Mode: validate or test mode
        Return: correct_dict: save the average predicting accuracy of every attribute
        """
        self.model.eval()
        self.set_transform(mode)
        data_loader = None

        if self.validate_loader == None and mode == "validate":
            self.validate_loader = get_loader(image_dir = self.image_dir,
                                    attr_path = self.attr_path, 
                                    selected_attrs = self.selected_attrs,
                                    mode=mode, batch_size=self.batch_size, transform=self.transform)
        elif self.test_loader == None and mode == "test":
            self.test_loader = get_loader(image_dir = self.image_dir,
                                    attr_path = self.attr_path, 
                                    selected_attrs = self.selected_attrs,
                                    mode=mode, batch_size=self.batch_size, transform=self.transform)
        if mode == 'validate':
            data_loader = self.validate_loader
        elif mode == 'test':
            data_loader = self.test_loader

        print("{}_dataset size: {}".format(mode,len(data_loader.dataset)))
        
        correct_dict = {}
        for attr in self.selected_attrs:
            correct_dict[attr] = 0

        confusion_matrix_dict = {}
        confusion_matrix_dict['TP'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['TN'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['FP'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['FN'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['precision'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['recall'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['TPR'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['FPR'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['F1'] = [0 for i in range(len(self.selected_attrs))]

        with torch.no_grad():
            for batch_idx, samples in enumerate(data_loader):
                images, labels = samples
                images = images.to(self.device)
                #scatter= scatter.to(self.device)
                labels = torch.stack(labels).t().tolist()
                outputs = self.model(images)
                print(outputs)

                for i in range(self.batch_size):
                    for j, attr in enumerate(self.selected_attrs):
                        if j in self.out_attr:
                            for k in range(len(self.relationship)):
                                if j in self.relationship[k]:
                                    in_attr_j = self.relationship[k][0]
                            pred = 0.5*self.P[j,in_attr_j] * outputs[i].data[in_attr_j] + 0.5*outputs[i].data[j]
                            #print(self.P[j,in_attr_j]* outputs[i].data[in_attr_j],outputs[i].data[j])
                        else:
                        	pred = outputs[i].data[j]                        
                                                                                                
                        pred = outputs[i].data[j]
                        pred = 1 if pred > self.attr_threshold[j] else 0

                        # record accuracy
                        if pred == labels[i][j]:
                            correct_dict[attr] = correct_dict[attr] + 1

                        if pred == 1 and labels[i][j] == 1:
                            confusion_matrix_dict['TP'][j] += 1  # TP
                        if pred == 1 and labels[i][j] == 0:
                            confusion_matrix_dict['FP'][j] += 1  # FP
                        if pred == 0 and labels[i][j] == 1:
                            confusion_matrix_dict['FN'][j] += 1  # TN  
                        if pred == 0 and labels[i][j] == 0:
                            confusion_matrix_dict['TN'][j] += 1  # FN
                if batch_idx % 50 == 0:
                    print("[{}]: Batch_idx : {}/{}, time: {}".format(mode, 
                                batch_idx, int(len(data_loader.dataset)/self.batch_size), 
                                utils.timeSince(self.start_time)))
            i = 0
            # get the average accuracy
            for attr in self.selected_attrs:
                correct_dict[attr] = correct_dict[attr] * 100 / len(self.validate_loader.dataset)
                confusion_matrix_dict['precision'][i] = confusion_matrix_dict['TP'][i]/(confusion_matrix_dict['FP'][i] 
                                                        + confusion_matrix_dict['TP'][i] + 1e-6)
                confusion_matrix_dict['recall'][i]= confusion_matrix_dict['TP'][i]/(confusion_matrix_dict['FN'][i] 
                                                    + confusion_matrix_dict['TP'][i] + 1e-6)
                confusion_matrix_dict['TPR'][i]= confusion_matrix_dict['TP'][i]/(confusion_matrix_dict['TP'][i] 
                                                    + confusion_matrix_dict['FN'][i] + 1e-6)
                confusion_matrix_dict['FPR'][i]= confusion_matrix_dict['FP'][i]/(confusion_matrix_dict['FP'][i] 
                                                    + confusion_matrix_dict['TN'][i] + 1e-6)
                confusion_matrix_dict['F1'][i] = 2*confusion_matrix_dict['precision'][i]*confusion_matrix_dict['recall'][i]/(confusion_matrix_dict['precision'][i] + confusion_matrix_dict['recall'][i] + 1e-6)                                                                          
                i += 1
            
            mean_attributes_acc = 0.0
            for k, v in correct_dict.items():
                mean_attributes_acc += v
            mean_attributes_acc /= len(self.selected_attrs)

        #return correct_dict, confusion_matrix_dict, mean_attributes_acc,temp_loss/(batch_idx + 1)
        return correct_dict, confusion_matrix_dict, mean_attributes_acc



    def fit(self, model_path=""):
        """
        This function is to combine the train and evaluate, finally getting a best model.
        """
        print("-------------------------------")
        print("You method is {}-{}-{}_epoches".format(self.exp_version, self.model_type, self.epoches))
        print("-------------------------------")

        #if isinstance(model_path,torch.nn.DataParallel):
            #model_path = model_path.module
        start_epoch = 0
        if model_path != "":
            self.load_model_dict(model_path)
            print("The model has loaded the state dict on {}".format(model_path))
            
        self.optim_ = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = self.momentum)

        if resume_path:
            if Path(resume_path).is_file():
                print("=> loading checkpoint '{}'".format(resume_path))
                checkpoint = torch.load(resume_path, map_location="cpu")
                start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume_path, checkpoint['epoch']))
                self.optim_.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                    print("=> no checkpoint found at '{}'".format(resume_path))
        
                    
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim_, [30,80], gamma=0.1,last_epoch=start_epoch - 1)

                    
        if args.tensorboard is not None:
            #opts_prefix = "_".join(args.opts)
            train_writer = SummaryWriter(log_dir=args.tensorboard + "/"  + "_train")
            val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + "_val")
 

        train_losses = []
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        #self.scheduler.step()
        eval_acc_dict = {}
        #confusion_matrix_df = None 
        for attr in self.selected_attrs:
            eval_acc_dict[attr] = []
        self.start_time = time.time()
        for epoch in range(self.epoches):
            running_loss = self.train(epoch)
            print("{}/{} Epoch:  in training process average loss: {:.4f}".format(epoch + 1, self.epoches, running_loss))
            print("The running time since the start is : {} ".format(utils.timeSince(self.start_time)))
            average_acc_dict, confusion_matrix_dict, mean_attributes_acc = self.evaluate("validate")
            print("{}/{} Epoch: in evaluating process average accuracy:{}".format(epoch + 1, self.epoches, average_acc_dict))
            print("{}/{} Epoch: the mean accuracy is {}".format(epoch+1, self.epoches, mean_attributes_acc))
            print("The running time since the start is : {} ".format(utils.timeSince(self.start_time)))
            train_losses.append(running_loss)
            #val_losses.append(val_loss)

            average_acc = 0
            
            if args.tensorboard is not None:
                train_writer.add_scalar("loss", running_loss, epoch)
                #val_writer.add_scalar("loss", val_losses, epoch)
                val_writer.add_scalar("acc", mean_attributes_acc, epoch)
            # Record the evaluating accuracy of every attribute at current epoch
            for attr in self.selected_attrs:
                eval_acc_dict[attr].append(average_acc_dict[attr])
                average_acc += average_acc_dict[attr]
            average_acc /= len(self.selected_attrs) # overall accuracy
            
            # find a better model, save it 

            if average_acc > best_acc:
                print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_acc:.3f} to {average_acc:.3f}")
                model_state_dict = self.model.state_dict()
                torch.save(
                    {
                        'epoch': epoch + 1,
                        #'arch': cfg.MODEL.ARCH,
                        'state_dict': model_state_dict,
                        'optimizer_state_dict': self.optim_.state_dict()
                        },
                    str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, best_acc,average_acc)))
                    )
                best_acc = average_acc
            else:
                print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_acc:.3f} ({average_acc:.3f})")



        # save the accuracy in files
        eval_acc_csv = pd.DataFrame(eval_acc_dict, index = [i for i in range(self.epoches)]).T 
        eval_acc_csv.to_csv("./result/" + self.exp_version + '-' +  self.model_type + "-eval_accuracy"+ ".csv");

        # save the loss files
        train_losses_csv = pd.DataFrame(train_losses)
        train_losses_csv.to_csv("./result/" + self.exp_version + '-' +  self.model_type + "-losses" +".csv")

        # load best model weights
        self.model_save_path = "./result/" + self.exp_version + '-' +  self.model_type + "-best_model_params" + ".pth"
        self.model.load_state_dict(best_model_wts)
        self.LOADED = True
        torch.save(best_model_wts, self.model_save_path)        
        print("The model has saved in {}".format(self.model_save_path))
        # test the model with test dataset.
        test_acc_dict, confusion_matrix_dict, mean_attributes_acc = self.evaluate("test")
        test_acc_csv = pd.DataFrame(test_acc_dict, index=['accuracy'])
        test_acc_csv.to_csv("./result/" + self.exp_version + '-' + self.model_type + "-test_accuracy" + '.csv')
        test_confusion_matrix_csv = pd.DataFrame(confusion_matrix_dict, index=self.selected_attrs)
        test_confusion_matrix_csv.to_csv("./result/" + self.exp_version + '-' + self.model_type + '-confusion_matrix.csv', index=self.selected_attrs)

        report_dict = {}
        report_dict["model"] = self.model_type
        report_dict["version"] = self.exp_version
        report_dict["mean_attributes_accuracy"] = mean_attributes_acc
        report_dict["speed"] = self.test_speed()
        report_json = json.dumps(report_dict)
        report_file = open("./result/" + self.exp_version + "-" + self.model_type + "-report.json", 'w')
        report_file.write(report_json)
        report_file.close()
        print(report_dict)

    def test_speed(self, image_num=100, model_path=""):
        if model_path  != "":
            self.model.load_state_dict(torch.load(model_path),map_location='cuda:0')
            print("You load the model params: {}".format(model_path))

        self.model.eval()

        with torch.no_grad():
            self.set_transform(mode="test")
            self.test_loader = get_loader(image_dir = self.image_dir,
                                    attr_path = self.attr_path, 
                                    selected_attrs = self.selected_attrs,
                                    mode="test", batch_size=image_num, transform=self.transform)
            
            for idx, samples in enumerate(self.test_loader):
                images, labels = samples
                images = images.to(self.device)
                #scatter= scatter.to(self.device)
                labels = torch.stack(labels).t().tolist()
                start_time = time.time()
                #outputs = self.model(images,scatter)
                end_time = time.time()

                if idx == 0:
                    speed = image_num / (end_time - start_time)
                    print("You test {} images. The cost time is {}. The speed is {} images/s.".format(image_num,(end_time - start_time),speed))
                    print("---------------------------------------------------------")
                    return end_time-start_time
                    break

#--------------- exe ----------------------------- #
if __name__ == "__main__":
    seed_everything()

    # too more params to send.... not a good way....use the config.py to improve it
    solver = Solver(epoches=epochs, batch_size=batch_size, learning_rate=learning_rate, model_type=model_type, optim_type=optim_type, momentum=momentum, pretrained=pretrained, loss_type=loss_type,exp_version=exp_version)
    try:
        #model_path = nn.DataParallel(model_path).cuda()
        solver.fit(model_path=model_path)
    except InterruptedError:
        print("early stop...")
        print("save the model dict....")
        solver.save_model_dict(exp_version+"_"+model_path + "_earlystop.pth")
