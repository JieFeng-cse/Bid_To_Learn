import os 
import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image
from sklearn.preprocessing import label_binarize
import torch
from torch.serialization import save
from torch.utils.data import Dataset
from torch.distributions.normal import Normal
from torchvision import transforms,models,utils
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
from layers import Agent, ResnetAgent
from models import Resnet9, Resnet1, Resnet2
import time
import numpy as np
import torch.nn.functional as F
import math
# SEED = 27
train_path = './dataset_cifar10/train'
val_path = './dataset_cifar10/test'

#data loader
class MyDataset(Dataset):
    def __init__(self, data_path:str, transform=None):
        self.data_path = data_path
        self.path_list = []
        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),   
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transform
        for (dirpath, dirnames, filenames) in os.walk(data_path):
            self.path_list += [os.path.join(dirpath, file) for file in filenames]
    def __getitem__(self, idx: int):
        # img to tensor and label to tensor
        img_path = self.path_list[idx]
        if img_path.split('/')[3] == 'airplane': 
            label = 0
        elif img_path.split('/')[3] == 'automobile':
            label = 1
        elif img_path.split('/')[3] == 'bird':
            label = 2
        elif img_path.split('/')[3] == 'cat':
            label = 3
        elif img_path.split('/')[3] == 'deer':
            label = 4
        elif img_path.split('/')[3] == 'dog':
            label = 5
        elif img_path.split('/')[3] == 'frog':
            label = 6
        elif img_path.split('/')[3] == 'horse':
            label = 7
        elif img_path.split('/')[3] == 'ship':
            label = 8
        elif img_path.split('/')[3] == 'truck':
            label = 9
        else:
            label = int(img_path.split('/')[3]) 
        label = torch.as_tensor(label, dtype=torch.int64)
        img = Image.open(img_path)
        img = self.transform(img)
        if img.shape[0]==1:
            img=img.repeat(3,1,1)
        return img, label
    def __len__(self) -> int:
        return len(self.path_list)

class MyDatasetval(Dataset):
    def __init__(self, data_path:str, transform=None):
        self.data_path = data_path
        self.path_list = []
        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.Resize(size = (32,32)),#image size            
                transforms.ToTensor(),   #
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transform
        for (dirpath, dirnames, filenames) in sorted(os.walk(data_path)):
            # print(dirpath) 
            self.path_list += [os.path.join(dirpath, file) for file in filenames]
    def __getitem__(self, idx: int):
        # img to tensor and label to tensor
        img_path = self.path_list[idx]
        if img_path.split('/')[3] == 'airplane': 
            label = 0
        elif img_path.split('/')[3] == 'automobile':
            label = 1
        elif img_path.split('/')[3] == 'bird':
            label = 2
        elif img_path.split('/')[3] == 'cat':
            label = 3
        elif img_path.split('/')[3] == 'deer':
            label = 4
        elif img_path.split('/')[3] == 'dog':
            label = 5
        elif img_path.split('/')[3] == 'frog':
            label = 6
        elif img_path.split('/')[3] == 'horse':
            label = 7
        elif img_path.split('/')[3] == 'ship':
            label = 8
        elif img_path.split('/')[3] == 'truck':
            label = 9
        else:
            label = int(img_path.split('/')[3]) 
        label = torch.as_tensor(label, dtype=torch.int64)
        img = Image.open(img_path)
        img = self.transform(img)
        if img.shape[0]==1:
            img=img.repeat(3,1,1)
        return img, label
    def __len__(self) -> int:
        return len(self.path_list)
#training
def train_seperate(e, train_loader, device,agent1, agent2, optimizer1, optimizer2, sched1, sched2, writer):
    agent1.train()
    agent2.train()
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    i = 0
    for images, labels in train_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
    
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        #get the prediction results and bids 
        pred1, bid1 = agent1(images)
        pred2, bid2 = agent2(images)

        #calculate the cross-entropy loss
        pred1 = F.softmax(pred1, dim=1)
        log_p1 = (torch.log(pred1+math.exp(-70))) #add a small number to avoid nan
        #one_hot means it is one if it corresponding to the right label, else zero
        one_hot = torch.zeros(pred1.shape,dtype=torch.double,device=device).scatter_(1, torch.unsqueeze(labels, dim=1),1)
        #sigma is the standard deviation for bids
        sigma = 0.1*torch.ones_like(bid1).to(device)*torch.sqrt(torch.tensor([2])).to(device)
        ones_t = torch.ones_like(bid1).to(device)
        bid_diff = bid1-bid2.detach()
        dis1_1 = Normal(bid_diff, sigma)
        dis1_2_fp = Normal(bid_diff+0.01*ones_t, sigma)
        dis1_2_sp = Normal(bid_diff-0.01*ones_t, sigma)
        p_a1_loss = dis1_1.cdf(torch.zeros_like(bid1).to(device))
        a_1 = torch.sum(one_hot*log_p1,dim=1).unsqueeze(1).double()

        fp_loss_1 = torch.exp(bid2.detach()+0.005)*(1-p_a1_loss)
        sp_loss_1 = torch.exp(bid1.to(device)+0.005)*(1-p_a1_loss)
        #this is just a small experiment to have more balanced result
        #generally, we use loss1 = -a_1 * p_a1_loss + sp_loss_1*0.9 + fp_loss_1*0.1, no first if
        if e < 50:
            loss1 = -a_1 + F.smooth_l1_loss(torch.mean(bid1), torch.mean(bid2.detach()))
        else:
            loss1 = -a_1 * p_a1_loss + sp_loss_1*0.9 + fp_loss_1*0.1 #+ F.smooth_l1_loss(torch.mean(bid1), torch.mean(bid2.detach()))

        loss1 = loss1.sum()
        loss1/=pred1.shape[0]

        # loss2 = F.cross_entropy(pred2, labels)
        pred2 = F.softmax(pred2, dim=1) 
        
        log_p2 = (torch.log(pred2+math.exp(-70)))
    
        bid_diff2 = bid2-bid1.detach()
        dis2_1 = Normal(bid_diff2, sigma)
        dis2_2_fp = Normal(bid_diff2+0.01*ones_t, sigma) #fp means first price
        dis2_2_sp = Normal(bid_diff2-0.01*ones_t, sigma)
        p_a2_loss = dis2_1.cdf(torch.zeros_like(bid2).to(device))
        a_2 = torch.sum(one_hot*log_p2,dim=1).unsqueeze(1).double()
        fp_loss_2 = torch.exp(bid2+0.005)*(1-p_a2_loss)
        sp_loss_2 = torch.exp(bid1.detach().to(device)+0.005)*(1-p_a2_loss) 
        
        if e<50:
            loss2 = -a_2 + F.smooth_l1_loss(torch.mean(bid1.detach()), torch.mean(bid2))
        else:
            loss2 = -a_2 * p_a2_loss + sp_loss_2*0.9 + fp_loss_2*0.1 
        
        loss2 = loss2.sum()
        loss2/=pred2.shape[0] 
        loss1.backward()        
        optimizer1.step()

        loss2.backward()
        optimizer2.step()

        sched1.step() #learning rate scheduler, you could tune as you wish
        sched2.step()
        
    
        total_loss += loss1.item()+loss2.item()
        if e<50:
            pred = pred1.detach() * 0.5 + pred2.detach() * 0.5
        else:
            pred = pred1.detach() * p_a1_loss.detach() + pred2.detach()*p_a2_loss.detach()
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    
    accuracy = correct_pred.cpu()/float(total_num)
    writer.add_scalar('loss/train', total_loss, e)
    writer.add_scalar('accuracy/train', accuracy, e)
    return total_loss/i, accuracy

def eval_seperate(e, eval_loader, device, agent1, agent2, writer):
    agent1.eval()
    agent2.eval()
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    i = 0
    for images, labels in eval_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
    
        pred1, bid1 = agent1(images)
        pred2, bid2 = agent2(images)
        if i==1:
            print(torch.mean(bid1), torch.mean(bid2))

        pred1 = F.softmax(pred1, dim=1)
        log_p1 = (torch.log(pred1+math.exp(-70)))
        one_hot = torch.zeros(pred1.shape,dtype=torch.double,device=device).scatter_(1, torch.unsqueeze(labels, dim=1),1)


        sigma = 0.1*torch.ones_like(bid1).to(device)*torch.sqrt(torch.tensor([2])).to(device)
        ones_t = torch.ones_like(bid1).to(device)
        bid_diff = bid1-bid2.detach().to(device)
        dis1_1 = Normal(bid_diff, sigma)
        dis1_2_fp = Normal(bid_diff+0.01*ones_t, sigma)
        dis1_2_sp = Normal(bid_diff-0.01*ones_t, sigma)
        p_a1_loss = dis1_1.cdf(torch.zeros_like(bid1).to(device))
        a_1 = torch.sum(one_hot*log_p1,dim=1).unsqueeze(1)
    
        fp_loss_1 = torch.exp(bid2.detach()+0.005)*(1-p_a1_loss)
        sp_loss_1 = torch.exp(bid1.to(device)+0.005)*(1-p_a1_loss)
        loss1 = -a_1 * p_a1_loss + sp_loss_1*0.9+ fp_loss_1*0.1 
        
        loss1 = loss1.sum()
        loss1/=pred1.shape[0]

        # loss2 = F.cross_entropy(pred2, labels)
        pred2 = F.softmax(pred2, dim=1) 
        log_p2 = (torch.log(pred2+math.exp(-70)))
        bid_diff = bid2-bid1.detach().to(device)
        dis2_1 = Normal(bid_diff, sigma)
        dis2_2_fp = Normal(bid_diff+0.01*ones_t, sigma) #fp means first price
        dis2_2_sp = Normal(bid_diff-0.01*ones_t, sigma)
        p_a2_loss = dis2_1.cdf(torch.zeros_like(bid2).to(device))
        a_2 = torch.sum(one_hot*log_p2,dim=1).unsqueeze(1)
        fp_loss_2 = torch.exp(bid2+0.005)*(1-p_a2_loss)
        sp_loss_2 = torch.exp(bid1.detach().to(device)+0.005)*(1-p_a2_loss)
        loss2 = -a_2 * p_a2_loss + sp_loss_2*0.9 + fp_loss_2*0.1
        loss2 = loss2.sum()
        loss2/=pred2.shape[0] 

        total_loss += loss1.item()+loss2.item()
        # probs = probs1.detach()
        if e<50:
            pred = pred1.detach() * 0.5 + pred2.detach() * 0.5
        else:
            pred = pred1.detach() * p_a1_loss.detach() + pred2.detach()*p_a2_loss.detach()
        
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    writer.add_scalar('loss/val', total_loss, e)
    writer.add_scalar('accuracy/val', accuracy, e)
    return total_loss/i, accuracy


def main(epochs, seed):
    global SEED
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    print(SEED)
    device = torch.device('cuda:0')
    train_ds = MyDataset(train_path)
    val_ds = MyDatasetval(val_path)
    writer = SummaryWriter()


    new_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128,
                                                shuffle=True, pin_memory=True, num_workers=12)

    validate_loader = torch.utils.data.DataLoader(val_ds, batch_size=128,
                                                shuffle=True, pin_memory=True, num_workers=12)

    agent1 = Resnet9(3,10)
    agent2 = Resnet9(3,10)
    # agent1 = ResNet18()
    # agent2 = ResNet18()
    agent1 = agent1.to(device)
    agent2 = agent2.to(device)

    max_lr = 1e-3
    #this resnet9 implementation may have some problems, if you use adam as the optimizer, then
    #the validation performance might suddenly drop to 0.1 at sometime.
    #if you want to use adam, please use ResNet18 or other models.
    optimizer1 = torch.optim.SGD(agent1.parameters(),lr=1e-4, weight_decay=1e-3)
    optimizer2 = torch.optim.SGD(agent2.parameters(),lr=1e-4, weight_decay=1e-3)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(optimizer1, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(new_train_loader))
    sched2 = torch.optim.lr_scheduler.OneCycleLR(optimizer2, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(new_train_loader))

    best_acc = 0.0
    for e in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_seperate(e, new_train_loader, device, agent1, agent2, optimizer1, optimizer2, sched1, sched2, writer)
        
        val_loss, val_acc = eval_seperate(e, validate_loader, device, agent1, agent2, writer)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | train_acc {:5.5f} | val_loss {:5.5f} | val_acc {:5.5f}'.format(
                        e, (time.time()-epoch_start_time), train_loss, train_acc, val_loss, val_acc))
        for param_group in optimizer1.param_groups:
            print(param_group['lr'])
        if not (e+1)%500 or val_acc>=best_acc:
            if val_acc>=best_acc:
                best_acc = val_acc
                model_pth = './model'
                if not os.path.exists(model_pth):
                    os.mkdir(model_pth)
                model1_pth = os.path.join(model_pth, f'cnnmodel1t{SEED}_sto_resnet_fixed_best.pt') #nl means no penalty loss
                model2_pth = os.path.join(model_pth, f'cnnmodel2t{SEED}_sto_resnet_fixed_best.pt')
                torch.save(agent1,model1_pth)
                torch.save(agent2,model2_pth)
                print("model saved")
            else:
                model_pth = './model'
                if not os.path.exists(model_pth):
                    os.mkdir(model_pth)
                model1_pth = os.path.join(model_pth, f'cnnmodel1t{SEED}_sto_know_{e}.pt') #nl means no penalty loss
                model2_pth = os.path.join(model_pth, f'cnnmodel2t{SEED}_sto_know_{e}.pt')
                torch.save(agent1,model1_pth)
                torch.save(agent2,model2_pth)
    print(best_acc)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def test(enum,noise):
    device = torch.device('cuda')
    pth1 = f'model/cnnmodel1t27_sto_{enum}.pt'
    pth2 = f'model/cnnmodel2t27_sto_{enum}.pt'
    test_agent1=torch.load(pth1)
    test_agent2=torch.load(pth2)
    test_agent1.eval()
    test_agent2.eval()
    test_img = 'datasets copy/train/cat.1.jpg'
    trans_func = transforms.Compose(
            [
                transforms.Resize(size = (224,224)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                # AddGaussianNoise(0., 1.)
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    img = Image.open(test_img)
    img = trans_func(img) + noise
    img = img.unsqueeze(0).to(device)
    # print(img.shape)
    pred1, bid1, _ = test_agent1(img)
    pred2, bid2, _ = test_agent2(img)
    pred1 = F.softmax(pred1)
    pred2 = F.softmax(pred2)
    print(pred1[0,1].detach().cpu(), pred2[0,1].detach().cpu(), torch.exp(bid1[0,0]).detach().cpu(), torch.exp(bid2[0,0]).detach().cpu())
    return pred1[0,1].detach().cpu(), pred2[0,1].detach().cpu(), torch.exp(bid1[0,0]).detach().cpu(), torch.exp(bid2[0,0]).detach().cpu()
#test2 is used to test the checkpoints and get the output for each datapoint 
def test2(e_num):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    device = torch.device('cuda')
    #this path points to the dataset you want to test
    #you could use val_path for validation set and train_path for training set
    train_ds = MyDatasetval(val_path)
    pth1 = f'model/cnnmodel1t{SEED}_sto_know_{e_num}.pt'
    pth2 = f'model/cnnmodel2t{SEED}_sto_know_{e_num}.pt'
    pth1 = f'model/cnnmodel1t{SEED}_sto_resnet_fixed_best.pt'
    pth2 = f'model/cnnmodel2t{SEED}_sto_resnet_fixed_best.pt'
    test_agent1=torch.load(pth1)
    test_agent2=torch.load(pth2)

    validate_loader = torch.utils.data.DataLoader(train_ds, batch_size=128,
                                                shuffle=False, pin_memory=True, num_workers=8)
    save_pth = f'50cnn_log{SEED}_sto_resnet_best.txt'
    test_write(save_pth, validate_loader,device,test_agent1,test_agent2)

# write the model output into a txt file, the output will be ordered by its class
def test_write(save_pth, eval_loader, device, agent1, agent2):
    agent1.eval()
    agent2.eval()
    correct_pred = 0
    total_num = 0

    i=0
    log=[]
    for images, labels in eval_loader:
        i+=1
        images = images.to(device)
        # if i<100:
        #     images += torch.randn(images.shape).cuda()
        labels = labels.to(device)
        pred1, bid1 = agent1(images)
        pred2, bid2 = agent2(images)

        pred1 = F.softmax(pred1, dim=1)
        pred2 = F.softmax(pred2, dim=1)

        one_hot = torch.zeros(pred1.shape,dtype=float,device='cuda').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        log_pred1= torch.log(pred1+math.exp(-70))
        log_pred2= torch.log(pred2+math.exp(-70))
        
        v1 = torch.sum(one_hot*log_pred1,dim=1).unsqueeze(1)
        v2 = torch.sum(one_hot*log_pred2,dim=1).unsqueeze(1)
        
        sigma = torch.ones_like(bid1).to(device)*torch.sqrt(torch.tensor([2])).to(device)
        
        dis1_1 = Normal(bid1-bid2.detach(), sigma)
        p_a1_loss = dis1_1.cdf(torch.zeros_like(bid1).to(device))
        dis2_1 = Normal(bid2-bid1.detach(), sigma)
        p_a2_loss = dis2_1.cdf(torch.zeros_like(bid2).to(device))

        pred = pred1*p_a1_loss + pred2*p_a2_loss
        pred = torch.argmax(pred, dim=1)
        tmp = torch.cat([v1,torch.exp(bid1),v2,torch.exp(bid2),labels.unsqueeze(-1),pred.unsqueeze(-1)],dim=1)
        # tmp = torch.cat([v1,(bid1),v2,(bid2),labels.unsqueeze(-1),pred.unsqueeze(-1)],dim=1)
        log.append(tmp.detach().cpu().numpy())
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    print(accuracy, SEED)
    log = np.vstack(log)
    np.savetxt(save_pth,log, fmt='%1.4e')

#this test function is for single agent baseline
def test2_sg(e_num):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    device = torch.device('cuda')
    train_ds = MyDatasetval(val_path)
    
    pth1 = f'model/cnnmodel1t{SEED}_sto_resnet_bl_best.pt'
    
    test_agent1=torch.load(pth1)

    validate_loader = torch.utils.data.DataLoader(train_ds, batch_size=512,
                                                shuffle=False, pin_memory=True, num_workers=8)
    save_pth = f'50cnn_log{SEED}_sto_resnet_bl_best.txt'
    test_write_sg(save_pth, validate_loader,device,test_agent1)

def test_write_sg(save_pth, eval_loader, device, agent1):
    agent1.eval()
    correct_pred = 0
    total_num = 0

    i=0
    log=[]
    for images, labels in eval_loader:
        i+=1
        images = images.to(device)
        # if i<100:
        #     images += torch.randn(images.shape).cuda()
        labels = labels.to(device)
        pred1 = agent1(images)

        pred1 = F.softmax(pred1, dim=1)

        one_hot = torch.zeros(pred1.shape,dtype=float,device='cuda').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        log_pred1= torch.log(pred1+math.exp(-70))
        
        # v1 = torch.sum(one_hot*log_pred1,dim=1).unsqueeze(1)
        v1,_ = torch.max(pred1,dim=1)
        v1 = v1.unsqueeze(-1)

        pred = pred1
        pred = torch.argmax(pred, dim=1)
        tmp = torch.cat([v1,labels.unsqueeze(-1),pred.unsqueeze(-1)],dim=1)
        # tmp = torch.cat([v1,(bid1),v2,(bid2),labels.unsqueeze(-1),pred.unsqueeze(-1)],dim=1)
        log.append(tmp.detach().cpu().numpy())
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    print(accuracy, SEED)
    log = np.vstack(log)
    np.savetxt(save_pth,log, fmt='%1.4e')
    
if __name__ == "__main__":
    seed = 49#27 
    #500 is the number of epochs
    main(500, seed) # normal 1500,0.7768; 0.7858
    # seeds = [0,18,25,26,28,42,53,102,114,115,132,172]
    global SEED
    SEED = seed
    # p1_l = []
    # p2_l = []
    # b1_l = []
    # b2_l = []
    # x = []
    # noise = torch.randn([3,224,224])*0.0#5
    # for i in range(20): #38
    #     p1,p2,b1,b2 = test(enum=i*500+499,noise=noise)
    #     x.append(i*500+499)
    #     p1_l.append(float(p1))
    #     p2_l.append(float(p2))
    #     b1_l.append(float(b1))
    #     b2_l.append(float(b2))
    # plt.plot(x,p1_l,label='Prediction of agent 1')
    # plt.plot(x,p2_l,label='Prediction of agent 2')
    # plt.plot(x,b1_l, marker='o', linestyle='dashed',label='Bid of agent 1')
    # plt.plot(x,b2_l, marker='o', linestyle='dashed',label='Bid of agent 2')
    # plt.xlabel('Checkpoint number')
    # plt.ylabel('Bid or Prediction')
    # plt.legend()
    # plt.show()
    
    

    test2(499)
    # for i in range(50):
    #     test2(5*i)



# 
# for i, item in enumerate(tqdm(train_ds)):
#     # pass
#     print(item[0].shape)
#     break
# img_PIL_Tensor = train_ds[1][0]
# new_img_PIL = transforms.ToPILImage()(img_PIL_Tensor).convert('RGB')
# plt.imshow(new_img_PIL)
# plt.show()


#0.90720