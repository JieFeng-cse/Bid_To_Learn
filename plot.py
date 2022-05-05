import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import numpy as np
from scipy import stats
from matplotlib  import cm
from scipy import stats
plt.style.use('seaborn-whitegrid')
# fig = plt.figure()
# ax = Axes3D(fig)
# seed = 73
# pth = f'50cnn_log{seed}.txt'
X1=[]
Y1=[]
X2=[]
Y2=[]
X3=[]
Y3=[]
X4=[]
Y4=[]
sum_b=0
sum_agent2 = 0
# # # seed_set = set([0,2,4,13,18,25,26,28,42,53,73,82,102,114,115,132,139,150,154,168,172,173,186])
# # # seed_set = [4,13,73,82,122,124,139,150,154,173,186]
seed_set = [27]
for seed in seed_set:
    # pth = f'log/90_{seed}cnn_log.txt'
    # pth = f'50cnn_log{seed}10000.txt'
    pth = '50cnn_log49_sto_resnet_best.txt'
    with open(pth,'r') as f:
        lines = f.readlines()
        # print(len(lines))
        for line in lines:
            val = [float(s) for s in line.split()] # 5, v1,b1,v2,b2,gt
            if val[1]<val[3]:
                sum_b += val[0]>val[2]
                # sum_b += val[4] == val[5]
                sum_agent2 += 1
            
            # if val[0] > val[2]: # b1>b2,v1>v2
            #     X1.append(val[1])
            #     Y1.append(val[3])
            X1.append(val[1])
            Y1.append(val[3])
            X2.append(val[0])
            Y2.append(val[2])
            # elif val[0] < val[2]: # b1>b2,v1<v2
            #     X2.append(val[1])
            #     Y2.append(val[3])    
            # X1.append(abs(val[1]-val[3]))
            # Y1.append(abs(val[0]-val[2]))
            # X2.append(val[1])
            # Y2.append(val[0])
    # break
# plt.plot([0,0.1,0.12],[0,0.1,0.12])
print(sum_b, sum_agent2)
z = np.asarray(X2)-np.asarray(Y2)
z[z<-0.1] = -0.1
z[z>0.1] = 0.1
plt.scatter(X1,Y1,alpha=0.3,s=20,c=z,label='v1>v2', cmap = cm.jet)
plt.colorbar(cmap = cm.jet)
# plt.scatter(X1,Y1,alpha=0.3,c='#1f77b4',label='v1>v2')
# plt.scatter(X2,Y2,alpha=0.3,c='#ff7f0e',label='v1<v2')

# plt.scatter(X1,Y1,alpha=0.3,label='v1>v2')
# # # plt.scatter(X3,Y3,alpha=0.3,label='b1<b2,v1>v2')
# # # plt.scatter(X4,Y4,alpha=0.3,label='b1<b2,v1<v2')
plt.xlabel('Bid 1')
plt.ylabel('Bid 2')
plt.legend()
# plt.axis([0, 0.13, 0, 0.13])
plt.show()

# pth = '50cnn_log49_sto_resnet_best.txt'
X1=[]
Y1=[]
X2=[]
Y2=[]
sum_b=0
sum_agent2 = 0
i = 0
with open(pth,'r') as f:
    lines = f.readlines()
    for line in lines:
        i += 1
        # if i< 10000:
        #     continue
        val = [float(s) for s in line.split()] # 5, v1,b1,v2,b2,gt
        if val[0] > -1:
            sum_b +=1
        if val[2] > -1:
            sum_agent2 +=1
        # if val[1]>val[3]:
        #     # print(val[1],val[3])
        #     sum_b += val[0]<val[2]
        #     sum_agent2 += 1
            
        X1.append(val[0])
        Y1.append(val[1])
        X2.append(val[2])
        Y2.append(val[3])
print(sum_b, sum_agent2)
print(len(X1))
plt.scatter(X2,Y2,alpha=0.6,label='agent2')
plt.scatter(X1,Y1,alpha=0.6,label='agent1')

# plt.axis([-1, 0, 0,0.15])
plt.xlabel(f'Evaluation{0}')
plt.ylabel('Bid')
plt.legend()
plt.show()

X1=[]
Y1=[]
X2=[]
Y2=[]
X3=[]
Y3=[]
X4=[]
Y4=[]
sum_b=0
sum_agent2 = 0
i=0
for seed in seed_set:
    pth = '50cnn_log49_sto_resnet_best.txt'
    with open(pth,'r') as f:
        lines = f.readlines()
        for line in lines:
            i += 1
            # if i<10000:
            #     continue
            val = [float(s) for s in line.split()] # 5, v1,b1,v2,b2,gt
            # if val[1]>val[3]:
            #     sum_b += val[0]<val[2]
            #     sum_agent2 += 1
            
            X1.append(val[0])
            Y1.append(val[2])
            
            # X2.append(val[0])
            # Y2.append(val[2])
    # break
# plt.plot([0,0.1,0.12],[0,0.1,0.12])
print(sum_b, sum_agent2)
sum_10k_1=np.zeros((10,1))
sum_10k_2=np.zeros((10,1))
for i in range(10):
    sum_10k_1[i]=np.mean(X1[i*1000:(i+1)*1000])
    sum_10k_2[i]=np.mean(Y1[i*1000:(i+1)*1000])
print(len(X1))
for i in range(len(X1)):
    X1[i] = sum_10k_1[i//1000]
    Y1[i] = sum_10k_2[i//1000]
plt.scatter(range(len(X1)),X1,alpha=0.3,c='#1f77b4',label='agent1')
plt.scatter(range(len(X1)),Y1,alpha=0.3,c='#ff7f0e',label='agent2')
# plt.scatter(range(10),sum_10k_1,alpha=0.3,c='#1f77b4',label='agent1')
# plt.scatter(range(10),sum_10k_2,alpha=0.3,c='#ff7f0e',label='agent2')

# plt.scatter(X1,Y1,alpha=0.3,label='v1>v2')
# # # plt.scatter(X3,Y3,alpha=0.3,label='b1<b2,v1>v2')
# # # plt.scatter(X4,Y4,alpha=0.3,label='b1<b2,v1<v2')
plt.xlabel('Sample id')
# plt.ylabel('Largest Prob')
# plt.ylabel('Accuracy')
plt.ylabel('log(prob)')
# plt.ylabel('bid')
plt.legend()
# plt.axis([0, 10000, 0, 1])
plt.show()