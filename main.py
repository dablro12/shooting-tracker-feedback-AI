#!pip install tabulate #가독성 print
#!pip install -q mediapipe==0.10.0
from tabulate import tabulate
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#densenet model setting
import torchvision.models as models
from torchvision.models import densenet, DenseNet121_Weights
#Resnet model setting
from torchvision.models import resnet
#ConfusionMatrixDisplay로 표현하기 위해 사이킷런 import
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
import torchvision.models
#pose standard module
import pandas as pd
import matplotlib.image as img
import seaborn as sns  #시각화 
import mediapipe as mp
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#CNN visualization module
from PIL import Image
#angle module
import math
#Regression model import
import random

#################################################### data naming change####################################################
data_root = 'data/' 
logdir_path = os.path.normpath('result')
if os.path.exists(logdir_path) == False:
  os.mkdir(logdir_path)
  
#이름바꾸기 형식에 맞게 
def rename(data_root):
  tmp = glob(data_root+"*/*.jpg")
  for i in tmp:
    string = ""
    tmp_li = (i.split('/')[-1].split('-'))
    string = tmp_li[0] + '_' + tmp_li[1]
    #print(string)
    #print(i)
    #### 데이터 더 많이 넣어서 해보기 !!!!
    #print(data_path = i.split('/'))
    #os.rename(i, data_path + string)

#여러 확장자를 jpg로 형식바꾸기 
def any2jpg(data_root):
  tmp = glob(data_root+"*/"+"*/*")
  #print(tmp)
  for img in tmp:
    protocol = (img.split('.')[-1])
    if protocol != 'jpg':
      #print(img)
      rename = img.split(protocol)[0] + "jpg"
      #print(rename)
      os.rename(img, rename)
#any2jpg(data_root)


####################################################data mask preprocessing with bidirect filtering####################################################d
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
image_li = []
df = pd.DataFrame()
pj_path = 'mp'
save_PATH = os.path.join('data','mask')
dataset = glob('data/test/*/*') + glob('data/train/*/*') + glob('data/valid/*/*')
print(type(dataset))
print(f"dataset 개수 : {len(dataset)}")
###########setting############
contrast = 0.8
brightness = 1

###########
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode = True,
    enable_segmentation=True,
) as pose:
    for data in dataset:
        #배경 제거
        image = cv2.imread(data)
        image_li.append(image)
        ## 대조 밝기 변경 : 특정 데이터들떄문
        image = cv2.convertScaleAbs(image, alpha = contrast, beta = brightness)
        h, w, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        annotated_image = image.copy()
        #사람만 추출
        ### 공동 양방향 필터사용 * https://learn.foundry.com/ko/nuke/content/comp_environment/filters/bilateral.html
        condition = np.stack((results.segmentation_mask,) * 3, axis = -1) 
        bg_image = np.zeros(image.shape, dtype = np.uint8)
        bg_image[:] = (192,192,192) #배경 회색으로 처리       
        annotated_image = np.where(condition, annotated_image, bg_image)
        x = []
        #image.flags.writeable = False
        #print(data)
        
        data_path = os.path.join(save_PATH, data.split('/')[-3],data.split('/')[-2],data.split('/')[-1])
        cv2.imwrite(data_path, annotated_image)
        

############## data load ##############
#데이터 개수 확인
train_O_len = len(os.listdir('data/train/O/'))
train_X_len = len(os.listdir('data/train/X/'))
valid_O_len = len(os.listdir('data/valid/O'))
valid_X_len = len(os.listdir('data/valid/X'))
test_O_len = len(os.listdir('data/test/O/'))
test_X_len = len(os.listdir('data/test/X/'))
print("---------------Train------------")
print(train_O_len)
print(train_X_len)
print("---------------VALID------------")
print(len(os.listdir('data/valid/O')))
print(len(os.listdir('data/valid/X')))
print("---------------Test------------")
print(test_O_len)
print(test_X_len)
print('---------------TOTAL------------')
print(train_O_len+train_X_len+valid_O_len+valid_X_len+test_O_len+test_X_len)
#데이터 shpae 알아보기
root = glob('./data/*/*/*')
root
for i in root:
  img = cv2.imread(os.path.join(i))
  #print(img.shape)
  break

############################## data transform #######################################
#data transform
train_transforms = transforms.Compose(
  [
    transforms.RandomRotation(degrees=(0,15)),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.ToTensor()
  ]
)
valid_transforms = transforms.Compose(
  [
    transforms.Resize((256, 256)),
    transforms.ToTensor()
  ]
)

test_transforms = transforms.Compose(
  [
    transforms.Resize((256, 256)),
    transforms.ToTensor()
  ]
)
train_batch_size = 6
valid_batch_size = 4

#prefix setting 
train_path = 'data/train/'
valid_path = 'data/valid/'
test_path = 'data/test/'

#data check 
check_trainset = torchvision.datasets.ImageFolder(root = train_path, transform = train_transforms)
check_trainloader = torch.utils.data.DataLoader(check_trainset, batch_size = train_batch_size, shuffle = True)
check_validset = torchvision.datasets.ImageFolder(root = valid_path, transform = valid_transforms)
check_validloader = torch.utils.data.DataLoader(check_validset, batch_size = valid_batch_size, shuffle = True)
testset = torchvision.datasets.ImageFolder(root = test_path, transform = test_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size = test_O_len+test_X_len, shuffle = False)
#data check
for X, y in check_trainloader:
  print(X.shape, y.shape)
  f = X[0][0].numpy()
  plt.imshow(f, cmap='gray')
  plt.show()
  break
  
for X, y in check_validloader:
  print(X.shape, y.shape)
  f = X[0][0].numpy()
  plt.figure()
  plt.imshow(f, cmap='gray')
  plt.show()
  break
  
for X, y in testloader:
  print(X.shape, y.shape)
  f = X[0][0].numpy()
  plt.figure()
  plt.imshow(f, cmap='gray')
  plt.show()
  break
############################## CNN Model 검증(densenet , resnet) #######################################
############################## 1. densenet #######################################
model = densenet.DenseNet()
model.features.conv0 = torch.nn.Conv2d(in_channels= 3,  out_channels=64, kernel_size=7)
model.classifier = torch.nn.Linear(in_features= 1024, out_features= 2, bias= True)
#loss, accuracy plot
target_cls =check_trainset.classes 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
print(model)

############################## denset training #######################################
#result check
loss_train = np.array([])
accs_train = np.array([])
accs_valid = np.array([])

#data load
trainset = torchvision.datasets.ImageFolder(root = train_path, transform = train_transforms)
validset = torchvision.datasets.ImageFolder(root = valid_path, transform = valid_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch_size, shuffle = True)
validloader = torch.utils.data.DataLoader(validset, batch_size = valid_batch_size, shuffle = True)
#hyper parma setting
lr = 42e-4
num_epochs = 500
# loss, algo
loss =torch.nn.CrossEntropyLoss() #CE Loss
alg = torch.optim.Adam(model.parameters(), lr = lr) #adam
for epoch in range(num_epochs):
    i=0
    l_epoch = 0
    correct = 0
    model.train()
    for X,y in trainloader:
        i=i+1
        X,y = X.to(device),y.to(device)
        y_hat=model(X)
        correct += (y_hat.argmax(dim=1)==y).sum()
        l=loss(y_hat,y)
        l_epoch+=l
        alg.zero_grad()
        l.backward()
        alg.step()

    loss_train = np.append(loss_train,l_epoch.cpu().detach().numpy()/i)
    accs_train = np.append(accs_train,correct.cpu()/len(trainset))

    correct = 0
    model.eval()
    for X,y in validloader:
        X,y = X.to(device),y.to(device)
        y_hat = model(X)
        correct += (y_hat.argmax(dim=1)==y).sum()

    accs_valid = np.append(accs_valid,correct.cpu()/len(validset))



    if epoch%5 == 0:
        print('epoch: %d '%(epoch))
        print('train loss: ',loss_train[-1])
        print('train accuracy: ',accs_train[-1])
        print('valid accuracy: ',accs_valid[-1])
        plt.figure(2,dpi=80)
        plt.subplot(121)
        plt.plot(loss_train,label='train loss')
        plt.legend(loc='upper right')
        plt.subplot(122)
        plt.plot(accs_train,label='train accuracy')
        plt.plot(accs_valid,label='valid accuracy')
        plt.legend(loc='upper left')
        plt.title('epoch: %d '%(epoch))
        plt.savefig('result/cnn_model/densenet/Densenet_loss.png')
        #plt.show()
        plt.close(2)
        #model save

    if accs_valid[-1] >= 0.98 and accs_train[-1] >= 0.98 :
        torch.save(   
            model.state_dict(), os.path.join( f"./model/final_densenet.pth")
        )
        break

# loss check
plt.figure(2,dpi=80)
plt.subplot(121)
plt.plot(loss_train,label='train loss')
plt.legend(loc='upper right')
plt.subplot(122)
plt.plot(accs_train,label='train accuracy')
plt.plot(accs_valid,label='valid accuracy')
plt.legend(loc='lower right')
plt.savefig('./model/loss/densenet_loss.jpg')
plt.show()

N = 0
I = Image.open(validset.imgs[N][0])
X = train_transforms(I)
y = validset.targets[N]

print(target_cls[y])
y_hat = model(X.unsqueeze(0).to(device))
print(y_hat.cpu().detach().numpy())
y_hat = y_hat.argmax(dim=1)
print(f'prediction of model: {target_cls[y_hat.cpu().numpy()[0]]}')
############################## denset model visualization #######################################
#summary validation set
y_list = np.array([])
y_hat_list = np.array([])
for X,y in testloader:
  y_hat = model(X.to(device))    
  y_hat = y_hat.argmax(dim=1)
  y_list = np.append(y_list,y)
  y_hat_list = np.append(y_hat_list,y_hat.cpu().numpy())

#print(len(y_hat_list))

print(classification_report(
    y_list,
    y_hat_list,
    target_names=target_cls))

#summary confusion matrix 
cm = confusion_matrix(
    y_list,
    y_hat_list,
    #normalize='true',
)

#ConfusionMatirxDisplay(confusion_matrx = {confusion_matrix var}, display_labels = {결과 class})
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_cls,
)
disp.plot(ax=plt.subplots(1, 1, facecolor='white')[1])

total= test_O_len+test_X_len
###valid accuracy 측정 
TT = cm[0][0] + cm[1][1]
test_prediction = 100 * (TT / total)
print(f"test data 총 {total}명 중 맞춘 확률 : {test_prediction}%")  


############################## 2. resnet  #######################################
#finetuning
resnet_model = torchvision.models.resnet101()
resnet_model.fc = torch.nn.Linear(in_features =2048, out_features=2, bias = True)
print(resnet_model)

#hyper pram, loss. alg 정의
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet_model.to(device)

#hyper parma setting
lr = 42e-4
num_epochs = 500
# loss, algo
loss =torch.nn.CrossEntropyLoss() #CE Loss
alg = torch.optim.Adam(resnet_model.parameters(), lr = lr) #adam

#train
#result check
loss_train = np.array([])
accs_train = np.array([])
accs_valid = np.array([])

#data load
trainset = torchvision.datasets.ImageFolder(root = train_path, transform = train_transforms)
validset = torchvision.datasets.ImageFolder(root = valid_path, transform = valid_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch_size, shuffle = True)
validloader = torch.utils.data.DataLoader(validset, batch_size = valid_batch_size, shuffle = True)

for epoch in range(num_epochs):
    i=0
    l_epoch = 0
    correct = 0
    resnet_model.train()
    for X,y in trainloader:
        i=i+1
        X,y = X.to(device),y.to(device)
        y_hat=resnet_model(X)
        correct += (y_hat.argmax(dim=1)==y).sum()
        l=loss(y_hat,y)
        l_epoch+=l
        alg.zero_grad()
        l.backward()
        alg.step()

    loss_train = np.append(loss_train,l_epoch.cpu().detach().numpy()/i)
    accs_train = np.append(accs_train,correct.cpu()/len(trainset))

    correct = 0
    resnet_model.eval()
    for X,y in validloader:
        X,y = X.to(device),y.to(device)
        y_hat = resnet_model(X)
        correct += (y_hat.argmax(dim=1)==y).sum()

    accs_valid = np.append(accs_valid,correct.cpu()/len(validset))



    if epoch%5 == 0:
        print('epoch: %d '%(epoch))
        print('train loss: ',loss_train[-1])
        print('train accuracy: ',accs_train[-1])
        print('valid accuracy: ',accs_valid[-1])
        plt.figure(2,dpi=80)
        plt.subplot(121)
        plt.plot(loss_train,label='train loss')
        plt.legend(loc='upper right')
        plt.subplot(122)
        plt.plot(accs_train,label='train accuracy')
        plt.plot(accs_valid,label='valid accuracy')
        plt.legend(loc='upper left')
        plt.title('epoch: %d '%(epoch))
        plt.savefig('./result/cnn_model/resnet_loss_curve.png')
        #plt.show()
        plt.close(2)
        
#resnet_model save        
    if accs_valid[-1] >= 0.98 and accs_train[-1] >= 0.98 :
        torch.save(   
            resnet_model.state_dict(), os.path.join( f"./result/final_resnet.pth")
        )
        break


#loss, accuracy plot
target_cls =trainset.classes 
target_cls

#result check
plt.figure(2,dpi=80)
plt.subplot(121)
plt.plot(loss_train,label='train loss')
plt.legend(loc='upper right')
plt.subplot(122)
plt.plot(accs_train,label='train accuracy')
plt.plot(accs_valid,label='valid accuracy')
plt.legend(loc='lower right')
plt.savefig('./model/loss/resnet_loss.jpg')
plt.show()

N = 0
I = Image.open(validset.imgs[N][0])
X = train_transforms(I)
y = validset.targets[N]

#print(target_cls[y])
y_hat = resnet_model(X.unsqueeze(0).to(device))
#print(y_hat.cpu().detach().numpy())
y_hat = y_hat.argmax(dim=1)
print(f'prediction of resnet_model: {target_cls[y_hat.cpu().numpy()[0]]}')

#visualization
#summary validation set
y_list = np.array([])
y_hat_list = np.array([])
resnet_model.eval()
for X,y in testloader:
  y_hat = resnet_model(X.to(device))    
  y_hat = y_hat.argmax(dim=1)
  y_list = np.append(y_list,y)
  y_hat_list = np.append(y_hat_list,y_hat.cpu().numpy())

#print(len(y_hat_list))

print(classification_report(
    y_list,
    y_hat_list,
    target_names=target_cls))

#summary confusion matrix 
cm = confusion_matrix(
    y_list,
    y_hat_list,
    #normalize='true',
)

#ConfusionMatirxDisplay(confusion_matrx = {confusion_matrix var}, display_labels = {결과 class})
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_cls,
)
disp.plot(ax=plt.subplots(1, 1, facecolor='white')[1])

total= test_O_len+test_X_len
###valid accuracy 측정 
TT = cm[0][0] + cm[1][1]
test_prediction = 100 * (TT / total)
print(f"test data 총 {total}명 중 맞춘 확률 : {test_prediction}%")  

############################## Test #######################################
#desnenet testing
### 저장한 모델.pth 를 지정해줘야함
test_model_name = './model/final_densenet.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
#model load
test_model = densenet.DenseNet()
test_model.features.conv0 = torch.nn.Conv2d(in_channels= 3,  out_channels=64, kernel_size=7)
test_model.classifier = torch.nn.Linear(in_features= 1024, out_features= 2, bias= True)
test_model.load_state_dict(torch.load(os.path.join(test_model_name)))
print(test_model)
#summary validation set
y_list = np.array([])
y_hat_list = np.array([])

for X,y in testloader:
  y_hat = test_model(X)    
  y_hat = y_hat.argmax(dim=1)
  y_list = np.append(y_list,y)
  y_hat_list = np.append(y_hat_list,y_hat.cpu().numpy())

#print(len(y_hat_list))
#ConfusionMatrixDisplay로 표현하기 위해 사이킷런 import
#print(classification_report(
#    y_list,
#    y_hat_list,
#    target_names=target_cls))

print(classification_report(
    y_list,
    y_hat_list,
    target_names=target_cls))

#summary confusion matrix 
#cm = confusion_matrix(
#    y_list,
#    y_hat_list,
#    #normalize='true',
#)


#ConfusionMatirxDisplay(confusion_matrx = {confusion_matrix var}, display_labels = {결과 class})
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_cls,
)
disp.plot(ax=plt.subplots(1, 1, facecolor='white')[1])

total= test_X_len+test_O_len
###valid accuracy 측정 
TT = cm[0][0] + cm[1][1]
test_prediction = 100 * (TT / total)
print(f"test data 총 {total}명 중 맞춘 확률 : {test_prediction}%")  

### 저장한 모델.pth 를 지정해줘야함
test_model_name = './model/final_resnet.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
#model load
resnet_model = torchvision.models.resnet101()
resnet_model.fc = torch.nn.Linear(in_features =2048, out_features=2, bias = True)
resnet_model.load_state_dict(torch.load(test_model_name).features)
print(resnet_model)
#summary validation set
y_list = np.array([])
y_hat_list = np.array([])

for X,y in testloader:
  y_hat = test_model(X)    
  y_hat = y_hat.argmax(dim=1)
  y_list = np.append(y_list,y)
  y_hat_list = np.append(y_hat_list,y_hat.cpu().numpy())

#print(len(y_hat_list))
#ConfusionMatrixDisplay로 표현하기 위해 사이킷런 import
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

#print(classification_report(
#    y_list,
#    y_hat_list,
#    target_names=target_cls))

#summary confusion matrix 
cm = confusion_matrix(
    y_list,
    y_hat_list,
    #normalize='true',
)


#ConfusionMatirxDisplay(confusion_matrx = {confusion_matrix var}, display_labels = {결과 class})
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_cls,
)
disp.plot(ax=plt.subplots(1, 1, facecolor='white')[1])

total= test_O_len + test_X_len
###valid accuracy 측정 
TT = cm[0][0] + cm[1][1]
test_prediction = 100 * (TT / total)
print(f"test data 총 {total}명 중 맞춘 확률 : {test_prediction}%")  
class calculator:
  def __init__(self, array, width, height):
    x1 = 0
    x2 = 0
    x3 = 0
    y1 = 0
    y2 = 0
    y3 = 0
    self.array = array
    self.width = width
    self.height = height
  def angle1(self):
    self.x1, self.y1 = self.array[12]
    self.x2, self.y2 = self.array[14]
    self.x3, self.y3 = [self.array[12][0],self.height]
    
    try:
      a1 = (self.y2 - self.y1) / (self.x2 - self.x1)
    except ZeroDivisionError:
      a1 = (self.y2 - self.y1) / 1
    a2 = (self.y3 - self.y1) / 1.0e-10
    try:
      a3 = (self.y3 - self.y2) / (self.x3 - self.x2)
    except ZeroDivisionError:
      a3 = (self.y3 - self.y2) / 1
      
    angle_rad1 = math.atan(abs((a1 - a3) / (1 + a1 * a3)))
    angle1 = math.degrees(angle_rad1)
    angle_rad2 = math.atan(abs((a3)))
    angle2 = 90 - math.degrees(angle_rad2)
    #print(angle1, angle2)
    return 180 - angle1 - angle2
  def angle2(self):
    angle = 0 #ball
    return float(angle)
  def angle3(self):
    self.x1, self.y1 = self.array[23]
    self.x2, self.y2 = self.array[25]
    self.x3, self.y3 = self.array[27]
    
    try:
      a1 = (self.y2 - self.y1) / (self.x2 - self.x1)
    except ZeroDivisionError:
      a1 = (self.y2 - self.y1) / 1
    try: 
      a2 = (self.y3 - self.y2) / (self.x3 - self.x2)
    except ZeroDivisionError:
      a2 = (self.y3 - self.y2) / 1
    
    angle_rad = math.atan(abs((a1 - a2) / (1 + a1 * a2)))
    angle = math.degrees(angle_rad)
    return angle
  def angle4(self):
    self.x1, self.y1 = self.array[25]
    self.x2, self.y2 = self.array[27]
    self.x3, self.y3 = self.array[31]
    
    try:
      a1 = (self.y2 - self.y1) / (self.x2 - self.x1)
    except ZeroDivisionError:
      a1 = (self.y2 - self.y1) / 1
    try: 
      a2 = (self.y3 - self.y2) / (self.x3 - self.x2)
    except ZeroDivisionError:
      a2 = (self.y3 - self.y2) / 1
    
    angle_rad = math.atan(abs((a1 - a2) / (1 + a1 * a2)))
    angle = math.degrees(angle_rad)
    return 180 - angle
  def angle5(self):
    self.x1, self.y1 = self.array[26]
    self.x2, self.y2 = self.array[30]
    self.x3, self.y3 = [self.array[26][0],self.array[30][1]]
    
    try:
      a1 = (self.y2 - self.y1) / (self.x2 - self.x1)
    except ZeroDivisionError:
      a1 = (self.y2 - self.y1) / 1
    try: 
      a2 = (self.y3 - self.y2) / (self.x3 - self.x2)
    except ZeroDivisionError:
      a2 = (self.y3 - self.y2) / 1
    
    angle_rad = math.atan(abs((a1 - a2) / (1 + a1 * a2)))
    angle = math.degrees(angle_rad)
    return angle
  def angle6(self):
    self.x1, self.y1 = self.array[12]
    self.x2, self.y2 = self.array[24]
    self.x3, self.y3 = [self.array[12][0],self.array[24][1]]
    
    try:
      a1 = (self.y2 - self.y1) / (self.x2 - self.x1)
    except ZeroDivisionError:
      a1 = (self.y2 - self.y1) / 1
    try:
      a2 = (self.y3 - self.y2) / (self.x3 - self.x2)
    except ZeroDivisionError:
      a2 = (self.y3 - self.y2) / 1
    
    angle_rad = math.atan(abs((a1 - a2) / (1 + a1 * a2)))
    angle = math.degrees(angle_rad)
    return angle
  def values(self):
    a1 = self.angle1()
    a2 = self.angle2()
    a3 = self.angle3()
    a4 = self.angle4()
    a5 = self.angle5()
    a6 = self.angle6()
    ary = []
    ary.append(a1)
    ary.append(a2)
    ary.append(a3)
    ary.append(a4)
    ary.append(a5)
    ary.append(a6)
    return ary
  
  pj_path = 'mp'
save_PATH = os.path.join('result')
dataset = glob('data/test/O/*')+ glob('data/train/O/*') + glob('data/valid/O/*')
#print(dataset)
image_li = []
for data in dataset:
  #img = cv2.GaussianBlur(raw_img, (11, 11), 0)  # blur
  img = cv2.imread(data)
  #define param :) contrast = 0~127, brightness : 0~100
  contrast =0.8
  brightness = 1
  img = cv2.convertScaleAbs(img, alpha= contrast, beta = brightness)
  
  #img = cv2.equalizeHist(img, dst=None)
  h, w, _ = img.shape
  #img = center_crop(img, 800)
  image_li.append(img)
  
print(image_li[0].shape)
print(len(image_li))
#plt.imshow(image_li[0])

i=1
for img in image_li:
  plt.imshow(img)
  break
#!rm -rf result/{test,train,valid}/{O,X}/{mask,pose}/*
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pj_path = 'mp'
save_PATH = os.path.join('result')
#data_class = ['valid']
data_class = ['train', 'valid', 'test']
classes = ['O', 'X']

print(data_class)
        ###########
with mp_pose.Pose(
    static_image_mode = True,
    model_complexity = 2,
    enable_segmentation=True,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
) as pose:
    for datatype in data_class:
        for class_one in classes:
            image_li = []
            df = pd.DataFrame()
            dataset = glob(f'data/{datatype}/{class_one}/*')
            #print(dataset)
            #print(dataset)
            ###########setting############
            contrast =0.7
            brightness = 1.5
            for data in dataset:
                #배경 제거
                #print(data)
                image = cv2.imread(data)
                image_li.append(image)
                ## 대조 밝기 변경 : 특정 데이터들떄문
                image = cv2.convertScaleAbs(image, alpha = contrast, beta = brightness)
                #print(image)
                h, w, _ = image.shape
                #print(h, w, _)
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                #없어도해
                if not results.pose_landmarks:
                   continue
                annotated_image = image.copy()
                #사람만 추출
                ### 공동 양방향 필터사용 * https://learn.foundry.com/ko/nuke/content/comp_environment/filters/bilateral.html
                
                condition = np.stack((results.segmentation_mask,) * 3, axis = -1) >0.1
                
                bg_image = np.zeros(image.shape, dtype = np.uint8)
                bg_image[:] = (192,192,192) #배경 회색으로 처리       
                annotated_image = np.where(condition, annotated_image, bg_image)
                x = []
                image.flags.writeable = False
                #print(data)
                for k in range(33):
                    if results.pose_landmarks:
                        x.append(results.pose_landmarks.landmark[k].x)
                        x.append(results.pose_landmarks.landmark[k].y)
                        x.append(results.pose_landmarks.landmark[k].z)
                        x.append(results.pose_landmarks.landmark[k].visibility)
                
                # list x를 dataframe으로 변경
                tmp = pd.DataFrame(x)
                # dataframe에 정보 쌓아주기
                # 33개 landmarks의 132개 정보가 dataframe에 담긴다.
                df = pd.concat([df, tmp])
                
                #이미지 위에 pose landmark 그리기  
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                image_path = f'result/{datatype}/{class_one}/pose/' + str(data.split('/')[-1])
                cv2.imwrite(image_path, image)
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                anno_image_path = f'result/{datatype}/{class_one}/mask/' + str(data.split('/')[-1])
                #print(anno_image_path)
                cv2.imwrite(anno_image_path, annotated_image)
                
            #data angle cal
            if not results.pose_landmarks:
                    continue
            res = results.pose_landmarks
            land_list = [[landmark.x, landmark.y]for landmark in res.landmark]
            land_dict = {}
            for idx in range(len(image_li)):
                width, height, _ = image_li[idx].shape
                scaled_list = []
                filename = str(dataset[idx])
                land_dict[filename] = []
                for x, y in land_list:
                    x = round(x * width)
                    y = round(y * height)
                    land_dict[filename].append([x,y])

            save_df = pd.DataFrame(columns=['filepath', 'std1','std2','std3','std4','std5','std6'])
            for key, val in land_dict.items():
                cal = calculator(val, width = width, height = height)
                angle = cal.values()
                #print(angle)
                save_df = save_df.append(
                    pd.DataFrame([[key, angle[0], angle[1], angle[2], angle[3], angle[4], angle[5]]],
                                columns=['filepath','std1','std2','std3','std4','std5','std6']),
                    ignore_index=True)
            print(f"data name : {datatype}/{class_one}, len : {len(dataset)}")
            print(tabulate(save_df, headers='keys', tablefmt='psql', showindex=True))
            save_df.to_csv(f"./result/csv/{datatype}_{class_one}_pose_data.csv", mode='w')  

### warning

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./result/csv/total_X_pose_data.csv')
#제일잘한걸 ground truth로 할거
gt_df = pd.concat([
  #pd.read_csv('./result/csv/train_O_pose_data.csv').loc[{1,3}],  #2,13
  #pd.read_csv('./result/csv/valid_O_pose_data.csv').loc[{1,3}], #1, 11
  #pd.read_csv('./result/csv/valid_O_pose_data.csv').loc[{1,3}], #1, 11
  pd.read_csv('./result/csv/total_X_pose_data.csv'), #1, 11
  #pd.read_csv('./result/csv/train_X_pose_data.csv'), #1, 11
  #pd.read_csv('./result/csv/test_X_pose_data.csv'), #1, 11
  ]
)
#ff에는 train모든 것들 다들어감 
train_df = pd.concat([
  pd.read_csv('./result/csv/train_O_pose_data.csv'),
  pd.read_csv('./result/csv/train_X_pose_data.csv'),
  ])
#test 
test_df = pd.concat([
  pd.read_csv('./result/csv/test_O_pose_data.csv'),
  pd.read_csv('./result/csv/test_X_pose_data.csv'),
  ])
valid_df = pd.concat([
  pd.read_csv('./result/csv/valid_O_pose_data.csv'),
  pd.read_csv('./result/csv/valid_X_pose_data.csv'),
])

O_df = pd.concat([
  pd.read_csv('./result//csv/train_O_pose_data.csv'),
  pd.read_csv('./result//csv/valid_O_pose_data.csv'),
])

X_df = pd.concat([
  pd.read_csv('./result//csv/train_X_pose_data.csv'),
  pd.read_csv('./result//csv/valid_X_pose_data.csv'),
])


gt_df.shape, ff_df.shape, test_df.shape, gt_df.shape[0]+ff_df.shape[0]+test_df.shape[0]
gt_df.describe()
ff_df.describe()
gt_df.iloc[0:]
heatmap_df = gt_df[['std1','std2','std3','std4','std5','std6']]
sns.heatmap(heatmap_df)

sns.distplot(gt_df['std1'], color ='blue', label = 'O')
sns.distplot(ff_df['std1'], color ='red', label = 'X')
plt.legend(title='O/X')
#plt.savefig('./result/plot/scoring_std1.png')
plt.show()

sns.distplot(gt_df['std3'], color ='blue', label = 'O')
sns.distplot(ff_df['std3'], color ='red', label = 'X')
plt.legend(title='O/X')
#plt.savefig('./result/plot/scoring_std3.png')
plt.show()

sns.distplot(gt_df['std4'], color ='blue', label = 'O')
sns.distplot(ff_df['std4'], color ='red', label = 'X')
plt.legend(title='O/X')
#plt.savefig('./result/plot/std4.png')
plt.show()

sns.distplot(gt_df['std5'], color ='blue', label = 'O')
sns.distplot(ff_df['std5'], color ='red', label = 'X')
plt.legend(title='O/X')
#plt.savefig('./result/plot/std5.png')
plt.show()

sns.distplot(gt_df['std6'], color ='blue', label = 'O')
sns.distplot(ff_df['std6'], color ='red', label = 'X')
plt.legend(title='O/X')
#plt.savefig('./result/plot/std6.png')
plt.show()

gt_df.boxplot(return_type = 'axes')
ff_df.boxplot(return_type = 'axes')
gt_df.iloc[:,2:].mean()

#gt_df = pd.read_csv('./data/pose_data/O_pose_data.csv')
#gt_df.insert(9, 'label', 1.0)
std_df = gt_df.iloc[:,2:]
filepath_df = gt_df.iloc[:,1] #파일명만 따로가지고옴
std_dict = { 
            'std1' : std_df['std1'].mean(),
            #'std2' : std_df['std2'].mean(),
            'std3' : std_df['std3'].mean(),
            'std4' : std_df['std4'].mean(),
            'std5' : std_df['std5'].mean(),
            'std6' : std_df['std6'].mean(),
}
std_dict

###############train######################
#train_df = pd.read_csv('./data/pose_data/O_pose_data.csv')
#train_df.insert(9, 'label', 1.0)
X_df = train_df.iloc[:,2:]
filepath_df = train_df.iloc[:,1] #파일명만 따로가지고옴

X_test_std_dict = {
            'std1' : list(X_df['std1'].values),
            #'std2' : list(X_df['std2'].values),
            'std3' : list(X_df['std3'].values),
            'std4' : list(X_df['std4'].values),
            'std5' : list(X_df['std5'].values),
            'std6' : list(X_df['std6'].values),
}
X_test_std_dict
score_dict = dict()

print(len(X_test_std_dict.items()))


#점수화시키기
for key, values in X_test_std_dict.items():
  #print(len(values))
  std = std_dict[key]
  #print(std)
  score_li = []
  for i in range(len(values)):
    score = (1-(abs(values[i]- std)/std)) * 100
    score_li.append(score)
  score_dict[key] = score_li
print(len(score_dict['std1'])) 
  
#for key, values in score_dict.items():
#  print("첫번쨰 데이터만 확인하기")
#  print(f"{filepath_df.values[0]} : {score_dict[key][0]}")

X_p_score = pd.DataFrame(score_dict)
X_p_score= X_p_score.reset_index()
filepath_df = filepath_df.reset_index()

result = pd.concat([X_p_score.iloc[:,1:], filepath_df.iloc[:,-1]], axis = 1)
result.to_csv('./result/std_scoring_csv/train_scoring.csv')
result

#각 std별 개인별점수
score_gt_df = pd.read_csv('./result/std_scoring_csv/train_scoring.csv')
O_df = score_gt_df.iloc[:,:]
#score sum : std1_score + std2_score + std4_score + std5_score + std6_score 
total_score_append = list()
for index,row in O_df.iterrows():
  total_score = (row['std1'] + row['std3']+ row['std4']+ row['std5']+ row['std6']) /5
  total_score_append.append(total_score)
    
O_total_df = pd.DataFrame({'total_score': total_score_append})
O_total_df
total_O_df = pd.concat([score_gt_df, O_total_df],axis=1)

total_O_df
print(tabulate(total_O_df, headers='keys', tablefmt='psql', showindex=True))

#각 std별 개인별점수
score_train_df = pd.read_csv('./result/std_scoring_csv/train_scoring.csv')
X_df = score_train_df.iloc[:,:]
#score sum : std1_score + std2_score + std4_score + std5_score + std6_score 
total_score_append = list()
for index,row in X_df.iterrows():
  total_score = (row['std1'] + row['std3']+ row['std4']+ row['std5']+ row['std6']) /5
  total_score_append.append(total_score)
    
X_total_df = pd.DataFrame({'total_score': total_score_append})
X_total_df
total_X_df = pd.concat([score_train_df, X_total_df],axis=1)
total_X_df.to_csv('./result/std_scoring_csv/total_train_scoring.csv')

print(tabulate(total_X_df, headers='keys', tablefmt='psql', showindex=True))


###############train######################
#ff_df = pd.read_csv('./data/pose_data/O_pose_data.csv')
#ff_df.insert(9, 'label', 1.0)
X_df = valid_df.iloc[:,2:]
filepath_df = valid_df.iloc[:,1] #파일명만 따로가지고옴

X_test_std_dict = {
            'std1' : list(X_df['std1'].values),
            #'std2' : list(X_df['std2'].values),
            'std3' : list(X_df['std3'].values),
            'std4' : list(X_df['std4'].values),
            'std5' : list(X_df['std5'].values),
            'std6' : list(X_df['std6'].values),
}
X_test_std_dict
score_dict = dict()

print(len(X_test_std_dict.items()))


#점수화시키기
for key, values in X_test_std_dict.items():
  #print(len(values))
  std = std_dict[key]
  #print(std)
  score_li = []
  for i in range(len(values)):
    score = (1-(abs(values[i]- std)/std)) * 100
    score_li.append(score)
  score_dict[key] = score_li
print(len(score_dict['std1'])) 
  
#for key, values in score_dict.items():
#  print("첫번쨰 데이터만 확인하기")
#  print(f"{filepath_df.values[0]} : {score_dict[key][0]}")

X_p_score = pd.DataFrame(score_dict)
X_p_score= X_p_score.reset_index()
filepath_df = filepath_df.reset_index()

result = pd.concat([X_p_score.iloc[:,1:], filepath_df.iloc[:,-1]], axis = 1)
result.to_csv('./result/std_scoring_csv/valid_scoring.csv')
result

#각 std별 개인별점수
score_gt_df = pd.read_csv('./result/std_scoring_csv/valid_scoring.csv')
O_df = score_gt_df.iloc[:,:]
#score sum : std1_score + std2_score + std4_score + std5_score + std6_score 
total_score_append = list()
for index,row in O_df.iterrows():
  total_score = (row['std1'] + row['std3']+ row['std4']+ row['std5']+ row['std6']) /5
  total_score_append.append(total_score)
    
O_total_df = pd.DataFrame({'total_score': total_score_append})
O_total_df
total_O_df = pd.concat([score_gt_df, O_total_df],axis=1)

total_O_df
print(tabulate(total_O_df, headers='keys', tablefmt='psql', showindex=True))

#각 std별 개인별점수
score_ff_df = pd.read_csv('./result/std_scoring_csv/valid_scoring.csv')
X_df = score_ff_df.iloc[:,:]
#score sum : std1_score + std2_score + std4_score + std5_score + std6_score 
total_score_append = list()
for index,row in X_df.iterrows():
  total_score = (row['std1'] + row['std3']+ row['std4']+ row['std5']+ row['std6']) /5
  total_score_append.append(total_score)
    
X_total_df = pd.DataFrame({'total_score': total_score_append})
X_total_df
total_X_df = pd.concat([score_ff_df, X_total_df],axis=1)
total_X_df.to_csv('./result/std_scoring_csv/total_valid_scoring.csv')

print(tabulate(total_X_df, headers='keys', tablefmt='psql', showindex=True))

###############test######################
#test_df = pd.read_csv('./data/pose_data/O_pose_data.csv')
#test_df.insert(9, 'label', 1.0)
X_df = test_df.iloc[:,2:]
filepath_df = test_df.iloc[:,1] #파일명만 따로가지고옴

X_test_std_dict = {
            'std1' : list(X_df['std1'].values),
            #'std2' : list(X_df['std2'].values),
            'std3' : list(X_df['std3'].values),
            'std4' : list(X_df['std4'].values),
            'std5' : list(X_df['std5'].values),
            'std6' : list(X_df['std6'].values),
}
X_test_std_dict
score_dict = dict()

print(len(X_test_std_dict.items()))


#점수화시키기
for key, values in X_test_std_dict.items():
  #print(len(values))
  std = std_dict[key]
  #print(std)
  score_li = []
  for i in range(len(values)):
    score = (1-(abs(values[i]- std)/std)) * 100
    score_li.append(score)
  score_dict[key] = score_li
print(len(score_dict['std1'])) 
  
#for key, values in score_dict.items():
#  print("첫번쨰 데이터만 확인하기")
#  print(f"{filepath_df.values[0]} : {score_dict[key][0]}")

X_p_score = pd.DataFrame(score_dict)
X_p_score= X_p_score.reset_index()
filepath_df = filepath_df.reset_index()

result = pd.concat([X_p_score.iloc[:,1:], filepath_df.iloc[:,-1]], axis = 1)
result.to_csv('./result/std_scoring_csv/test_scoring.csv')
result

#각 std별 개인별점수
score_gt_df = pd.read_csv('./result/std_scoring_csv/test_scoring.csv')
O_df = score_gt_df.iloc[:,:]
#score sum : std1_score + std2_score + std4_score + std5_score + std6_score 
total_score_append = list()
for index,row in O_df.iterrows():
  total_score = (row['std1'] + row['std3']+ row['std4']+ row['std5']+ row['std6']) /5
  total_score_append.append(total_score)
    
O_total_df = pd.DataFrame({'total_score': total_score_append})
O_total_df
total_O_df = pd.concat([score_gt_df, O_total_df],axis=1)

total_O_df
print(tabulate(total_O_df, headers='keys', tablefmt='psql', showindex=True))

#각 std별 개인별점수
score_test_df = pd.read_csv('./result/std_scoring_csv/test_scoring.csv')
X_df = score_test_df.iloc[:,:]
#score sum : std1_score + std2_score + std4_score + std5_score + std6_score 
total_score_append = list()
for index,row in X_df.iterrows():
  total_score = (row['std1'] + row['std3']+ row['std4']+ row['std5']+ row['std6']) /5
  total_score_append.append(total_score)
    
X_total_df = pd.DataFrame({'total_score': total_score_append})
X_total_df
total_X_df = pd.concat([score_test_df, X_total_df],axis=1)
total_X_df.to_csv('./result/std_scoring_csv/total_test_scoring.csv')

print(tabulate(total_X_df, headers='keys', tablefmt='psql', showindex=True))


total_O_df = pd.read_csv('./result/csv/total_O_pose_data.csv')
heatmap_df = total_O_df[['std1','std3','std4','std5','std6']]
sns.heatmap(heatmap_df)

total_X_df = pd.read_csv('./result/csv/total_X_pose_data.csv')
heatmap_df = total_X_df[['std1','std3','std4','std5','std6']]
sns.heatmap(heatmap_df)
reg_gt_df = pd.read_csv('./result/std_scoring_csv/std_O_scoring.csv')
reg_ff_df = pd.read_csv('./result/std_scoring_csv/std_X_scoring.csv')

sns.distplot(reg_gt_df['std1'], color ='blue', label = 'O')
sns.distplot(reg_ff_df['std1'], color ='red', label = 'X')
plt.legend(title='O/X')
plt.savefig('./result/plot/scoring_std1.png')
plt.show()

sns.distplot(reg_gt_df['std3'], color ='blue', label = 'O')
sns.distplot(reg_ff_df['std3'], color ='red', label = 'X')
plt.legend(title='O/X')
plt.savefig('./result/plot/scoring_std3.png')
plt.show()

sns.distplot(reg_gt_df['std4'], color ='blue', label = 'O')
sns.distplot(reg_ff_df['std4'], color ='red', label = 'X')
plt.legend(title='O/X')
plt.savefig('./result/plot/scoring_std4.png')
plt.show()

sns.distplot(reg_gt_df['std5'], color ='blue', label = 'O')
sns.distplot(reg_ff_df['std5'], color ='red', label = 'X')
plt.legend(title='O/X')
plt.savefig('./result/plot/scoring_std5.png')
plt.show()

sns.distplot(reg_gt_df['std6'], color ='blue', label = 'O')
sns.distplot(reg_ff_df['std6'], color ='red', label = 'X')
plt.legend(title='O/X')
plt.savefig('./result/plot/scoring_std6.png')
plt.show()

reg_gt_df.iloc[0:]
heatmap_df = reg_gt_df[['std1','std3','std4','std5','std6']]
sns.heatmap(heatmap_df)

reg_ff_df.iloc[0:]
heatmap_df = reg_ff_df[['std1','std3','std4','std5','std6']]
sns.heatmap(heatmap_df)

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch.optim as optim

train_data = pd.concat([
  pd.read_csv('./result/csv/train_O_pose_data.csv'),
  pd.read_csv('./result/csv/train_X_pose_data.csv'),
])
valid_data = pd.concat([
  pd.read_csv('./result/csv/valid_O_pose_data.csv'),
  pd.read_csv('./result/csv/valid_X_pose_data.csv'),
])
train_data['res'] = 0
train_data['res'][0:5] = 1
valid_data['res'] = 0
valid_data['res'][0:4] = 1

test_data = pd.concat([
  pd.read_csv('./result/csv/test_O_pose_data.csv'),
  pd.read_csv('./result/csv/test_X_pose_data.csv'),
])
test_data['res'] = 0
test_data['res'][0:4] = 1

# 훈련 데이터셋 클래스 정의
class AngleDataset(Dataset):
    def __init__(self, angles, labels):
        self.angles = angles
        self.labels = labels
        
    def __len__(self):
        return len(self.angles)
    
    def __getitem__(self, index):
        angle = self.angles[index]
        label = self.labels[index]
        return angle, label
      
class ValidationDataset(Dataset):
    def __init__(self, angles, labels):
        self.angles = angles
        self.labels = labels
        
    def __len__(self):
        return len(self.angles)
    
    def __getitem__(self, index):
        angle = self.angles[index]
        label = self.labels[index]
        return angle, label
    
# MLP 모델 클래스 정의
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
        
    def init_weights(self):
        self.features.weight.data.uniform_ = (-0.5,0.5)
        self.classifier.weight.data.uniform_ = (-0.5,0.5)
        self.classifier.bias.data.zero_()
    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x

class_mapping = {0: 'O', 1: 'X'}

train_angle1 = torch.tensor(train_data.iloc[:,2:3].to_numpy()).to(torch.float32)
train_angle3 = torch.tensor(train_data.iloc[:,4:5].to_numpy()).to(torch.float32)
train_angle4 = torch.tensor(train_data.iloc[:,5:6].to_numpy()).to(torch.float32)
train_angle5 = torch.tensor(train_data.iloc[:,6:7].to_numpy()).to(torch.float32)
train_angle6 = torch.tensor(train_data.iloc[:,7:8].to_numpy()).to(torch.float32)
train_labels = torch.tensor(train_data.iloc[:,8:9].to_numpy()).to(torch.float32)

valid_angle1 = torch.tensor(valid_data.iloc[:, 2:3].to_numpy()).to(torch.float32)
valid_angle3 = torch.tensor(valid_data.iloc[:, 4:5].to_numpy()).to(torch.float32)
valid_angle4 = torch.tensor(valid_data.iloc[:, 5:6].to_numpy()).to(torch.float32)
valid_angle5 = torch.tensor(valid_data.iloc[:, 6:7].to_numpy()).to(torch.float32)
valid_angle6 = torch.tensor(valid_data.iloc[:, 7:8].to_numpy()).to(torch.float32)
valid_labels = torch.tensor(valid_data.iloc[:, 8:9].to_numpy()).to(torch.float32)

angles = [[angle1, angle3, angle4, angle5, angle6] for angle1, angle3, angle4, angle5, angle6 in zip(train_angle1, train_angle3, train_angle4, train_angle5, train_angle6)]
labels = [class_mapping[int(label.item())] for label in train_labels]
labels = [0 if label == 0 else 1 for label in train_labels]
train_labels = torch.tensor(labels, dtype=torch.long)

valid_angles = [[angle1, angle3, angle4, angle5, angle6] for angle1, angle3, angle4, angle5, angle6 in zip(valid_angle1, valid_angle3, valid_angle4, valid_angle5, valid_angle6)]
valid_labels = [class_mapping[int(label.item())] for label in valid_labels]
valid_labels = [0 if label == 0 else 1 for label in valid_labels]
valid_labels = torch.tensor(labels, dtype=torch.long)

scaler = StandardScaler()
angles = scaler.fit_transform(angles)

valid_angles = torch.tensor(valid_angles, dtype=torch.float32)
valid_angles = scaler.transform(valid_angles)

#############dataset & dataloader
batch_size = 10

dataset = AngleDataset(angles, labels)
valid_dataset = ValidationDataset(valid_angles, valid_labels)

train_loader = DataLoader(dataset, batch_size= batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size= batch_size, shuffle=True)

# 모델 생성
input_dim = 5
hidden_dim = 100
output_dim = 2
model = MLP(input_dim, hidden_dim, output_dim)
# 손실 함수와 옵티마이저 정의
#criterion = nn.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 1000
loss_train_epoch = []
loss_train_step = []
loss_valid_epoch = []
loss_valid_step = []
acc_valid_epoch =[]
acc_valid_step =[]


for epoch in range(num_epochs):
    model.train()
    for batch_input, batch_label in train_loader:
        #print(batch_label)
        optimizer.zero_grad()
        # Forward 패스
        outputs = model(batch_input)
        # print(outputs)
        # print(batch_label)
        loss = criterion(outputs, batch_label)
        #print(outputs)
        # Backward 패스 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        loss_train_step.append(loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    
    # 현재 에포크의 손실 출력
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        
        for batch_valid_input, batch_valid_label in valid_loader:
            total_valid = 0
            correct_valid = 0    
            #print(batch_valid_label)
            ##print(batch_valid_label)
            batch_valid_output = model(batch_valid_input)
            #print(batch_valid_output)
            # print(batch_valid_output)
            # print(batch_valid_label)
            batch_valid_loss = criterion(batch_valid_output, batch_valid_label)
            #print(batch_valid_output)
            #print(batch_valid_loss.item())
            #다합하기 3번나오는거 해서 loss 평균구하기
            valid_loss = batch_valid_loss.item()
            valid_loss
            #print(batch_valid_output)
            #print(len(valid_loader))
            #정확도 확인
            #_, valid_predicted = torch.max(batch_valid_output.data, 1)
            valid_predicted = torch.argmax(batch_valid_output.data, dim= 1)
            #print(batch_valid_label)
            #print(valid_predicted)
            print(batch_valid_label)
            print(valid_predicted)
            total_valid += batch_valid_label.size(0)
            correct_valid += (valid_predicted == batch_valid_label).sum().item()
            print(correct_valid)
            valid_accuracy = correct_valid / batch_size
#            print(batch_valid_label)
#            print(batch_valid_output)
            loss_valid_step.append(valid_loss)
            acc_valid_step.append(valid_accuracy)
        #print(2)
        y_vd_hat = model(batch_valid_input)
        l_all = criterion( y_vd_hat, batch_valid_label,)
        loss_valid_epoch.append(l_all.detach())        
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")
        
        

torch.save(   
    model.state_dict(), "./model/MLP.pth"
)

plt.plot(loss_valid_epoch,':')
plt.xlabel('step')
plt.ylabel('loss')
plt.axis([0,num_epochs, 0, max(loss_valid_epoch)])
plt.show()
    
test_angle1 = torch.tensor(test_data.iloc[:,2:3].to_numpy()).to(torch.float32)
test_angle3 = torch.tensor(test_data.iloc[:,4:5].to_numpy()).to(torch.float32)
test_angle4 = torch.tensor(test_data.iloc[:,5:6].to_numpy()).to(torch.float32)
test_angle5 = torch.tensor(test_data.iloc[:,6:7].to_numpy()).to(torch.float32)
test_angle6 = torch.tensor(test_data.iloc[:,7:8].to_numpy()).to(torch.float32)
test_labels = torch.tensor(test_data.iloc[:,8:9].to_numpy()).to(torch.float32)

# 테스트 데이터 생성
test_angles = [[angle1, angle3, angle4, angle5, angle6] for angle1, angle3, angle4, angle5, angle6 in zip(test_angle1, test_angle3, test_angle4, test_angle5, test_angle6)]

# 데이터를 NumPy 배열로 변환
test_angles = torch.tensor(test_angles, dtype=torch.float32)

# 입력 데이터 정규화
test_angles = scaler.transform(test_angles)

# 모델 예측
model.eval()
with torch.no_grad():
    test_inputs = torch.tensor(test_angles, dtype=torch.float32)
    outputs = model(test_inputs)
    predicted = torch.argmax(outputs.data, dim = 1)

# 예측 결과 출력
predicted_labels_int = [[label.item()] for label in predicted]
predicted_labels = [class_mapping[label.item()] for label in predicted]
#print(predicted_labels_int)
test_labels_int = [[int(label.item())] for label in test_labels]
print(f"원래 :{test_labels_int}")
print(f"예측 :{predicted_labels_int}")
      
#print(len(predicted_labels))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 실제 레이블과 예측 레이블을 numpy 배열로 변환
true_labels = test_labels.numpy()
predicted_labels = np.array(predicted_labels_int)

# Confusion Matrix 생성
cm = confusion_matrix(true_labels, predicted_labels)

# 클래스 레이블
class_labels = list(class_mapping.values())

# Confusion Matrix 시각화
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=class_labels,
       yticklabels=class_labels,
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

# 각 셀에 숫자 표시
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.show()
        
        
############################## data transform #######################################
############################## data transform #######################################
############################## data transform #######################################
############################## data transform #######################################
############################## data transform #######################################
############################## data transform #######################################
############################## data transform #######################################
############################## data transform #######################################
############################## data transform #######################################
############################## data transform #######################################
############################## data transform #######################################