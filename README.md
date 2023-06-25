# 2023-Biomedical AI Project
# shooting-tracker-feedback-AI
# 축구 슈팅 자세 개선을 위한 AI Pose Tracking correlation Feed back AI 👵👴
# AI Pose Tracking Correlation Feedback AI for Improving Soccer Shot Formation with Mediapipe Pose
#### Project nickname : 슈팅왕통키
#### Project execution period : 2023.03~2023.06
#### Project Hosting : [최대현](https://valiant-barnacle-6bf.notion.site/Feedback-AI-78d9ec8ddb0846cb800280940824d7ad?pvs=4)
-----------------------
## Description
축구 슈팅 중 공을 발등에 맞추는 것은 슈팅을 잘하기 위해선 중요하다. 발등에 공을 잘 맞추게 되면 원하는 방향으로 강하고 정확한 슈팅을 할 수 있다. 그러나 혼자서 슈팅을 연습할 때는 공을 발등에 맞추기가 어렵다. 더불어, 사람의 눈으로 슈팅 자세를 확인하고, 전문가가 아닌 이상 축구 슈팅자세의 적합성을 판단하는 것은 노하우가 존재하지 않은 전문가가 아닌 이상 바른 교정을 하기엔 많은 시간이 필요하다. 따라서 아마추어들이 많은 어려움을 겪는다는 문제를 해결하기 위해 축구 선수 출신의 코치에게 조언을 구하여, 슈팅 시 디딤발의 위치, 허벅지 각도 등 여러가지 조건을 정확히 맞추어야 한다는 것을 알아, 이에 대한 엄격한 기준을 선정하여 축구 자세를 교정해줄 수 있는 딥러닝 서비스를 제작하게 되었다. 사용자들이 휴대폰만 있으면 자신의 슈팅 사진을 입력하여, 그것을 분석하여 얼마나 정확하게 슈팅 했는지 판별하고, 피드백을 제공하는 ‘축구 자세 교정 AI를 이용한 미래의 손흥민 만들기 프로젝트 : 슈팅왕통키’를 진행했다.

![4](https://github.com/dablro12/shooting-tracker-feedback-AI/assets/54443308/0fe3cc7f-a3a3-4d27-85c8-76476d6d372f)
서비스 흐름도

### 1. function list
![noname01](https://user-images.githubusercontent.com/78933101/135653625-d14db805-ab5e-4505-a7b0-5d0a3af2ad59.png)

|구분|기능|구현|
|------|---|---|
|S/W|자세 적합성 확인|Pretrained CNN Model with ResNet, DenseNet|
|S/W|각도 측정 데이터 변환|Mediapipe Pose with google|
|S/W|Visualization|Gradio|
|H/W|입력 모듈|Iphone 12 pro|

### 2. detailed function
#### Software
![2](https://user-images.githubusercontent.com/78933101/135657961-cbbc1775-37d0-42d4-a47c-26850674f580.png)
S/W 구성도

- shooting posture correlation inference : 주어진 축구 슈팅 이미지를 가지고 예측해주는 모델
- img2csv : 추출한 이미지 각도 데이터를 csv 데이터로 변환하는 프로그램
- Analysis : 입력된 각도 및 이미지를 분석하는 모델
- 시각화 : gradio를 이용하여 웹페이지로 구현가능한 웹서비스


## Environment

> Python Version 3.9.7 (Window)
> pytorch latest version
> Linux Ubuntu


## Prerequisite
> import gradio as gr
>
> import tensorflow as tf
>
> import numpy as np
>
> from PIL import Image
>
> import requests
>
> import torch
>
> from torchvision.transforms import ToTensor
>
> from torchvision.models import densenet
>
> from torchvision import transforms
>
> import torchvision
>
> from tensorflow.keras.models import Model
>
> import math
>
> import cv2 as cv
>
> import os
>
> import mediapipe as mp
>
> import pandas as pd
>
> from google.colab.patches import cv2_imshow as cv_imshow
>
> import matplotlib.pyplot as plt
>

## Files
`data` Data file with model(densenet), data, plot, csv 

`main.py` Main code python file

`main.ipynb` Main code for practicing project

`inferece.ipynb` visualization and usage file to service


## Usage 
`inference.ipynb`
