import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from PIL import Image
import torchvision.transforms.functional as F
from captum.attr import FeatureAblation
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class Siamese(torch.nn.Module):
    def __init__(self, 
                 fcnet = None, 
                 feature_extractor='resnet50', 
                 cotrain=True, 
                 simclr=None,
                 activation='sigmoid'):
        
        super(Siamese, self).__init__()
        # Init feature extractor
        self.fextractor = get_feature_extractor(feature_extractor, fcnet, cotrain, simclr=simclr)

    def forward(self, x1, x2):
        x1 = self.fextractor(x1)
        x2 = self.fextractor(x2)
        return torch.nn.Sigmoid()(x1-x2)
    
    def get_model(self):
        return self.fextractor

class SiameseN(torch.nn.Module):
    def __init__(self, 
                 fcnet = None, 
                 feature_extractor='resnet50', 
                 cotrain=True, 
                 ncriteria=10, 
                 simclr=None):
        
        super(SiameseN, self).__init__()

        self.fextractor = get_feature_extractor(feature_extractor, fcnet, cotrain, model = 'siamese10', ncriteria = ncriteria, simclr=simclr)
        
        self.dense = nn.Sequential(torch.nn.Linear(ncriteria, 4), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(4, 2))

        for param in self.dense.parameters():
            param.requires_grad = True
        
    def forward(self, x1, x2):
        x1 = self.fextractor(x1)
        x2 = self.fextractor(x2)
        return self.dense(torch.nn.Sigmoid()(x1-x2))
    
    def get_model(self):
        return self.fextractor
    
    def get_dense(self):
        return self.dense
    

def get_feature_extractor(feature_extractor = 'resnet50', fcnet = None, cotrain=True, ncriteria=10, model='siamese1', simclr = None):
    if(feature_extractor == 'resnet50'):    
        fextractor = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        in_features = 2048
        if(simclr):
            print('load simclr resnet')
            ressimclr = ResNetSimCLR('resnet50', 1000)
            state_dict = torch.load(simclr)
            ressimclr.load_state_dict(state_dict['state_dict'])
            fextractor = ressimclr.backbone
        fextractor.fc = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet
    elif(feature_extractor == 'resnet101'):    
        fextractor = models.resnet101(weights='ResNet101_Weights.DEFAULT')
        in_features = 2048
        if(simclr):
            print('load simclr resnet')
            ressimclr = ResNetSimCLR('resnet101', 1000)
            state_dict = torch.load(simclr)
            ressimclr.load_state_dict(state_dict['state_dict'])
            fextractor = ressimclr.backbone
        fextractor.fc = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet
    elif(feature_extractor == 'densnet121'):
        fextractor = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        in_features = 1024
        fextractor.classifier = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet
        # fextractor._modules['classifier'] = fextractor._modules.pop('classifier')
    elif(feature_extractor == 'vgg19'):
        fextractor = models.vgg19()
        fextractor.load_state_dict(torch.load('./pretrained/vgg19-dcbb9e9d.pth'))
        in_features = 25088 # https://www.geeksforgeeks.org/vgg-16-cnn-model/ length of vgg19
        fextractor.classifier = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet 
        # fextractor._modules['fc'] = fextractor._modules.pop('classifier')
    elif(feature_extractor == 'vit16'):
        fextractor = models.vit_b_16()
        in_features = 768
        fextractor.load_state_dict(torch.load('./pretrained/vit_b_16-c867db91.pth'))
        fextractor.heads.head = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet 
        # fextractor.classifier = get_default_fc(in_features) if (fcnet == None) else fcnet
    else:
        assert False, 'No feature extractor founded'

    for param in fextractor.parameters():
            param.requires_grad = cotrain
    if(feature_extractor == 'resnet50' or feature_extractor == 'resnet101'):        
        for param in fextractor.fc.parameters():
            param.requires_grad = True
    elif(feature_extractor == 'vit16'):
        for param in fextractor.heads.parameters():
            param.requires_grad = True
    else:
        for param in fextractor.classifier.parameters():
            param.requires_grad = True

    return fextractor

def get_default_fc(in_features=2048, model='siamese1', ncriteria=10):
    if(model=='siamese1'):
        ret =  nn.Sequential(torch.nn.Linear(in_features, 256),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(256, 64),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(64, 1))
    else:
        ret =  nn.Sequential(torch.nn.Linear(in_features, 256),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(256, 64),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(64, ncriteria))
    return ret