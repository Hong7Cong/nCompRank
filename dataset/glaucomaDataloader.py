import torch
# from torchvision.utils import make_grid
from torchvision.io import read_image
import matplotlib.pyplot as plt
import glob
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
# from transformers import ViTForImageClassification
import torchvision.transforms.functional as F
import pandas as pd
from utils.constants import *

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def getlistpatients():
    meta_data = pd.read_csv("./datasets/ohts_merged_20200918.csv")

    # Get list of patients' id that have glaucoma
    glau_patients = []
    for i in range(len(meta_data)):
        if(meta_data.iloc[i].enpoagdisc == "YES"):
            if(meta_data.iloc[i].ran_id not in glau_patients):
                glau_patients.append(meta_data.iloc[i].ran_id)
    return glau_patients

class GlaucomaDataset(Dataset):
    def __init__(self, 
                datapath = "/mnt/c/Users/PCM/Dropbox/longtitude/",
                metadata = "/mnt/c/Users/PCM/Dropbox/ohts_merged_20200918.csv",
                phase="train", 
                mode="pairwise", 
                transform=None,
                datalen=100,
                certain=True,
                seed=None):
        self.phase = phase
        self.datalen = datalen # Number of image pairs for training/testing
        self.certain = certain
        self.mode = mode
        if(seed):
            seed_everything(seed)

        self.transform = data_transforms[self.phase] if(transform == None) else transform
        
        self.setofpaths = [] # For listwise
        if(self.phase == 'train'):
            self.glau_patients = glob.glob(f'{datapath}*')[:300]
            # print("len(self.glau_patientssss) = " + f'{datapath}*', len(self.glau_patients))
        elif(self.phase == 'val'):
            self.glau_patients = glob.glob(f'{datapath}*')[300:350]
        elif(self.phase == 'test'):
            self.glau_patients = glob.glob(f'{datapath}*')[350:]
        else:
            assert False, 'No phase founded'

        assert (len(self.glau_patients) != 0), 'No patients found, re-check datapath={datapath}'

        if(self.mode == "mdindex-longtitude"):
            glau_patients = self.glau_patients
            NOPATIENTS = len(glau_patients)
            globalmeta = pd.read_csv(metadata, low_memory=False)
            globalmeta.filename = globalmeta.filename.str.replace('(tif)', 'jpg', regex=True).to_numpy()

            self.paths1 = []
            self.paths2 = []
            self.mdA = []
            self.mdB = []
            self.complabels = []
            self.debug = []
            curlen = 0
            while(curlen < self.datalen):
                random_pick_patient = torch.randint(0, NOPATIENTS, (1,))[0]
                random_pick_actions = torch.randint(0, 4, (1,))[0]
                pat_eye = glau_patients[random_pick_patient]
                meta_data   = pd.read_csv(f'{pat_eye}/meta_data.csv')
                listglau    = meta_data[meta_data.enpoagdisc == "YES"].filename.str.replace('(tif)', 'jpg', regex=True).to_numpy()
                listnorm    = meta_data[meta_data.enpoagdisc != "YES"].filename.str.replace('(tif)', 'jpg', regex=True).to_numpy()
                mdindexA = None
                mdindexB = None
                if(len(listglau) == 0):
                    continue

                if(random_pick_actions == 0):
                    pick2imgs = torch.randint(0, len(listnorm), (2,))
                    self.paths1.append(f"{pat_eye}/" + listnorm[pick2imgs[0]])
                    self.paths2.append(f"{pat_eye}/" + listnorm[pick2imgs[1]])
                    mdindexA = globalmeta[globalmeta.filename == listnorm[pick2imgs[0]]].mdindex.to_numpy()[0]
                    mdindexB = globalmeta[globalmeta.filename == listnorm[pick2imgs[1]]].mdindex.to_numpy()[0]
                    self.complabels.append(torch.tensor(mdindexA > mdindexB))
                    
                elif(random_pick_actions == 1):
                    pick2imgs = torch.stack([torch.randint(0, len(listnorm), (1,)),torch.randint(0, len(listglau), (1,))]).squeeze()
                    self.paths1.append(f"{pat_eye}/" + listnorm[pick2imgs[0]])
                    self.paths2.append(f"{pat_eye}/" + listglau[pick2imgs[1]])
                    self.complabels.append(torch.tensor(True))
                    
                elif(random_pick_actions == 2):
                    pick2imgs = torch.stack([torch.randint(0, len(listglau), (1,)),torch.randint(0, len(listnorm), (1,))]).squeeze()
                    self.paths1.append(f"{pat_eye}/" + listglau[pick2imgs[0]])
                    self.paths2.append(f"{pat_eye}/" + listnorm[pick2imgs[1]])
                    self.complabels.append(torch.tensor(False))
                    
                else:
                    pick2imgs = torch.randint(0, len(listglau), (2,))
                    self.paths1.append(f"{pat_eye}/" + listglau[pick2imgs[0]])
                    self.paths2.append(f"{pat_eye}/" + listglau[pick2imgs[1]])
                    mdindexA = globalmeta[globalmeta.filename == listglau[pick2imgs[0]]].mdindex.to_numpy()[0]
                    mdindexB = globalmeta[globalmeta.filename == listglau[pick2imgs[1]]].mdindex.to_numpy()[0]
                    self.complabels.append(torch.tensor(mdindexA > mdindexB))
                    
                self.mdA.append(mdindexA)
                self.mdB.append(mdindexB)
                curlen = curlen + 1

            self.mdA = np.stack(self.mdA)
            self.mdB = np.stack(self.mdB)
            self.complabels = torch.stack(self.complabels) * 1

        elif(self.mode == "mdindex-latitude"):
            self.paths1 = []
            self.paths2 = []
            self.complabels = []
            curlen = 0
            meta_data = pd.read_csv('./datasets/ohts_merged_20200918.csv', low_memory=False)
            data_sortby_md = meta_data.loc[:,['filename', 'mdindex']].sort_values(by=['mdindex']).dropna().reset_index(drop=True)
            while(curlen < self.datalen):
                pathA, pathB, mdindexA, mdindexB = get_pair_mdindex_latitude(data_sortby_md)
                self.paths1.append('/mnt/c/Users/PCM/Dropbox/glauLarge/' + pathA)
                self.paths2.append('/mnt/c/Users/PCM/Dropbox/glauLarge/' + pathB)
                self.complabels.append(torch.tensor(mdindexA > mdindexB))
                curlen = curlen + 1
            self.complabels = torch.stack(self.complabels) * 1
        else:
            assert False, f"No mode {self.mode} found,  re-try mode=mdindex-longtitude for example"

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img1 = self.paths1[index]
        img2 = self.paths2[index]
        labels = self.complabels[index]
        name1 = img1
        name2 = img2
        if(self.transform):
            img1 = self.transform(Image.open(img1))
            img2 = self.transform(Image.open(img2))

        return (img1, img2), labels, (name1, name2, self.mdA[index], self.mdB[index])

    # def get_path1(self):
    #     return self.paths1
        
    def __len__(self):
        return self.datalen
    
    def get_classlabels(self):
        return torch.stack(self.classlabelsA), torch.stack(self.classlabelsB)

def get_mask(path, prefix = "./datasets/annotated_eyes/"):
    """
    This function return annotation mask aggregated from multiple medical experts.
    The overlapping between expert's annomations are considered the same important
    as the area not overlapping
    Input:
        path: The path to an image
        prefix:  Prefix until image ID
    Output: 
        aggregate_mask: Annotation mask of an image (have the same height and width with
    original images but color dimension = 1). Values in Mask is true/false, indicate
    which pixel is consider contain glaucoma.
    """
    image_ID = path.split(prefix)[1].split(".jpg")[0]
    # Looking for annotation of images ID
    files = glob.glob(f"./datasets/annotators/OHTS_{image_ID}_*.png")
    sum = 0
    if(files == []):
        return torch.tensor(0)
    for f in files:
        temp = read_image(f)
        sum = sum + temp

    aggregate_mask = (sum[0] != 0)
    return aggregate_mask

def get_pair_mdindex_latitude(data_sortby_md):
    """
    This function return pair of sample for comparison with respect to mdindex
    Input:
        data_sortby_md
    Output: 
        pathA and pathB
    """
    std = np.std(data_sortby_md.mdindex)
    range_of_uncertain = std*0.6 # (percentage OF standard deviation)
    pickA = torch.randint(0, len(data_sortby_md), (1,))
    mdindexA = data_sortby_md.iloc[[pickA[0]]].mdindex.to_numpy()[0]
    data_wo_noisypair = pd.concat([data_sortby_md[data_sortby_md.mdindex < mdindexA - range_of_uncertain], data_sortby_md[data_sortby_md.mdindex > mdindexA + range_of_uncertain]])
    temp = torch.randint(0, len(data_wo_noisypair), (1,))
    pickB = data_wo_noisypair.iloc[[temp[0]]].index
    mdindexB = data_sortby_md.iloc[[pickB[0]]].mdindex.to_numpy()[0]
    nameA = data_sortby_md.filename.iloc[[pickA[0]]].str.replace('(tif)', 'jpg', regex=True).replace('-S.jpg', '-L.jpg', regex=True).to_numpy()[0]
    nameB = data_sortby_md.filename.iloc[[pickB[0]]].str.replace('(tif)', 'jpg', regex=True).replace('-S.jpg', '-L.jpg', regex=True).to_numpy()[0]
    return nameA, nameB, mdindexA, mdindexB