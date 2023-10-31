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
                path2images="./datasets/annotated_eyes/", 
                phase="train", 
                mode="pairwise", 
                transform=None,
                datalen=100,
                certain=True,
                seed=None):
        self.phase = phase
        # total_images = glob.glob(f"{path2images}/*.jpg")
        # self.eyes_files = total_images[:500] if(phase == "train") else total_images[500:]
        # self.n = len(self.eyes_files)             # Total number of images
        self.datalen = datalen                     # Number of pair of images for training/testing
        self.certain = certain
        if(seed):
            seed_everything(seed)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) if(transform == None) else transform
        
        self.mode = mode
        self.setofpaths = [] # For listwise
        if(phase == 'train'):
            self.glau_patients = glob.glob(f'./datasets/longtitude/*')[:300]
        elif(phase == 'val'):
            self.glau_patients = glob.glob(f'./datasets/longtitude/*')[300:350]
        elif(phase == 'test'):
            self.glau_patients = glob.glob(f'./datasets/longtitude/*')[350:]
        else:
            assert False, 'No phase founded'

        if(self.mode == "pointwise"):
            self.path2data = self.eyes_files
            print('TBA')

        elif(self.mode == "pairwise"):
            # Create random pick 2 out of n
            random_pick = torch.randint(0, self.n, (self.datalen*2,))
            self.index1 = random_pick[:self.datalen]
            self.index2 = random_pick[self.datalen:]
            # Save Path to each pair of images
            self.paths1 = [self.eyes_files[i] for i in self.index1] 
            self.paths2 = [self.eyes_files[i] for i in self.index2]
            
        elif(self.mode == "listwise"):
            for i in range(self.datalen):
                rand_len = np.random.randint(3,30) # Generate random len of list
                rand_index = torch.randint(0, self.n, (rand_len,)) # Pick rand_len indexs from n
                rand_path = [self.eyes_files[i] for i in rand_index]
                self.setofpaths.append(rand_path)

        elif(self.mode == "pairwise-longtitude"):
            glau_patients = self.glau_patients
            NOPATIENTS = len(glau_patients)
            random_pick_patient = torch.randint(0, NOPATIENTS, (self.datalen,))
            # random_pick_eyes    = torch.randint(0, 2, (self.datalen,))

            self.paths1 = []
            self.paths2 = []
            self.complabels = []
            self.classlabelsA = []
            self.classlabelsB = []
            # self.complabels = []
            self.debug = []
            for i in random_pick_patient:

                pat_eye = glau_patients[i]
                meta_data   = pd.read_csv(f'{pat_eye}/meta_data.csv')
                list_imgs   = meta_data.filename.str.replace('(tif)', 'jpg', regex=True).to_numpy() 
                pick2imgs   = torch.randint(0, len(list_imgs), (2,))

                self.paths1.append(f"{pat_eye}/" + list_imgs[pick2imgs[0]])
                self.paths2.append(f"{pat_eye}/" + list_imgs[pick2imgs[1]])
                self.complabels.append((pick2imgs[0] < pick2imgs[1]))

                self.classlabelsA.append((meta_data.iloc[pick2imgs[0].numpy()].enpoagdisc == 'YES') * torch.tensor(1))
                self.classlabelsB.append((meta_data.iloc[pick2imgs[1].numpy()].enpoagdisc == 'YES') * torch.tensor(1))

                self.debug.append((meta_data.iloc[pick2imgs[0].numpy()].filename, meta_data.iloc[pick2imgs[1].numpy()].filename))

            self.complabels = torch.stack(self.complabels) * 1

        elif(self.mode == "multitask-longtitude"):
            glau_patients = self.glau_patients
            NOPATIENTS = len(glau_patients)

            self.paths1 = []
            self.paths2 = []
            self.complabels = []
            self.classlabelsA = []
            self.classlabelsB = []
            # self.complabels = []
            self.debug = []
            curlen = 0
            while(curlen < self.datalen):
                random_pick_patient = torch.randint(0, NOPATIENTS, (1,))[0]
                random_pick_actions = torch.randint(0, 4, (1,))[0]
                pat_eye = glau_patients[random_pick_patient]
                meta_data   = pd.read_csv(f'{pat_eye}/meta_data.csv')
                listglau    = meta_data[meta_data.enpoagdisc == "YES"].filename.str.replace('(tif)', 'jpg', regex=True).to_numpy()
                listnorm    = meta_data[meta_data.enpoagdisc != "YES"].filename.str.replace('(tif)', 'jpg', regex=True).to_numpy()
                if(len(listglau) == 0):
                    continue

                if(random_pick_actions == 0):
                    pick2imgs = torch.randint(0, len(listnorm), (2,))
                    self.paths1.append(f"{pat_eye}/" + listnorm[pick2imgs[0]])
                    self.paths2.append(f"{pat_eye}/" + listnorm[pick2imgs[1]])
                    self.complabels.append((pick2imgs[0] < pick2imgs[1]))
                    self.classlabelsA.append(torch.tensor(0))
                    self.classlabelsB.append(torch.tensor(0))
                    
                elif(random_pick_actions == 1):
                    pick2imgs = torch.stack([torch.randint(0, len(listnorm), (1,)),torch.randint(0, len(listglau), (1,))]).squeeze()
                    self.paths1.append(f"{pat_eye}/" + listnorm[pick2imgs[0]])
                    self.paths2.append(f"{pat_eye}/" + listglau[pick2imgs[1]])
                    self.complabels.append(torch.tensor(True))
                    self.classlabelsA.append(torch.tensor(0))
                    self.classlabelsB.append(torch.tensor(1))
                    
                elif(random_pick_actions == 2):
                    pick2imgs = torch.stack([torch.randint(0, len(listglau), (1,)),torch.randint(0, len(listnorm), (1,))]).squeeze()
                    self.paths1.append(f"{pat_eye}/" + listglau[pick2imgs[0]])
                    self.paths2.append(f"{pat_eye}/" + listnorm[pick2imgs[1]])
                    self.complabels.append(torch.tensor(False))
                    self.classlabelsA.append(torch.tensor(1))
                    self.classlabelsB.append(torch.tensor(0))
                    
                else:
                    pick2imgs = torch.randint(0, len(listglau), (2,))
                    self.paths1.append(f"{pat_eye}/" + listglau[pick2imgs[0]])
                    self.paths2.append(f"{pat_eye}/" + listglau[pick2imgs[1]])
                    self.complabels.append((pick2imgs[0] < pick2imgs[1]))
                    self.classlabelsA.append(torch.tensor(1))
                    self.classlabelsB.append(torch.tensor(1))
                curlen = curlen + 1
            self.complabels = torch.stack(self.complabels) * 1
        
        elif(self.mode == "mdindex-longtitude"):
            glau_patients = self.glau_patients
            NOPATIENTS = len(glau_patients)
            globalmeta = pd.read_csv('./datasets/ohts_merged_20200918.csv', low_memory=False)
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
            assert False, f"No mode {self.mode} found,  try mode=mdindex-longtitude instead"

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if(self.mode == "pointwise"):
            self.path2data[index]
            print('TBA')

        elif(self.mode == "pairwise"):
            img1 = self.paths1[index]
            mask1 = get_mask(self.paths1[index])
            img2 = self.paths2[index] #read_image(self.paths2[index])
            mask2 = get_mask(self.paths2[index])
            if(self.certain):
                target = torch.tensor(1) if (torch.sum(mask1) > torch.sum(mask2)) else torch.tensor(0)
            else:
                target = torch.nn.Softmax(dim=0)(torch.stack([torch.sum(mask1).float()/torch.Tensor([50.0])[0], torch.sum(mask2).float()/torch.Tensor([50.0])[0]]))[0]
            if(self.transform):
                img1 = self.transform(Image.open(img1))
                img2 = self.transform(Image.open(img2))
            return (img1, img2), target

        elif(self.mode == "listwise"):
            # Get labels - position
            maskoflist = torch.stack([torch.sum(get_mask(i)) for i in self.setofpaths[index]])
            # position_label = torch.argsort(maskoflist)
            # Get list of images
            listofimgs = [self.transform(Image.open(i)) for i in self.setofpaths[index]]
            return torch.stack(listofimgs), maskoflist
        
        elif(self.mode == "pairwise-longtitude"):
            img1 = self.paths1[index]
            img2 = self.paths2[index]
            target = self.complabels[index]
            name = self.debug[index]
            if(self.transform):
                img1 = self.transform(Image.open(img1))
                img2 = self.transform(Image.open(img2))

            return (img1, img2), target, name
        
        elif(self.mode == "multitask-longtitude"):
            img1 = self.paths1[index]
            img2 = self.paths2[index]
            target1 = self.classlabelsA[index]
            target2 = self.classlabelsB[index]
            target3 = self.complabels[index]
            name1 = img1
            name2 = img2
            if(self.transform):
                img1 = self.transform(Image.open(img1))
                img2 = self.transform(Image.open(img2))

            return (img1, img2), target3, (name1, name2, target1, target2)
        
        elif(self.mode == "mdindex-longtitude" or self.mode == "mdindex-latitude"):
            img1 = self.paths1[index]
            img2 = self.paths2[index]
            labels = self.complabels[index]
            name1 = img1
            name2 = img2
            if(self.transform):
                img1 = self.transform(Image.open(img1))
                img2 = self.transform(Image.open(img2))

            return (img1, img2), labels, (name1, name2, self.mdA[index], self.mdB[index])

        assert False, f"No item return,  please check dataset mode"

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