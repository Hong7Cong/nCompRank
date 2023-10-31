import torch
from torchvision import models, transforms
from utils import *
from datetime import date
from glaucomaDataloader import *
import argparse
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description="Just an example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--datalen", action="store", type=int, default=10)
parser.add_argument("-a", "--vallen", action="store", type=int, default=100)
parser.add_argument("-l", "--lr", action="store", type=float, default=1e-5)
parser.add_argument("-m", "--momentum", action="store", type=float, default=0)
parser.add_argument("-b", "--batch_size", action="store", type=int, default=4)
parser.add_argument("-e", "--step_size", action="store", type=int, default=10)
parser.add_argument("-o", "--mode", action="store", type=str, default='mdindex-longtitude')
parser.add_argument("-v", "--issaved", action="store", type=str, default='local')
parser.add_argument("-f", "--fextract", action="store", type=str, default='resnet50', help= 'Feature-extractor Selection')
parser.add_argument("-c", "--cotrain", action="store", type=str, default='False', help= 'Ranknet and Feature Extractor is Co-trained or not')
parser.add_argument("--simclr", action="store", type=str, default='/mnt/c/Users/PCM/Dropbox/pretrained/SimCLR/checkpoint_0050.pth.tar')
parser.add_argument("-t", "--token", action="store", type=str, default='sl.Bl84pN1AEQbMOE6ZZwSxfQEDqRGxzahbMkAhSgeKtATZYS8wCidqP4pBWh2PcNHMQeCo1wQJ0QJMEnOCDAtMEuMiSfUJb8cWv-HUR2JMmHrBPTs2EEzJI3KM659YAqvLttTYVroKrWq2FrnbDo0e01I', help= 'Tokens to access and save to dropbox')
args = parser.parse_args()
config = vars(args)
print(config)

datalen     = args.datalen
lr          = args.lr
momentum    = args.momentum
batch_size  = args.batch_size
mode        = args.mode
issaved     = args.issaved  
vallen      = args.vallen
step_size   = args.step_size
fextract    = args.fextract
cotrain     = (args.cotrain == 'True' or args.cotrain == 'true')
simclr      = args.simclr if (args.simclr != 'None') else None
print('cotrain ', cotrain)
access_token = args.token
transferData = TransferData(access_token)

print("--- Init parameters and Load dataset to local memory ---")
mean        = [0.6821, 0.4575, 0.2626]#[ 0.7013, -0.1607, -0.7902]#[0.485, 0.456, 0.406]
std         = [0.1324, 0.1306, 0.1022]#[0.5904, 0.5008, 0.3771]#[0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# ranknet = RankNet_wresnet()
cuda = torch.cuda.is_available()
glaucoma_train_dataset = GlaucomaDataset(phase="train", 
                                         mode=mode,
                                         datalen=datalen,
                                         transform = data_transforms['train']) # Returns pairs of images and target same/different
glaucoma_test_dataset = GlaucomaDataset(phase="val", 
                                         mode=mode,
                                         datalen=vallen,
                                         transform = data_transforms['val'])

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
glaucoma_train_loader = torch.utils.data.DataLoader(glaucoma_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
glaucoma_test_loader = torch.utils.data.DataLoader(glaucoma_test_dataset, batch_size=batch_size, **kwargs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

ranknet = RankNet_wresnet2(feature_extractor=fextract, cotrain = cotrain, simclr = simclr)
ranknet.to(device)
loss_fn = torch.nn.CrossEntropyLoss()

if(cotrain):
    optimizer = optim.SGD(ranknet.parameters(), lr=lr, momentum=momentum)
else:
    if(fextract == 'resnet50'):
        print('Update params of fully-connected only')
        optimizer = optim.SGD([{'params': ranknet.fextractor.fc.parameters(), 'lr': lr*0.1}, {'params': ranknet.dense.parameters()}], lr=lr, momentum=momentum)
    else:
        optimizer = optim.SGD([{'params': ranknet.fextractor.classifier.parameters()}, {'params': ranknet.dense.parameters()}], lr=lr, momentum=momentum)

scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8)

if(simclr):
    for param in ranknet.fextractor.parameters():
        param.requires_grad = False

optimizer.zero_grad()
acc_epoch = []

# print(f'begin training...with setting lr={lr}, momentum={momentum}, step_size={step_size}, datalen={datalen}, optimizer={optimizer}')
print('Start training')
for e in range(100):
# Loop epoch
    # losses = []
    # losses_val = []
    total_loss = 0
    temp_acc = 0
    comp_acc = 0
    val_acc = 0
    data_len = len(glaucoma_train_dataset)
    ranknet.train()
    for batch_idx, (data, target, _) in enumerate(glaucoma_train_loader):
        # Training phase
        optimizer.zero_grad()
        target = target.to(device)
        feature1 = data[0].to(device)
        feature2 = data[1].to(device)

        outputs = ranknet(feature1, feature2) #torch.squeeze(ranknet(feature1, feature2))
        outputs_reverse = ranknet(feature2, feature1)
        loss1 = loss_fn(outputs, target)
        loss2 = loss_fn(outputs_reverse, torch.abs(target-1)) #For multitask learning
        loss = loss1 + loss2
        # losses.append(loss.item())

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        comp_acc += torch.sum(target == torch.max(outputs, 1)[1]).tolist()
    
    ranknet.eval()
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(glaucoma_test_loader):
            target = target.to(device)
            feature1 = data[0].to(device)
            feature2 = data[1].to(device)
            outputs = ranknet(feature1, feature2)
            val_acc += torch.sum(target == torch.max(outputs, 1)[1]).tolist()

    scheduler.step()
    print(f"epoch {e}: With LR {optimizer.param_groups[0]['lr']} --- comp acc is {comp_acc/data_len} --- loss is {total_loss/data_len} --- Val acc is {val_acc/vallen}")
    
    if(e % 1 == 0 and issaved != 'nosave'):
        today = date.today()
        todat_str = today.strftime("%d%m%Y")
        save_name = f'siaseme10_mode{mode}_fextra{fextract}_lr{lr}_mo{momentum}_e{e}_{todat_str}_l{datalen}_cotrain{cotrain}.pt'

        if(issaved == 'dropbox'):
            torch.save(ranknet.state_dict(), f'./pretrained/mdindex_siamese10/{save_name}')
            transferData.upload_file(f'./pretrained/mdindex_siamese10/{save_name}', f'/pretrained-models/{save_name}')
            print(f'save to dropbox ./pretrained-models/{save_name}')
        elif(issaved == 'ggdrive'):
            torch.save(ranknet.state_dict(), f'/content/drive/MyDrive/hongn/pretrained/mdindex_siamese10/{save_name}')
            print(f'save to ggdrive /content/drive/MyDrive/hongn/pretrained/mdindex_siamese10/{save_name}')
        else:
            torch.save(ranknet.state_dict(), f'./pretrained/mdindex_siamese10/{save_name}')
            print(f'save locally to ./pretrained/mdindex_siamese10/{save_name}')

print('...DONE')