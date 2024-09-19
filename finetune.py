import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import model 
from dataloader import *
from utils import MetricLogger, ProgressLogger

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(state,filename='model'):
    torch.save(state, filename + '.pth.tar')

if __name__ == '__main__':
    ark_model = model.Resnet50_ark([25,7,20]).to(device)
    
    # Load the ResNet50 pre-trained with ARK
    model_dict = torch.load("Models/ODIR_RFMID_MURED_Fin/Adamw with cosine29.path.tar")
    ark_model.load_state_dict(model_dict['state_dict'])
    
    # Freeze all layers
    for param in ark_model.parameters():
        param.requires_grad = False
    
    # Add new head to adapt with EDC dataset
    new_head = nn.Linear(ark_model.model.fc.in_features, 3)
    ark_model.omni_heads[0] = new_head
    ark_model.cuda()

    Training_dataset = Eye_Dataset(images_path="dataset/eye_dieases_classification_images/", file_path="dataset/eye_diseases_classification_train.csv",
                                    imagetype="png", train=True)

    Training_dataloader = DataLoader(dataset=Training_dataset, batch_size=64, shuffle=True,
                                            num_workers=1, pin_memory=True)
    
    Validation_dataset = Eye_Dataset(images_path="dataset/eye_dieases_classification_images/", file_path="dataset/eye_diseases_classification_val.csv",
                                    imagetype="png", train=False)

    Validation_dataloader = DataLoader(dataset=Validation_dataset, batch_size=64, shuffle=True,
                                            num_workers=1, pin_memory=True)
    
    
    # Testing_dataset = Eye_Dataset(images_path="dataset/ODIR_5k_images/", file_path="dataset/ODIR-5k_test.csv",
    #                                 imagetype="jpg", train=False)

    # Testing_dataloader = DataLoader(dataset=Testing_dataset, batch_size=64, shuffle=True,
    #                                         num_workers=1, pin_memory=True)
    
    optimizer = torch.optim.Adam(ark_model.parameters(), lr=1e-4)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    init_loss = 999999
    best_val_loss = init_loss
    save_model_path = os.path.join("./ARK_Finetuned_Models","EDC")
    output_path = os.path.join("./Outputs/")

    test_results, test_results_teacher = [], []
    losses_cls_train = MetricLogger('Loss_cls_train', ':.4e')
    losses_cls_valid = MetricLogger('Loss_cls_valid', ':.4e')
    
    print("Start training")
    for ep in range(0, 30):
        for i, (sample, _, target) in enumerate(Training_dataloader):
            sample, target = sample.to(device), target.to(device)
            probs = ark_model(sample, 0)
            loss = criterion(probs[1], target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            losses_cls_train.update(loss.item(), sample.size(0))
        
        print("Start validation at epoch: ", ep)
        loss_valid = 0.0
        for i, (sample, _, target) in enumerate(Validation_dataloader):
            sample, target = sample.to(device), target.to(device)
            with torch.no_grad():
                probs = ark_model(sample, 0)
            loss = criterion(probs[1], target)
            loss_valid += loss.item()
        loss_valid /= i
        if loss_valid < best_val_loss:
            print("save best model at ", best_val_loss)
            best_val_loss = loss_valid
            #torch.save(ark_model, output_path + "ARK_best.pth.tar")
            save_checkpoint({
            'epoch': ep,
            'lossMIN': None,
            'state_dict': ark_model.state_dict(),
            'teacher': None,
            'optimizer': optimizer.state_dict(),
            'scheduler': None,
            },  filename="For you"+str(ep))