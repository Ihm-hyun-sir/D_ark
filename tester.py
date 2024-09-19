from utils import MetricLogger, ProgressLogger
import time
import torch
import model 
import os
from dataloader import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    testing_model = torch.load("Outputs/ODIR_MURED.tar")
    

  
    Testing_dataset = Eye_Dataset(images_path="dataset/eye_dieases_classification_images/", file_path="dataset/eye_diseases_classification_test.csv",
                                    imagetype="png", train=False)

    Testing_dataloader = DataLoader(dataset=Testing_dataset, batch_size=64, shuffle=True,
                                            num_workers=1, pin_memory=True)
    """
    
    Testing_dataset = Eye_Dataset(images_path="dataset/ODIR_5k_images/", file_path="dataset/ODIR-5k_test.csv",
                                    imagetype="jpg", train=False)

    Testing_dataloader = DataLoader(dataset=Testing_dataset, batch_size=64, shuffle=True,
                                            num_workers=1, pin_memory=True)
      """
    with torch.no_grad():
        testing_model.eval()
        outputs = []
        targets = []
        outAUROC = []

        for i, (sample, _, target) in enumerate(Testing_dataloader):
            print(f"Sample {i}")
            sample, target = sample.to(device), target.to(device)
            targets.append(target.cpu().numpy())
            output = F.softmax(testing_model(sample, 0)[1], dim=1)
            #output = torch.sigmoid(testing_model(sample, 0)[1])
            #output = testing_model(sample, 0)[1]
            outputs.append(output.cpu().numpy())  # assuming output shape is [batch_size, num_classes]

        targets = np.concatenate(targets, axis=0)
        outputs = np.concatenate(outputs, axis=0)

        print(targets)
        print(outputs)
        np.savetxt('labels.txt',targets,fmt='%d')
        np.savetxt('outputs.txt',outputs,fmt='%f')

        for i in range(3):
            outAUROC.append(roc_auc_score(targets[:, i], outputs[:, i]))
        
        print(outAUROC)
        print("Mean : ", np.mean(outAUROC))
