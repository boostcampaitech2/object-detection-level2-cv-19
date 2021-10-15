import os
import random
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class Trainer:
    
    def __init__(self, weight_save_path, seed=42):

        self.weight_save_path = weight_save_path
        os.makedirs(weight_save_path, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = seed        

    def train(self, model, dataloaders, criterion, optimizer, max_epoch, patience, save_name='best', sub_dir=None):        

        save_dir = os.path.join(self.weight_save_path, sub_dir) if sub_dir is not None else self.weight_save_path
        os.makedirs(save_dir, exist_ok=True)
        
        best_score = 0
        best_epoch = 0
        patience_cnt = 0
        
        for epoch in range(1, max_epoch+1):
            print('Epoch {}/{}'.format(epoch, max_epoch))
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_cnt = 0
                y_true, y_pred = [], []

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        y_true.extend(labels.tolist())
                        y_pred.extend(preds.tolist())

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    running_cnt += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                epoch_loss = running_loss / running_cnt
                epoch_acc = running_corrects.double() / running_cnt
                f1 = f1_score(y_true, y_pred, average='macro')

                print('{} Epoch: {} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch, epoch_loss, epoch_acc, f1))

                if phase == 'valid':
                    if f1 > best_score:
                        best_score = f1
                        best_epoch = epoch
                        patience_cnt = 0
                        torch.save(model.state_dict(), os.path.join(save_dir, f'{save_name}.pt'))
                    else:
                        if patience_cnt == patience:
                            print()
                            print('Training complete!')
                            print('Best f1 score {:.4f} at epoch {}'.format(best_score, best_epoch))
                            print()
                            return
                        patience_cnt += 1

            print()

        print('Training complete!')
        print('Best f1 score {:.4f} at epoch {}'.format(best_score, best_epoch))
        print()