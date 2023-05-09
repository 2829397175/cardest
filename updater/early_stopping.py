import numpy as np
import torch
import os



class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, save_path, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss,model,epoch_num):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
    def save_checkpoint(self, val_loss, model,epoch_num,save_model):
        '''Saves model when validation loss decrease.'''
        print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = self.save_path+"_epoch{}_model{:.3f}_best.pth".format(epoch_num,val_loss)
        if (save_model):
            try:
                torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
            except:
                print(f"save err: path:{path}")
        self.val_loss_min = val_loss
        return path