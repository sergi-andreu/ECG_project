"""
Code for defining a training utils
Author = @Sergi-Andreu

This is done to simplify the train.py script
"""

# Import required packages
import torch
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

# Define the crossentropy loss function and device 
CEloss = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define averagemeter class
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Define a get_loss function
def get_loss(model, x_batch, y_batch):
    yhat = model.forward(x_batch)
    y_batch = y_batch.float()
    loss = CEloss(yhat, y_batch)
    return loss

# Define the evaluation function
# it computes the loss, and also AUC and AUPRC values for each label
def evaluate(dl, model):
    # Set model to eval mode
    model.eval()
    
    # Initialize loggers
    ld = {}
    loss = 0
    loss_obj = CEloss
    y_preds = []
    y_trues = []
    pbar = dl
    with torch.no_grad():
        for i, (xecg, y_) in enumerate(pbar):

            y_trues.append(y_.detach().numpy())

            xecg = xecg.to(device)
            y_ = y_.to(device)

            y_pred = model.forward(xecg)
            y_preds.append(y_pred.cpu().detach().numpy())

            l = loss_obj(y_pred, y_)
            loss += l.item()
    loss /= len(dl)
    (y_preds, y_trues) = (np.concatenate(y_preds,axis=0), np.concatenate(y_trues,axis=0))
    y_preds = np.squeeze(y_preds)
    y_trues = np.squeeze(y_trues)
    
    # Not very good practice here, but try to compute the metrics or else set to 0
    # this is not recommended 
    try:
        ld['epoch_loss'] = loss
        ld['auc'] = roc_auc_score(y_trues, y_preds, average=None)
        ld['auprc'] = average_precision_score(y_trues, y_preds, average=None)

    except ValueError:
        ld['epoch_loss'] = loss
        ld['auc'] = 0
        ld['auprc'] = 0
    return ld

# Define an update function for the loss dictionary
def update_lossdict(lossdict, update, action='append'):
    for k in update.keys():
        if action == 'append':
            if k in lossdict:
                lossdict[k].append(update[k])
            else:
                lossdict[k] = [update[k]]
        elif action == 'sum':
            if k in lossdict:
                lossdict[k] += update[k]
            else:
                lossdict[k] = update[k]
        else:
            raise NotImplementedError
    return lossdict

