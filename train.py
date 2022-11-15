"""
Code for defining a training function.

Author = @Sergi-Andreu


This is a quite-simple script of a training function, calling Weights&Biases (if needed) for experiment tracking.

It is recommended that this is run in wandb (set wandb_flag=True) since it has not been tested much otherwise,
and there may be errors associated to logging the training progress.

"""

# Import packages
import torch
from .trainutils import *  # Import the train utils
"""
The train utils could be defined here (since they are not many)
but it has been done this way to simplify this script
"""

from .models import * # Import the models (Resnet18 and Resnet34)
"""
Indeed, the model is called as a parameter for the training function,
so this may not be required. However, keeping it as it-is to avoid errors,
but this may be removed in the future
"""

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
This is a bit ugly (the device should be called as a parameter, 
or defined inside the train function.
Keeping it this way to avoid errors
"""


def train(model, train_dataloader, test_dataloader, lr=5e-4, epochs=5, log_every=10, wandb_flag=True, verbose=True,
            save_model=False, save_loc=""):
            
    """
    Parameters:
            - model: initialized model used for training
            - train_dataloader: dataloader of the training data
            - test_dataloader: dataloader of the test (indeed, validation) data
            - lr: learning rate
            - epochs: number of epochs to train for
            - log_every: period of logging the results to wandb
            - wandb_flag: flag to choose whether to log the results into weight and biases
                          The wandb dictionary should be initialized. This can be done for examples
                          using a sweep, as seen in the train notebook
            - verbose: flag used to print (or not) results during training
            - save_model: flag indicating if to save the model (only when training is finalized)
            - save_loc: save path of the model
    """
    
    # If wandb_flag, set the parameters to the wandb config parameters
    if wandb_flag:
        import wandb
        run = wandb.init()
        lr = run.config.lr
        epochs = run.config.epochs
        log_every = run.config.log_every
        
        # Watch the model to have data on the gradients, etc, on Weights&Biases
        wandb.watch(model, log_freq=100)

    if save_model:
        if wandb_flag:
            save_loc = f"{save_loc}_lr_{run.config.lr}_epochs_{run.config.epochs}"
        else:
            save_loc = f"{save_loc}_model"
            
            
    # Define the optimizer. Here, Adam is used (quite standard)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    # Define an average meter object for the loss (AverageMeter class defined in the trainutils)
    loss_meter = AverageMeter()

    # Initialize the log dictionaries
    train_ld = {'loss' : []}
    val_ld = {}

    # Evaluate the model before training
    lossdict = evaluate(test_dataloader, model)

    for epoch in range(epochs):
            for i, (x, y) in enumerate(train_dataloader):
                # Set model to train mode
                model.train()
                        
                # Move tensors to device (preferably cuda)
                x = x.to(device)
                y = y.to(device)
            
                # Compute model outputs
                model.forward(x)
                # Compute loss
                loss = get_loss(model, x, y)
                
                # Compute backward pass and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update loggers
                loss_meter.update(loss.item())
                train_ld['loss'].append(loss.item())
                
                # Log into Weights&Biases
                # It is not recommended to log arrays; therefore, splitting the arrays corresponding
                # to AUC and AUPRC per-label (NORM, MI, STTC, CD, HYP)
                if wandb_flag:
                    if i%log_every == 0:
                        lossdict = evaluate(test_dataloader, model)
                        wandb.log({"train_loss": loss.item(), 
                                    "NORM auc": lossdict["auc"][0],
                                    "MI auc": lossdict["auc"][1],
                                    "STTC auc": lossdict["auc"][2],
                                    "CD auc": lossdict["auc"][3],
                                    "HYP auc": lossdict["auc"][4],

                                    "NORM auprc": lossdict["auprc"][0],
                                    "MI auprc": lossdict["auprc"][1],
                                    "STTC auprc": lossdict["auprc"][2],
                                    "CD auprc": lossdict["auprc"][3],
                                    "HYP auprc": lossdict["auprc"][4],

                                    "val loss": lossdict["epoch_loss"],

                                    "epoch": epoch})
            
            if verbose: print("Eval at epoch ", epoch)
            if not wandb_flag:
                lossdict = evaluate(test_dataloader, model)
                val_ld = update_lossdict(val_ld, lossdict)

                if verbose:
                    print(lossdict)
            loss_meter.reset()

    if save_model:
        torch.save(model, save_loc)

    return train_ld, lossdict, val_ld
