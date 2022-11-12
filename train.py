import torch
from .trainutils import *
from .models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_dataloader, test_dataloader, lr=5e-4, epochs=5, log_every=10, wandb_flag=True, verbose=True,
            save_model=False, save_loc=""):
    
    if wandb_flag:
        import wandb
        run = wandb.init()
        lr = run.config.lr
        epochs = run.config.epochs
        log_every = run.config.log_every
        
        wandb.watch(model, log_freq=100)

    if save_model:
        if wandb_flag:
            save_loc = f"{save_loc}_{run.config.lr}_{run.config.epochs}"
        else:
            save_loc = f"{save_loc}_model"

    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_meter = AverageMeter()

    train_ld = {'loss' : []}
    val_ld = {}

    lossdict = evaluate(test_dataloader, model)

    for epoch in range(epochs):
            for i, (x, y) in enumerate(train_dataloader):
                model.train()
                x = x.to(device)
                y = y.to(device)

                model.forward(x)
                loss = get_loss(model, x, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item())
                train_ld['loss'].append(loss.item())
                
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