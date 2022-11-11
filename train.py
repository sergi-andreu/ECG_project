def train(model, lr=5e-4):

    optimizer = torch.optim.Adam(model.parameters(), lr)

    train_ld = {'loss' : []}
    val_ld = {}

    epochs = 5

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
            
            print("Eval at epoch ", epoch)
            lossdict = evaluate(test_dataloader, model)
            val_ld = update_lossdict(val_ld, lossdict)

            loss_meter.reset()

    return train_ld, lossdict, val_ld