from tqdm import tqdm
import pickle
import wandb
import torch

def train_model(config):
    wandb.login()
    run=wandb.init(
        project = 'Acivity3',
        config = {
            'optimizer': config.optimizer,
            'model': config.model
        }
    )
    for _ in range(config.epochs):
        best_loss=float('inf')
        losses = 0
        for X, y in tqdm(config.dataloader):
            X=X.to(config.device)
            y=y.to(config.device)
            output = config.model(X)
            loss = config.criterion(output, y)
            config.optimizer.zero_grad()
            loss.backward()
            config.optimizer.step()
            acc = config.metric(output, y)
            losses += loss.item()
        losses = losses/config.dataloader.dataset.__len__()
        if best_loss<losses:
            config.model.load_state_dict(torch.load(config.model_file))
        else:
            torch.save(config.model.state_dict(), config.model_file)
        acc = config.metric.compute()
        print(f'Loss: {losses:0.3g} | Accuracy: {acc:0.3g}')
        wandb.log({'accuracy': acc, 'loss':losses})
        config.metric.reset()
    with open(config.model_file, 'wb') as file:
        pickle.dump(config.model,file)