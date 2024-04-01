from tqdm import tqdm

def train_model(config):
    for _ in range(config.epochs):
        losses = 0
        for X, y in tqdm(config.dataloader):
            output = config.model(X)
            loss = config.criterion(output, y)
            config.optimizer.zero_grad()
            loss.backward()
            config.optimizer.step()
            acc = config.metric(output, y)
            losses += loss.item()
        losses = losses/config.dataloader.dataset.__len__()
        acc = config.metric.compute()
        print(f'Loss: {losses:0.3g} | Accuracy: {acc:0.3g}')
        config.metric.reset()