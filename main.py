from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import torchmetrics
from torch import nn, optim
from tqdm import tqdm
import os

dirs = ['datasets', 'models']
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)
train_dataset = STL10(root='datasets', split='train',
                      download=True, transform=ToTensor())
test_dataset = STL10(root='datasets', split='test',
                     download=True, transform=ToTensor())
_, dataset = random_split(test_dataset, [0.9, 0.1])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class resnet50_TL(nn.Module):
    def __init__(self):
        super(resnet50_TL, self).__init__()
        self.resnet = resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.dropout=nn.Dropout(0.2)
        self.linear = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = self.linear(x)
        return x

if __name__=='__main__':
    model = resnet50_TL()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    metric = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=10)
    epochs = 5
    for epoch in range(epochs):
        losses = 0
        for X, y in tqdm(dataloader):
            output = model(X)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = metric(output, y)
            losses += loss.item()
        losses = losses/dataloader.dataset.__len__()
        acc = metric.compute()
        print(f'Loss: {losses:0.3g} | Accuracy: {acc:0.3g}')
        metric.reset()