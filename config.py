from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import torchmetrics
from torch import nn, optim
import torch
import os

class resnet50_TL(nn.Module):
    def __init__(self):
        super(resnet50_TL, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.dropout=nn.Dropout(0.2)
        self.linear = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = self.linear(x)
        return x
    
class config:
    epochs=10
    lr=0.1
    batch_size=32
    shuffle=True
    transform=ToTensor()
    test_frac=0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = STL10(root='datasets', split='train',
                      download=True, transform=transform)
    test_dataset = STL10(root='datasets', split='test',
                        download=True, transform=transform)
    _, dataset = random_split(test_dataset, [1-test_frac, test_frac])
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    dirs = ['datasets', 'models']
    model = resnet50_TL().to(device)
    metric = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    model_name='resnet_STL_TF'
    model_file = os.path.join('models', model_name)
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)