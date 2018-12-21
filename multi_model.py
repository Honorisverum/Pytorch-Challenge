import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.optim import lr_scheduler
import json
import os

"""
=======================================================
                    Load the data
=======================================================
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_workers = 1
batch_size = 30

data_dir = 'multi_model_data'
train_dir = os.path.join(os.getcwd(), data_dir, 'train')
valid_dir = os.path.join(os.getcwd(), data_dir, 'valid')
test_dir = os.path.join(os.getcwd(), data_dir, 'test')

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_data = datasets.ImageFolder(train_dir, transform=data_transforms['val'])
test_data = datasets.ImageFolder(train_dir, transform=data_transforms['test'])

train_loader = torch.utils.data.DataLoader(train_data,
                                           shuffle=True,
                                           batch_size=batch_size,
                                           num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(train_data,
                                         shuffle=True,
                                         batch_size=batch_size,
                                         num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(train_data,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          num_workers=num_workers)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

n_classes = len(cat_to_name)

"""
=======================================================
        Building and training the classifiers
        1)ResNet-152        : n_features = 2048    : name = fc
        2)Densenet-161      : n_features = 2208    : name = classifier
        3)Inception v3      : n_features = 2048    : name = fc
        4)
=======================================================
"""


def train(model, lr, epochs, dropout, criter, save_filename):
    for name, _ in model.named_children():
        pass
    in_features = getattr(model, name).in_features
    save_path = os.path.join(os.getcwd(), 'weights', save_filename)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(in_features, in_features // 2),
                               nn.ReLU(),
                               nn.Dropout(p=dropout),
                               nn.Linear(in_features // 2, in_features // 2 // 2),
                               nn.ReLU(),
                               nn.Dropout(p=dropout),
                               nn.Linear(in_features // 2 // 2, n_classes),
                               nn.LogSoftmax(dim=1))

    setattr(model, name, classifier)
    model = model.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           betas=(0.9, 0.999))

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=7,
                                    gamma=0.1)
    prev_val_accuracy = None

    for epoch in range(1, epochs + 1):

        print(f"Epoch {epoch}/{epochs}", end="\n" + "=" * 30 + "\n")
        scheduler.step()
        model.train()
        running_loss, running_acc = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criter(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            ps = torch.exp(outputs)
            _, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            running_acc += torch.mean(equality.type(torch.FloatTensor))

        model.eval()
        val_loss, val_accuracy = 0, 0

        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criter(outputs, labels)
            val_loss += loss.item()

            ps = torch.exp(outputs)
            top_ps, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            val_accuracy += torch.mean(equality.type(torch.FloatTensor))

        if prev_val_accuracy is None or val_accuracy > prev_val_accuracy:
            torch.save(model, save_path)
        prev_val_accuracy = val_accuracy

        print(f"Epoch {epoch} |"
              f"Train loss: {running_loss/len(train_loader):.4f} |"
              f"Train accuracy: {running_acc/len(train_loader):.4f} |"
              f"Validation loss: {val_loss/len(val_loader):.4f} |"
              f"Validation accuracy: {val_accuracy/len(val_loader):.4f} |")


resnet = torchvision.models.resnet152(pretrained=True)
densenet = torchvision.models.densenet161(pretrained=True)
inception = torchvision.models.inception_v3(pretrained=True)
learning_rate = 0.001
n_epochs = 25
p_dropout = 0.2
criterion = nn.NLLLoss()

train(resnet, learning_rate, n_epochs, p_dropout, criterion, 'resnet.pt')
print("\n"*10)
train(densenet, learning_rate, n_epochs, p_dropout, criterion, 'densenet.pt')
print("\n"*10)
train(inception, learning_rate, n_epochs, p_dropout, criterion, 'inception.pt')
