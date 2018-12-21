import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import json
import os

"""
=======================================================
                    Load the data
=======================================================
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_workers = 1
batch_size = 1

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
val_data = datasets.ImageFolder(valid_dir, transform=data_transforms['val'])
test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = torch.utils.data.DataLoader(train_data,
                                           shuffle=True,
                                           batch_size=batch_size,
                                           num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_data,
                                         shuffle=True,
                                         batch_size=batch_size,
                                         num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          num_workers=num_workers)


"""
=======================================================
                Load models and  
=======================================================
"""


# cat_to_name  :  dict from number of class to flower name
# n_classes    :  102
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
n_classes = len(cat_to_name)


load_dir = os.path.join(os.getcwd(), 'weights')
resnet_file = 'resnet904.pt'
densenet_file = 'densenet907.pt'

resnet_path = os.path.join(load_dir, resnet_file)
densenet_path = os.path.join(load_dir, densenet_file)

resnet = torch.load(resnet_path, map_location=device)
densenet = torch.load(densenet_path, map_location=device)

criterion = nn.NLLLoss()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def consolidate_eval(model):

    print("Consolidation part...")

    correct_acc = {k: 0 for k in range(1, n_classes + 1)}
    correct_ps = {k: [] for k in range(1, n_classes + 1)}

    model.eval()

    for images, labels in val_loader:
        """
        images: torch(batch_size, 3, 224, 224)
        labels: torch(batch_size)
        !!!classes start from 0!!!
        """
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)  # torch(batch_size, n_classes)
        ps = torch.exp(outputs)  # torch(batch_size, n_classes)
        top_ps, top_class = ps.topk(1, dim=1)  # both torch(batch_size, 1)
        equality = top_class == labels.view(*top_class.shape)  # torch(batch_size, 1)

        for top_p, top_c, eq in zip(top_ps, top_class, equality):
            if eq.item() == 1:
                correct_acc[top_c.item() + 1] += 1
                correct_ps[top_c.item() + 1].append(top_p.item())

    for k in correct_ps.keys():
        correct_ps[k] = sum(correct_ps[k]) / (len(correct_ps) if correct_ps else 1)

    return correct_ps, correct_acc


def test(model, criter):

    model.eval()
    test_loss, test_accuracy = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criter(outputs, labels)
        test_loss += loss.item()

        ps = torch.exp(outputs)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.sum(equality.type(torch.FloatTensor))

    print(f"Test loss: {test_loss/len(test_loader):.4f} |"
          f"Test accuracy: {test_accuracy/len(test_loader):.4f} |")


res_mean_ps, res_acc = consolidate_eval(resnet)
dense_mean_ps, dense_acc = consolidate_eval(densenet)
influence = {k: None for k in range(1, n_classes + 1)}
# 'resnet' or 'densenet'
for k in influence.keys():
    if res_acc[k] == dense_acc[k]:
        influence[k] = 'resnet' if res_mean_ps[k] >= dense_mean_ps[k] else 'densenet'
    else:
        influence[k] = 'resnet' if res_acc[k] > dense_acc[k] else 'densenet'


def multi_test(resnet, densenet, influence, criter):

    resnet.eval()
    densenet.eval()

    mt_loss, mt_accuracy = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        res_outputs = resnet(images)  # torch(batch_size, n_classes)
        den_outputs = densenet(images)  # torch(batch_size, n_classes)

        res_ps = torch.exp(res_outputs)  # torch(batch_size, n_classes)
        den_ps = torch.exp(den_outputs)  # torch(batch_size, n_classes)

        res_top_ps, res_top_class = res_ps.topk(1, dim=1)  # both torch(batch_size, 1)
        den_top_ps, den_top_class = den_ps.topk(1, dim=1)  # both torch(batch_size, 1)

        l = images.size(0)

        outputs = torch.zeros(l, n_classes)

        for i in range(l):
            pred_res_class = res_top_class[i].item() + 1
            pred_den_class = den_top_class[i].item() + 1
            if influence[pred_res_class] == 'resnet' and influence[pred_den_class] == 'densenet':
                outputs[i] = res_outputs[i] if res_top_ps[i].item() >= den_top_ps[i].item() else den_outputs[i]
            elif influence[pred_res_class] != 'resnet' and influence[pred_den_class] == 'densenet':
                outputs[i] = den_outputs[i]
            elif influence[pred_res_class] == 'resnet' and influence[pred_den_class] != 'densenet':
                outputs[i] = res_outputs[i]
            elif influence[pred_res_class] != 'resnet' and influence[pred_den_class] != 'densenet':
                outputs[i] = res_outputs[i] if res_top_ps[i].item() >= den_top_ps[i].item() else den_outputs[i]

        loss = criter(outputs, labels)
        mt_loss += loss.item()

        ps = torch.exp(outputs)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        mt_accuracy += torch.sum(equality.type(torch.FloatTensor))

    print(f"Test loss: {mt_loss/len(test_loader):.4f} |"
          f"Test accuracy: {mt_accuracy/len(test_loader):.4f} |")


test(resnet, criterion)
print("\n"*10)
test(densenet, criterion)
print("\n"*10)
multi_test(resnet, densenet, influence, criterion)
