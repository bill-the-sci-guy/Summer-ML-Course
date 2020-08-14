import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import os
from torchsummary import summary


# test, train draw curves
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# hyper parameters
num_epochs = 1
# How many times you go through your data for optimization by gradient descent

lr = 0.001
# step length in approaching min in gradient descent

store_dir = 'C:\\MLDATA\\a.pth'
# directory to store the best neural network


# Loader
def load_dataset():
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    # dictionary to specify how to make normalized data by .ToTensor() and .Normalize()

    data_dir = 'C:\\MLDATA'
    # directory where data is stored
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    # create a dataset dict {{}}
    data_loaders = {x: data.DataLoader(image_datasets[x], batch_size=256, num_workers=16, shuffle=True) for x in
                    ['train', 'test']}
    # create a dataloader dict
    data_size = {x: len(image_datasets[x]) for x in ['train', 'test']}
    # how many data are processed

    return data_loaders, data_size


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        # already trained by ImageNet
        for param in self.model.parameters():
            param.requires_grad = True
        # change the weight of the model

        self.model.fc = nn.Linear(self.model.fc.in_features, 2, bias=True)

    def forward(self, x):
        y = self.model(x)
        return y
    # y is the prediction


def train(data_loader, data_size):
    bestacc = 0
    model = Model()
    model = model.cuda()  # Moves all model parameters and buffers to the GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.model.parameters(), lr=lr, momentum=0.9)
    # Train all params
    # check momentum

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        tqdm.write('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # training
        for mode in ['train', 'test']:
            if mode == 'train':
                model.train = True
                tot_loss = 0.0
                optimizer.zero_grad()
            else:
                model.train = False
            correct = 0

            for batch in tqdm(data_loader[mode]):
                (inputs, labels) = batch
                (inputs, labels) = (Variable(inputs.cuda()), Variable(labels.cuda()))
                # turns batch into trainable

                outputs = model(inputs)
                # obtain score, nn.module calls forward

                _, preds = torch.max(outputs.data, 1)
                # print(outputs.data)
                # print(preds)
                # print(_)
                if mode == 'train':
                    loss = criterion(outputs, labels)
                    loss.backward()  #
                    optimizer.step()  # optimizer.step() refresh parameters in the network
                    tot_loss += loss.data
                # .to(torch.float32)
                correct += torch.sum(preds == labels.data).to(torch.float32)
            ### Epoch info ####
            if mode == 'train':
                epoch_loss = tot_loss / data_size[mode]
                print('train loss: ', epoch_loss)
            epoch_acc = correct / data_size[mode]

            if mode == 'test' and epoch_acc > bestacc:
                bestacc = epoch_acc
                torch.save(model.state_dict(), store_dir)
                print(mode + ' best acc: ', bestacc)
            print(mode + ' acc: ', epoch_acc)

    return model


def mains():
    data_loader, data_size = load_dataset()
    model = train(data_loader, data_size)
    summary(model, (3, 64, 64), batch_size=256)


if __name__ == '__main__':
    mains()
