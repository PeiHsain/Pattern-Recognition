# -*- coding: utf-8 -*-
"""
2022PR homework5 created by Pei Hsuan Tsai.
    implement the deep neural network by any deep learning framework,
    then train the DNN model by the Cifar-10 dataset.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import transforms
from sklearn.metrics import accuracy_score

# Check the GPU is avialible, else use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def Horizontal(img):
    'Horizontal flip the image randomly by probability p.\nOutput : horizontal image'
    horizental_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.7)
    ])
    img_pil = transforms.ToPILImage()(img)
    new_img = horizental_tf(img_pil)
    return new_img


def Rotation(img):
    'Rotate image randomly by the probability p.\nOutput : rotated image'
    rotation_tf = transforms.Compose([
        transforms.RandomRotation(degrees=40, fill=1)
    ])
    img_pil = transforms.ToPILImage()(img)
    new_img = rotation_tf(img_pil)
    return new_img


def Color(img):
    'Color Jitter.\nOutput : jittered image'
    color_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ColorJitter(brightness=0, contrast=(0, 5), saturation=(0, 2), hue=(-0.2, 0.2))
    ])
    img_pil = transforms.ToPILImage()(img)
    new_img = color_tf(img_pil)
    return new_img


def DataAugmentation(x, y):
    'Add horizontal and rotation data from original dataset.\nOutput : new dataset'
    add_size = 4
    data_num = len(x) * add_size
    new_x = np.zeros((data_num, 32, 32, 3), dtype=np.uint8) # max value = 255 (0b1111 1111)
    new_y = np.zeros((data_num, 1), dtype=np.uint8) # class 0-9
    for i in range(0, data_num, add_size):
        old_idx = i // add_size
        x_h = Horizontal(x[old_idx])
        x_r = Rotation(x[old_idx])
        x_c = Color(x[old_idx])
        
        new_x[i] = x[old_idx]
        new_x[i+1] = np.asarray(x_h)
        new_x[i+2] = np.asarray(x_r)
        new_x[i+3] = np.asarray(x_c)
        new_y[i:i+add_size] = y[old_idx]
    return new_x, new_y


class ImageDataset(Dataset):
    'Set image data as pytorch dataset'
    def __init__(self, x, y):
        # 32x32 RGB images in 10 classes, softmax -> element value in range 0-1
        x = x.astype('float32')
        x /= 255
        # Input (data_num, 32, 32, 3) -> output (data_num, 3, 32, 32)
        self.x_data = torch.from_numpy(x).permute(0, 3, 1, 2)
        self.y_data = torch.from_numpy(y)

    def __len__(self):
        # get lenght of x
        return len(self.x_data)

    def __getitem__(self, idx):
        # get data of x and y accroding to the index
        return self.x_data[idx], self.y_data[idx]


class NeuralNetwork(nn.Module):
    'CNN model'
    def __init__(self):
      # model layer
        # Input image (3, 32, 32) -> 3 RGB chennels, (32, 32) image size
        super(NeuralNetwork, self).__init__()
        # 1 convolution layer
        self.conv1 = self.conv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) # output = (32, 32, 32)
        # 2 convolution layer
        self.conv2 = self.conv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1) # output = (32, 32, 32)
        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2) # output = (32, 16, 16)
        # clear channels independantly with probability p
        self.dropout1 = nn.Dropout2d(p=0.2)
        # 3 convolution layer
        self.conv3 = self.conv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # output = (64, 16, 16)
        # 4 convolution layer
        self.conv4 = self.conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) # output = (64, 16, 16)
        # pooling layer -> output = (64, 8, 8)
        # clear channels independantly with probability p
        self.dropout2 = nn.Dropout2d(p=0.3)
        # 5 convolution layer
        self.conv5 = self.conv(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # output = (128, 8, 8)
        # 6 convolution layer
        self.conv6 = self.conv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) # output = (128, 8, 8)
        # pooling layer -> output = (128, 4, 4)
        # clear channels independantly with probability p
        self.dropout3 = nn.Dropout2d(p=0.4)

        # fully connected
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=512)
        self.relu = nn.ReLU()
        # clear channels independantly with probability p
        self.dropoutf = nn.Dropout2d(p=0.25)
        # last fully connected, output should be same as class_num
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        # soft max
        self.sm = nn.Softmax(dim=1)

    def conv(self, in_channels, out_channels, kernel_size, stride, padding):
        seq_modules = nn.Sequential(
            # convolution layer -> normalize
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return seq_modules

    def forward(self, x):
      # model structure
        # first convolution layer
        x = self.dropout1(self.pool(self.conv2(self.conv1(x))))
        # second convolution layer
        x = self.dropout2(self.pool(self.conv4(self.conv3(x))))
        # third convolution layer
        x = self.dropout3(self.pool(self.conv6(self.conv5(x))))
        # flatten all dimensions except batch. CNN -> FCN
        x = torch.flatten(x, 1)
        # fully connected layers
        x = self.dropoutf(self.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.sm(x)
        return x


if __name__ == "__main__":
    # Fixed the seed of random variables
    SEED = 555
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Hyper-parameters
    # It's a multi-class classification problem
    CLASS_INDEX = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                'dog': 5, 'frog': 6,'horse': 7,'ship': 8, 'truck': 9}
    CLASS_NUM = 10
    BATCH_SIZE = 50
    LEARN_RATE = 0.0001
    EPOCH = 0  # times of training model

    # Load data
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")

    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    # Ignore the filter warings
    import warnings
    warnings.filterwarnings('ignore')

    # Data augmentation
    new_x, new_y = DataAugmentation(x_train, y_train)

    # Data preprocess. Prepare image data for learning
    train_dataset = ImageDataset(new_x, new_y)
    test_dataset = ImageDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build the model
    model = NeuralNetwork().to(device)
    summary(model, (3, 32, 32))

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    # Training
    model.train() # training mode
    for i in range(EPOCH):
        total_loss = 0
        for x, y in train_loader:
            # put data into gpu(cpu) enviroment
            image = x.to(device)
            label = y.to(device)
            # remove one dimension
            label = label.squeeze()
            # initial optimizer, clear gradient
            optimizer.zero_grad()
            # train the model, forward -> backward -> update
            # put data into the model to do forward propagation
            output = model(image)
            # calculate loss
            loss_value = criterion(output, label)
            # use loss to do backward propagation and compute the gradient
            loss_value.backward()
            # do gradient descent
            optimizer.step()
            total_loss += loss_value
        print(f'Train Loss of Epoch {i+1}: {total_loss}')

    # Testing
    y_pred = []
    model.eval()  # evaluate mode
    for x, y in test_loader:
        # put data into gpu(cpu) enviroment
        image = x.to(device)
        label = y.to(device)
        # put data into the model to predict
        pred = model(image)
        # argmax to find the predict class with highest probability.
        pred_n = np.argmax(pred.to('cpu').detach().numpy(), axis=1)
        y_pred = np.concatenate((y_pred, pred_n))

    assert y_pred.shape == (10000,)

    y_test = np.load("y_test.npy")
    print("Accuracy of my model on test set: ", accuracy_score(y_test, y_pred))

    # Save model parameters
    torch.save(model.state_dict(), 'A102548_model_save.pt')