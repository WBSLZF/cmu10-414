import sys

import numpy

from python.needle.data import MNISTDataset, DataLoader

sys.path.append('../python')
import python.needle as ndl
import python.needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    FFN = nn.Sequential(nn.Linear(dim,hidden_dim),
                         norm(hidden_dim),
                         nn.ReLU(),
                         nn.Dropout(drop_prob),
                         nn.Linear(hidden_dim,dim),
                         norm(dim),
                         )
    return nn.Sequential(nn.Residual(FFN),nn.ReLU())
    ### END YOUR SOLUTION



def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    resnet = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                           *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                           nn.Linear(hidden_dim, num_classes))
    return resnet
    ### END YOUR SOLUTION
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    total_loss = []
    total_erro = 0.0
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
        for sample,target in dataloader:
            sample  = sample.reshape((sample.shape[0],-1))

            output = model(sample)
            logits = numpy.argmax(output.numpy(),axis=1)
            loss = nn.SoftmaxLoss()(output,target)

            opt.reset_grad()
            loss.backward()
            opt.step()

            total_erro += (logits != target.numpy()).sum()
            total_loss.append(loss.numpy())


    else :
        model.eval()
        for sample,target in dataloader:
            sample  = sample.reshape((sample.shape[0],-1))
            output = model(sample)

            logit = numpy.argmax(output.numpy(),axis=1)
            erro = (logit != target.numpy()).sum()
            total_loss.append(nn.SoftmaxLoss()(output,target).numpy())
            total_erro+= erro
    return total_erro/len(dataloader.dataset), np.mean(total_loss)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    ### BEGIN YOUR SOLUTION
    resnet = MLPResNet(28*28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz",
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, resnet, opt)
    test_err, test_loss = epoch(test_loader, resnet, None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
