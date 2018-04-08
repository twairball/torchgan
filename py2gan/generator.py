import torch
import torchvision
import torchvision.transforms as transforms

import os

dataset_mapping = dict(cifar10=torchvision.datasets.CIFAR10, 
                       mnist=torchvision.datasets.MNIST, 
                       fashion=torchvision.datasets.FashionMNIST)

def get_dataset_iterator(dataset_name='mnist', batch_size=32, data_path='./data', num_workers=2):
    ds = dataset_mapping[dataset_name]
    
    # fashion and mnist paths collide
    _data_path = data_path
    if dataset_name == 'fashion':
        _data_path = os.path.join(_data_path, 'fashion')
        print("fashion: %s" % _data_path)
        
    # normalize to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # train
    trainset = ds(root=_data_path, train=True,
                 download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    # test
    testset = ds(root=_data_path, train=False,
                 download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    
    # get some random training images
    train_iter = iter(trainloader)
    test_iter = iter(testloader)
    
    return train_iter, test_iter
