import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
import json
import PIL
from PIL import Image


def load_data(data_dir):
    '''
    Arguments : the path of the dataset
    Returns : The loaders for the train, validation and test datasets
    This function receives the location of the daatset, applies the necessery transformations (rotations,flips,normalizations and crops) and converts the images to tensor in order to be able to be fed into the neural network
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    data_transforms = {
        'test':transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
               ]),

        'train':transforms.Compose([
                transforms.RandomRotation(40),  # copy12: change 10 to 40
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ]),

        'valid':transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
                ])
        }
    # TODO: Load the datasets with ImageFolder
    image_datasets = {x:datasets.ImageFolder(data_dir+x, transform=data_transforms[x])
             for x in ['train','test','valid']}
    # TODO: Using the image datasets and the trainforms, define the dataloaders
 
    batch_size=64
    num_workers=0
    dataloaders ={
        x:torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
        shuffle=True,num_workers=num_workers)
        for x in ['train','test','valid'] 
    }
    return image_datasets, dataloaders

def model_setup(arch,hidden_units,flower_species,gpu):
    '''
    Arguments: The architecture for the network(densenet121,densenet161), the hyperparameters for the network (hidden units) and whether to use gpu or not
    Returns: Set up model for the Training
    '''
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
    else:
        print("{} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(arch))

    for param in model.parameters():
        param.requires_grad=False

    model.classifier= nn.Sequential(
                    nn.Linear(model.classifier.in_features,hidden_units),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_units, flower_species)
                    )
    use_cuda = torch.cuda.is_available() and gpu
    if use_cuda:
        model = model.cuda()

    return model

def optimizer_setup(model,lr):
    '''
    Arguments: The amodel and learning rate
    Returns: Set up criterion and optimizer for the Training
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr)
    # Decay LR by a factor of 0.1 every 4 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    return criterion, optimizer

def train(n_epochs, loaders, model, optimizer, criterion, gpu, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    print("start")
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        valid_accuracy=0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            use_cuda = torch.cuda.is_available() and gpu
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                    # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        valid_correct=0
        valid_total=0
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            valid_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            valid_total += data.size(0)
            
        # calculate average losses
        train_loss = train_loss/len(loaders['train'].dataset)
        valid_loss = valid_loss/len(loaders['valid'].dataset)
        valid_accuracy=valid_correct/valid_total    
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f} {}/{}'.format(
            epoch, 
            train_loss,
            valid_loss,
            valid_accuracy, valid_correct, valid_total
            ))
        
        ## TODO: save the model if validation loss has decreased
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    # return trained model
    return model

def test(loaders, model, criterion, gpu):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        use_cuda = torch.cuda.is_available() and gpu
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        #print(tuple(zip(pred.data.view_as(target).cpu().numpy(),target.data.cpu().numpy())))
        #print(target)
        # compare predictions to true label
        #print(target.data.cpu().numpy()[pred.data.view_as(target).cpu().numpy()!=target.data.cpu().numpy()])
        #print(pred.data.view_as(target).cpu().numpy()[pred.data.view_as(target).cpu().numpy()!=target.data.cpu().numpy()])
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

#model.load_state_dict(torch.load('model_transfer_densenet161.pt'))
# call test function    
#test(dataloaders, model, criterion_transfer, use_cuda)

def save_checkpoint(path,arch,image_datasets,model,hidden_units,flower_species):
    model.class_to_idx = image_datasets['train'].class_to_idx
    torch.save({'structure' :arch,
            'state_dict':model.state_dict(),
            'hidden_units':hidden_units,
            'flower_species':flower_species,
            'class_to_idx':model.class_to_idx},
            path)

def load_model(path,flower_species):
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    if structure=='densenet161':
        model = models.densenet161(pretrained=True)
    elif structure=='densenet201':
        model = models.densenet201(pretrained=True)
    else:
        print("Wrong Model")
    for param in model.parameters():
        param.requires_grad=False
    model.classifier= nn.Sequential(
                    nn.Linear(model.classifier.in_features,checkpoint['hidden_layer']),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(checkpoint['hidden_layer'], flower_species)
                    )
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
    img_np=np.array(img_pil)
    #print(img_np.shape)
    #img_np=img_np.transpose(2,0,1)
    #img_pil=Image.fromarray(img_np)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    #print(img_tensor.shape)
    
    return img_tensor

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
def predict(image_path, model, gpu,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    use_cuda=torch.cuda.is_available() and gpu
    if use_cuda:
        model.cuda()
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0).float()
    #img_torch = img_torch.float()
    model.eval()
    if use_cuda:
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)       
    probability = F.softmax(output.data,dim=1)
    #[np.asscalar(index.cpu().data[0].numpy())]
    value,index=torch.topk(probability,topk)
    
    return value[0].cpu().data.numpy(),index[0].cpu().data.numpy()
    