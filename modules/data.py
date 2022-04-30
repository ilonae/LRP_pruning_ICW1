'''
Codes for loading the MNIST data
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import albumentations as A
from albumentations.pytorch import ToTensorV2

import fnmatch
from tqdm import tqdm
import os,glob, random, cv2
from functools import lru_cache
from pathlib import Path
from skimage.io import imread as imread
from skimage.util import montage
import numpy as np
from skimage.color import label2rgb

from sklearn.model_selection import train_test_split

from torchsampler import ImbalancedDatasetSampler


import imageio
import numpy
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


class CQDataset(Dataset):
    def __init__(self, image_paths,class_to_idx, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx= class_to_idx
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image_filepath = os.path.normpath(image_filepath.replace("\\","/"))
        #print(image_filepath)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('\\')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
 

class ImageNetDatasetValidation(torch.utils.data.Dataset):
    """ This class represents the ImageNet Validation Dataset"""

    def __init__(self, trans=None, root_dir=None):

        # validation data paths
        if root_dir is None:
            self.baseDir = '/ssd7/skyeom/data/imagenet'
        else:
            self.baseDir = root_dir
        self.validationDir = os.path.join(self.baseDir, 'validation')
        self.validationLabelsDir = os.path.join(self.validationDir, 'info.csv')
        self.validationImagesDir = os.path.join(self.validationDir, 'images')

        # read the validation labels
        self.dataInfo = pd.read_csv(self.validationLabelsDir)
        self.labels = self.dataInfo['label'].values
        self.imageNames = self.dataInfo['imageName'].values
        self.labelID = self.dataInfo['labelWNID'].values

        self.len = self.dataInfo.shape[0]

        self.transforms = trans

    # we use an lru cache in order to store the most recent
    # images that have been read, in order to minimize data access times
    @lru_cache(maxsize=128)
    def __getitem__(self, index):

        # get the filename of the image we will return
        filename = self.imageNames[index]

        # create the path to that image
        imgPath = os.path.join(self.validationImagesDir, filename)

        # load the image an an numpy array (imageio uses numpy)
        img = imageio.imread(imgPath)

        # if the image is b&w and has only one colour channel
        # create two duplicate colour channels that have the
        # same values
        if (img.ndim == 2):
            img = numpy.stack([img] * 3, axis=2)

        # convert the array to a pil image, so that we can apply transformations
        img = Image.fromarray(img)

        # apply any transformations necessary
        if self.transforms is not None:
            img = self.transforms(img)

        # get the label
        labelIdx = int(self.labels[index])

        return img, labelIdx

    def __len__(self):
        return self.len


def get_mnist(datapath='../data/mnist/', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    # MNIST Dataset
    train_dataset = datasets.MNIST(root=datapath,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=download)

    test_dataset = datasets.MNIST(root=datapath,
                                  train=False,
                                  transform=transforms.ToTensor())
    return train_dataset, test_dataset


def get_cifar10(datapath='./data/', download=True):
    '''
    Get CIFAR10 dataset
    '''
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    # Cifar-10 Dataset
    train_dataset = datasets.CIFAR10(root=datapath,
                                     train=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         # transforms.Resize(256),
                                         # transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize
                                     ]),
                                     download=download)

    test_dataset = datasets.CIFAR10(root=datapath,
                                    train=False,
                                    transform=transforms.Compose([
                                        # transforms.Resize(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    return train_dataset, test_dataset


def get_imagenet(transform=None, root_dir=None):
    if root_dir is None:
        root_dir = '/ssd7/skyeom/data/imagenet'
    root_dir = Path(root_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

    # we can load the training data as an ImageFolder
    train = datasets.ImageFolder(root_dir / "train", train_transform)

    # but not the validation data
    # we use the custom made ImageNetDatasetValidation class for that
    val = ImageNetDatasetValidation(val_transform, root_dir=root_dir)

    return train, val


def get_cq500():

    
    data_path = '../ProtoPNet/datasets/cq500/'
    orig_dir= data_path + 'orig/'
    train_dir = data_path + 'train_cropped_augmented/'
    test_dir = data_path + 'test_cropped/'
    train_push_dir = data_path + 'train_cropped/'
    train_batch_size = 32
    test_batch_size = 30
    train_push_batch_size = 32
    img_size=224
    

    train_image_paths = [] #to store image paths in list
    classes = [] #to store class values


    #######################################################
    #               Define Transforms
    #######################################################

    #To define an augmentation pipeline, you need to create an instance of the Compose class.
    #As an argument to the Compose class, you need to pass a list of augmentations you want to apply. 
    #A call to Compose will return a transform function that will perform image augmentation.
    #(https://albumentations.ai/docs/getting_started/image_augmentation/)

    train_transforms = A.Compose(
        [
            A.SmallestMaxSize(max_size=350),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
            A.RandomCrop(height=256, width=256),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.SmallestMaxSize(max_size=350),
            A.CenterCrop(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    #1.
    # get all the paths from train_data_path and append image paths and class to to respective lists
    # eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
    # eg. class -> 26.Pont_du_Gard
    for data_path in glob.glob(orig_dir + '/*'):
        classes.append(data_path.split('\\')[-1]) 
        train_image_paths.append(glob.glob(data_path + '/*'))
        
    train_image_paths = [num for sublist in train_image_paths for num in sublist]
    #random.shuffle(train_image_paths)

    print('train_image_path example: ', train_image_paths[1])
    print('class example: ', classes[0])

    #2.
    # split train valid from train paths (80,20)
    train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

    #3.
    # create the test_image_paths
    test_image_paths = []
    for data_path in glob.glob(orig_dir + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = [num for sublist in test_image_paths for num in sublist]

    print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))
    #######################################################
    #      Create dictionary for class indexes
    #######################################################

    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    #print(class_to_idx)

    


    train_dataset = CQDataset(train_image_paths,class_to_idx, train_transforms)
    valid_dataset = CQDataset(valid_image_paths,class_to_idx, test_transforms) #test transforms are applied
    test_dataset = CQDataset(test_image_paths,class_to_idx, test_transforms)

    print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
    print('The label for 50th image in train dataset: ',train_dataset[49][1])

    

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return train_dataset, valid_dataset

def compute_img_mean_std(image_paths):
    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)
    imgs = np.stack(imgs, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()
    return means,stdevs

def get_ham1000():
    data_dir = './data/ham10000'
    #print(os.listdir(data_dir))
    all_image_path = glob.glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    norm_mean,norm_std = compute_img_mean_std(all_image_path)

    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    df_undup = df_original.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    df_original['duplicates'] = df_original['lesion_id']
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
    df_original.head()
    df_original['duplicates'].value_counts()
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
    df_val['cell_type_idx'].value_counts()

    def get_val_rows(x):
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    df_original['train_or_val'] = df_original['image_id']
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    df_train = df_original[df_original['train_or_val'] == 'train']

    data_aug_rate = [15,10,5,50,0,40,5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train=df_train.append(
                [df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    val_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    training_set = HAM10000(df_train, transform=train_transform)
    validation_set = HAM10000(df_val, transform=val_transform)

    return training_set,validation_set
