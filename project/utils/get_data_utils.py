import os, glob
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import sampler
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler


DATA_DIR = "/Users/wli169/Documents/Work/datasets/"

import zipfile
import urllib.request


import os
import urllib.request
import zipfile

# This script includes the following functions to acquire data
# get_np_ravdess: RAVDESS Emotional speech audio (with frames averaged over)
# get_np_ravdess2: AVDESS Emotional speech audio (with padding or truncation on number of frames)
# get_np_mnist: obtain MNIST numpy data
# get_dataloaders_mnist: obtain dataloaders in Pytorch.


def get_np_ravdess(emotions_labels, feature_extractor, chroma_T, mel_T):
    """
    # RAVDESS Emotional speech audio
    # https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
    # Speech audio-only files (16bit, 48kHz .wav) from the RAVDESS.
    # Download it to a local directory

    This portion of the RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440.
    The RAVDESS contains 24 professional actors (12 female, 12 male),
    vocalizing two lexically-matched statements in a neutral North American accent.
    Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions.
    Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

    The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:

    Filename identifiers:
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

    Filename example: 03-01-06-01-02-01-12.wav

    Args:
        feature_extractor: returns (num_features, ).
        The feature extractor used here will take average value across all frames

    Return:
        np.array(x) has shape (num_examples, num_features)
        np.array(y) has shape (num_examples,)
    """

    emotions = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    x, y = [], []
    # Load the data and extract features for each sound file
    for file in glob.glob(os.path.join(DATA_DIR, "RAVDESS-speech/Actor_*/*.wav")):
        # loops through all the .wav file
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        # file_name.split("-")[2] will return the third element of the
        # list that results from splitting the string by the "-" character,
        # i.e., return the emotion label
        if emotion not in emotions_labels:
            continue
        feature = feature_extractor(file, mfcc=True, chroma=chroma_T, mel=mel_T)
        # feature has shape (num_features, )
        x.append(feature)
        y.append(emotion)

    return np.array(x), np.array(y)
    #      np.array(x) has shape (num_examples, num_features)
    #      np.array(y) has shape (num_examples,)


def get_np_ravdess2(emotions_labels, feature_extractor, max_frames, chroma_T, mel_T):
    """
    # RAVDESS Emotional speech audio
    # https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
    # Speech audio-only files (16bit, 48kHz .wav) from the RAVDESS.
    # Download it to a local directory

    This portion of the RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440.
    The RAVDESS contains 24 professional actors (12 female, 12 male),
    vocalizing two lexically-matched statements in a neutral North American accent.
    Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions.
    Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

    The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:

    Filename identifiers:
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

    Filename example: 03-01-06-01-02-01-12.wav

    Args:
        feature_extractor: returns (num_frames, num_features).
        The feature extractor used here will not average over all frames.

    Return:
        np.array(x) has shape (num_examples, max_frames, num_features)
        np.array(y) has shape (num_examples,)
    """

    emotions = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    max_frames = max_frames

    x, y = [], []
    # Load the data and extract features for each sound file
    for file in glob.glob(os.path.join(DATA_DIR, "RAVDESS-speech/Actor_*/*.wav")):
        # loops through all the .wav file
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        # file_name.split("-")[2] will return the third element of the
        # list that results from splitting the string by the "-" character,
        # i.e., return the emotion label
        if emotion not in emotions_labels:
            continue
        feature = feature_extractor(file, mfcc=True, chroma=chroma_T, mel=mel_T)
        # feature has shape (num_frames, num_features)

        # Pad or truncate the feature sequence based on max_frames
        if max_frames:
            if feature.shape[0] > max_frames:
                feature = feature[:max_frames, :]
            else:
                pad_width = max_frames - feature.shape[0]
                feature = np.pad(
                    feature, pad_width=((0, pad_width), (0, 0)), mode="constant"
                )
                # The padding is done only at the end of the first axis (rows),
                # and no padding is done on the second axis (columns), maintaining the original number of features

        x.append(feature)
        y.append(emotion)

    return np.array(x), np.array(y)
    #      np.array(x) has shape (num_examples, max_frames, num_features)
    #      np.array(y) has shape (num_examples,)


def get_np_mnist(
    validation_fraction=None,
    train_transforms=transforms.ToTensor(),
    test_transforms=transforms.ToTensor(),
    download=False,
    seed=0,
):
    """
    obtain MNIST numpy files

    Explanation:
    transforms.ToTensor(): will divide pixels by 255, i.e., to normalize to range [0,1]

    Other possibility for transform are:
    (1)
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


    torchvision.transforms.Normalize((?, ), (?,)):
    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels,
    this transform will normalize each channel .

    output[channel] = (input[channel] - mean[channel]) / std[channel]
    So, the pixel value will be on [-1, 1]

    (2)
    transform =  transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    In mnist case, 0.1307 is the grand mean over H and W, 0.3081 is the grand std over H and W
    after being scaled on [0, 1].
    So, the pixel value will be measured in how many std from the mean.

    NOTE: All transform will not alter train_dataset.data itself, but
    only take effect when being loaded throught DataLoader
    """

    ## train data ##

    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, transform=train_transforms, download=False
    )

    # train dataset of MNIST images (in total 60000 images, each image of size 28x28 array)
    # test dataset of MNIST images (in total 10000 images, each image of size 28x28 array)

    # >>> type(train_dataset), train_dataset.data.shape, train_dataset.targets.shape
    # (<class 'torchvision.datasets.mnist.MNIST'>, torch.Size([60000, 28, 28]), torch.Size([60000]))
    # >>> type(train_dataset), train_dataset.data.dtype, train_dataset.targets.dtype
    # (<class 'torchvision.datasets.mnist.MNIST'>, torch.uint8, torch.int64)

    # for images, labels in train_loader:
    #     print('Image batch dimensions:', images.shape)
    #     print('Image label dimensions:', labels.shape)
    #     print('Class labels of 10 examples:', labels[:10])
    #     break

    # Image batch dimensions: torch.Size([60000, 1, 28, 28]) # NCHW format
    # Image label dimensions: torch.Size([60000])
    # Class labels of 10 examples: tensor([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])

    # labels are not one-hot encoded. Pytorch will do it automatically later.

    # Note the source data does not have validation dataset.
    # To create validation dataset, we split the training dataset using validation_fraction.

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    # len(train_loader) = number of batches
    train_dataset_x = next(iter(train_loader))[0].numpy()
    # next(iter(iterable)) returns the value of the iterator which is input and targets data
    train_dataset_y = next(iter(train_loader))[1].numpy()
    # train_dataset_y.dtype >>>dtype('int64'); train_dataset_x.dtype >>>dtype('float32')

    ###  Test data
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, transform=test_transforms, download=False
    )
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    test_dataset_x = next(iter(test_loader))[0].numpy()
    # next(iter(iterable)) returns the value of the iterator which is input and targets data
    test_dataset_y = next(iter(test_loader))[1].numpy()
    # train_dataset_y.dtype >>>dtype('int64'); train_dataset_x.dtype >>>dtype('float32')

    if validation_fraction is not None:
        np.random.seed(seed)
        num = int(validation_fraction * 60000)
        train_indices = np.arange(0, 60000 - num)
        valid_indices = np.arange(60000 - num, 60000)
        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)

        train_dataset_x, valid_dataset_x = (
            train_dataset_x[train_indices,],
            train_dataset_x[valid_indices,],
        )
        train_dataset_y, valid_dataset_y = (
            train_dataset_y[train_indices,],
            train_dataset_y[valid_indices,],
        )

    return (
        train_dataset_x,
        train_dataset_y,
        valid_dataset_x,
        valid_dataset_y,
        test_dataset_x,
        test_dataset_y,
    )


def get_dataloaders_mnist(
    batch_size,
    num_workers=0,
    validation_fraction=None,
    train_transforms=transforms.ToTensor(),
    test_transforms=transforms.ToTensor(),
    download=False,
    seed=0,
):
    """
    Obtain dataloaders in Pytorch.

    Credits: Sabastian Raschka
    """

    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, transform=train_transforms, download=download
    )

    valid_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, transform=test_transforms, download=download
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, transform=test_transforms, download=download
    )

    if validation_fraction is not None:
        torch.manual_seed(seed)
        num = int(validation_fraction * 60000)
        train_indices = torch.arange(0, 60000 - num)
        valid_indices = torch.arange(60000 - num, 60000)

        train_sampler = SubsetRandomSampler(train_indices)
        # sample without replacement
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=valid_sampler,
        )
        # using a customized sampler
        # to shuffle

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=train_sampler,
        )
        # using a customized sampler

    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            shuffle=True,
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader
