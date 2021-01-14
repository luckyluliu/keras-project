#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import copy
import tensorflow as tf
import keras
from pathlib import Path
from PIL import Image
from multiprocessing import Pool
import random
import pandas as pd
import numpy as np
import skimage
import skimage.io
from keras.preprocessing.image import ImageDataGenerator

class FundusSequence(tf.keras.utils.Sequence):
    def __init__(self, df, cube_dir, input_channels=7, batch_size=32, num_classes=5, augment=True, shuffle = True, datatype='png', **kwargs):
        if isinstance(df,str) or isinstance(df,Path):
            df = pd.read_csv(df)
            paths = df['imagepath']
            labels = df['label']
        elif isinstance(df,tuple) or isinstance(df,list):
            paths, labels = df
        elif isinstance(df,pd.DataFrame):
            paths = df['imagepath']
            labels = df['label']
        
        self.shuffle = shuffle
        self._datatype = datatype
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._cube_dir = Path(cube_dir)
        self._paths = list(paths)
        self._labels = list(labels)
        self._indexes = np.arange(len(self._labels))
        self._input_channels = input_channels
        self._augment = augment
        self._data_generator = KrIdg(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            fill_mode='constant',
            cval = 0,)

    def __len__(self):
        return len(self._paths) // self._batch_size + 1

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self._indexes)
    
    def _load_cube(self, batch_filepaths, batch_labels):
        batch_x = np.zeros((len(batch_filepaths), 512, 512, self._input_channels), dtype=np.uint8)
        batch_y = []
        for i, filepath in enumerate(batch_filepaths):
            path = Path(filepath)
            batch_y.append((keras.utils.to_categorical(batch_labels[i], num_classes=self._num_classes)))
            if self._datatype == 'png':
                imgpath = self._cube_dir / path
                lesion = cv2.imread(str(imgpath))
                x = cv2.resize(lesion, (512, 512))
            elif self._datatype == 'npy':
                lesionPath = self._cube_dir / 'fundus_lesion' / path.with_suffix('.npy')
                lesion = np.load(lesionPath)
                x = lesion[...,[0,1,2,4,5,6,7]]
            batch_x[i, ...] = x
        batch_y = np.asarray(batch_y)
        return batch_x, batch_y
    
    def __getitem__(self, idx):
        '''
        :return: X,Y data of shape (batchsize, width, height, channel)
        '''
        if idx >= len(self):
            raise ValueError('Asked to retrieve delement {idx}, '
                'but the Sequence '
                'has length {length}'.format(idx=idx, length=len(self)))
        # 根据索引获取datas集合中的数据
        batch_indexs = self._indexes[idx*self._batch_size:(idx+1)*self._batch_size]
        batch_filepaths = [self._paths[k] for k in batch_indexs]
        batch_labels = [self._labels[k] for k in batch_indexs]
        batch_x, batch_y = self._load_cube(batch_filepaths, batch_labels)
        if self._augment:
            batch_x = self._data_generator.fit(batch_x, augment=True)
        return batch_x, batch_y

class KrIdg(ImageDataGenerator):
    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits the data generator to some sample data.

        This computes the internal data stats related to the
        data-dependent transformations, based on an array of sample data.

        Only required if `featurewise_center` or
        `featurewise_std_normalization` or `zca_whitening` are set to True.

        # Arguments
            x: Sample data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, in case
             of RGB data, it should have value 3, and in case
             of RGBA data, it should have value 4.
            augment: Boolean (default: False).
                Whether to fit on randomly augmented samples.
            rounds: Int (default: 1).
                If using data augmentation (`augment=True`),
                this is how many augmentation passes over the data to use.
            seed: Int (default: None). Random seed.
       """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=self.dtype)
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + 1e-6)

        if self.zca_whitening:
            if scipy is None:
                raise ImportError('Using zca_whitening requires SciPy. '
                                  'Install SciPy.')
            flat_x = np.reshape(
                x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)
        return x
