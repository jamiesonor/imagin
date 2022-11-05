import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle, randrange
from torch import tensor, float32, save, load
from torch.utils.data import Dataset
from nilearn.image import load_img, smooth_img, clean_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal, fetch_atlas_destrieux_2009, fetch_atlas_harvard_oxford
from sklearn.model_selection import StratifiedKFold


class ABIDE(Dataset):
    def __init__(self, sourcedir, roi, dynamic_length=None, k_fold=None, target_feature='Autism', smoothing_fwhm=None):
        super().__init__()
        self.filename = 'abide'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi=='schaefer':
            self.roi = fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
            atlas_img = load_img(self.roi['maps'])
        elif roi=='aal':
            self.roi = fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
            atlas_img = load_img(self.roi['maps'])
        elif roi=='cc200':
            atlas_img = load_img(os.path.join(sourcedir, 'roi', 'cc200', 'cc200_roi_atlas.nii.gz'))

        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pth')):
            self.timeseries_dict = load(os.path.join(sourcedir, f'{self.filename}.pth'))
        else:
            roi_masker = NiftiLabelsMasker(atlas_img, standardize=True)
            self.timeseries_dict = {}
            img_list = [f for f in os.listdir(os.path.join(sourcedir, 'img', 'REST')) if f.endswith('nii.gz')]
            img_list.sort()
            for img in tqdm(img_list, ncols=60):
                id = img.split('.')[0]
                timeseries = roi_masker.fit_transform(load_img(os.path.join(sourcedir, 'img', 'REST', img)))
                # if not len(timeseries) == 1200: continue
                self.timeseries_dict[id] = timeseries
            save(self.timeseries_dict, os.path.join(sourcedir, f'{self.filename}.pth'))

        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'behavioral', 'abide.csv')).set_index('Subject')[target_feature]
        self.num_classes = len(behavioral_df.unique())
        self.behavioral_dict = behavioral_df.to_dict()
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]


    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        if not self.dynamic_length is None:
            sampling_init = randrange(len(timeseries) - self.dynamic_length)
            timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]
        label = self.behavioral_dict[int(subject)]

        if label == 0:
            label = tensor(0)
        elif label == 1:
            label = tensor(1)
        else:
            raise

        return {'id': subject, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}

