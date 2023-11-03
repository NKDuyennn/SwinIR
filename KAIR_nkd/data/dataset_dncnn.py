import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing.
    # Dataroot_H and L is both needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__()
        print('Dataset: Denosing on transmitted images - code fixed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        #self.sigma = opt['sigma'] if opt['sigma'] else 25                  #NO NEED BECAUSE USING TRANSMITTED IMAGES
        # self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        #self.paths_H = util.get_image_paths(opt['dataroot_H'])
########################################################################
########################### CHANGES ---- PHONG!! ########################
########################################################################
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = self.paths_L[index]
        # ------------------------------------
        # get H image and L image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_L = util.imread_uint(L_path, self.n_channels)

#########################################################################
    # def __getitem__(self, index):

    #     # ------------------------------------
    #     # get H image
    #     # ------------------------------------
    #     H_path = self.paths_H[index]
    #     img_H = util.imread_uint(H_path, self.n_channels)

    #     L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

            # --------------------------------
            # add noise  #NO NEED
            # --------------------------------
            # noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            # img_L.add_(noise) 

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape
            # patch_size = (min(H,W)//8)*8
            H = (H//8)*8
            W = (W//8)*8
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
            
            patch_H = img_H[0:H, 0:W, :]
            patch_L = img_L[0:H, 0:W, :]
            # --------------------------------
            # add noise   --- NO NEED
            # --------------------------------
            # np.random.seed(seed=0)
            # img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_H = util.single2tensor3(patch_H)
            img_L = util.single2tensor3(patch_L)
            
            # print(f'img data shape: {img_H.shape}')

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
