import os
import sys
import copy
import warnings

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet


warnings.filterwarnings("ignore")


class BGRemove():
    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    if device == 'cuda':
        modnet = modnet.cuda()

    def __init__(self, ckpt_path):
        self.parameter_load(ckpt_path)

    def parameter_load(self, ckpt_path):
        BGRemove.modnet.load_state_dict(
            torch.load(ckpt_path, map_location=BGRemove.device))
        BGRemove.modnet.eval()

    def file_load(self, filename):
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        return im

    def dir_check(self, path):
        os.makedirs(path, exist_ok=True)
        if not path.endswith('/'):
            path += '/'
        return path

    def pre_process(self, im):
        self.original_im = copy.deepcopy(im)

        # convert image to PyTorch tensor
        im = BGRemove.im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        self.height, self.width = im_h, im_w

        if max(im_h, im_w) < BGRemove.ref_size or min(im_h, im_w) > BGRemove.ref_size:
            if im_w >= im_h:
                im_rh = BGRemove.ref_size
                im_rw = int(im_w / im_h * BGRemove.ref_size)
            elif im_w < im_h:
                im_rw = BGRemove.ref_size
                im_rh = int(im_h / im_w * BGRemove.ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        if BGRemove.device == 'cuda':
            im = im.cuda()
        return im

    def post_process(self, mask_data, background=False, backgound_path='assets/background/background.jpg'):
        matte = F.interpolate(mask_data, size=(
            self.height, self.width), mode='area')
        matte = matte.repeat(1, 3, 1, 1)
        matte = matte[0].data.cpu().numpy().transpose(1, 2, 0)
        height, width, _ = matte.shape
        if background:
            back_image = self.file_load(backgound_path)
            back_image = cv2.resize(
                back_image, (width, height), cv2.INTER_AREA)
        else:
            back_image = np.full(self.original_im.shape, 255.0)

        self.alpha = np.uint8(matte[:, :, 0]*255)

        matte = matte * self.original_im + (1 - matte) * back_image
        return matte

    def image(self, filename, background=False, output='output/', save=True):
        output = self.dir_check(output)

        self.im_name = filename.split('/')[-1]
        im = self.file_load(filename)
        im = self.pre_process(im)
        _, _, matte = BGRemove.modnet(im, inference=False)
        matte = self.post_process(matte, background)

        if save:
            matte = np.uint8(matte)
            msg, name = self.save(matte, output, background)
            return name
        else:
            h, w, _ = matte.shape
            r_h, r_w = 720, int((w / h) * 720)
            image = cv2.resize(self.original_im, (r_w, r_h), cv2.INTER_AREA)
            matte = cv2.resize(matte, (r_w, r_h), cv2.INTER_AREA)

            full_image = np.uint8(np.concatenate((image, matte), axis=1))
            self.save(full_image, output, background)
            exit_key = ord('q')
            while True:
                if cv2.waitKey(exit_key) & 255 == exit_key:
                    cv2.destroyAllWindows()
                    break
                cv2.imshow(
                    'MODNet - {} [Press "Q" To Exit]'.format(self.im_name), full_image)

    def video(self, filename, background=False, output='output/'):
        output = self.dir_check(output)

        output_name = filename.split('/')[-1]
        extension = output_name.split('.')[-1]
        output_name = output_name.replace(extension, 'mp4')

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        cap = cv2.VideoCapture(filename)
        flag = 1
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        while (cap.isOpened()):
            ret, frame = cap.read()
            if flag:
                height, width, _ = frame.shape
                out = cv2.VideoWriter(output+output_name,
                                      fourcc, 20.0, (2*width, height))
                flag = 0

            if ret:
                print('Video is processing..', end='\r')

                im = self.pre_process(frame)
                _, _, matte = BGRemove.modnet(im, inference=False)
                matte = np.uint8(self.post_process(matte, background))
                full_image = np.concatenate((frame, matte), axis=1)
                full_image = np.uint8(cv2.resize(
                    full_image, (2*width, height), cv2.INTER_AREA))
                out.write(full_image)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def folder(self, foldername, background=False, output='output/'):
        output = self.dir_check(output)
        foldername = self.dir_check(foldername)

        for filename in os.listdir(foldername):
            try:
                self.im_name = filename
                im = self.file_load(foldername+filename)
                im = self.pre_process(im)
                _, _, matte = BGRemove.modnet(im, inference=False)
                matte = self.post_process(matte, background)
                status = self.save(matte, output, background)
                print(status)
            except:
                print('There is an error for {} file/folder'.format(foldername+filename))

    def webcam(self, background=False):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        width, height = 455, 512

        exit_key = ord('q')
        while(True):
            _, frame_np = cap.read()
            frame_np = cv2.resize(frame_np, (width, height), cv2.INTER_AREA)
            im = self.pre_process(frame_np)
            _, _, matte = BGRemove.modnet(im, inference=False)
            processed_image = self.post_process(matte, background)

            full_image = np.concatenate((frame_np, processed_image), axis=1)
            full_image = np.uint8(cv2.resize(
                full_image, (2*width, height), cv2.INTER_AREA))

            if cv2.waitKey(exit_key) & 255 == exit_key:
                cv2.destroyAllWindows()
                break
            cv2.imshow('MODNet - WebCam [Press "Q" To Exit]', full_image)

    def save(self, matte, output_path='output/', background=False):
        name = '.'.join(self.im_name.split('.')[:-1])+'.png'
        path = os.path.join(output_path, name)

        if background:
            try:
                matte = cv2.cvtColor(matte, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, matte)
                return "Successfully saved {}".format(path), name
            except:
                return "Error while saving {}".format(path), ''
        else:
            w, h, _ = matte.shape
            png_image = np.zeros((w, h, 4))
            png_image[:, :, :3] = matte
            png_image[:, :, 3] = self.alpha
            png_image = png_image.astype(np.uint8)
            try:
                png_image = cv2.cvtColor(png_image, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(path, png_image, [
                            int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                return "Successfully saved {}".format(path), name
            except:
                return "Error while saving {}".format(path), ''
