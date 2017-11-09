#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import glob
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

MEAN_IMG_FNAME = 'mean.JPG'
IMREAD_FLAG = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH
IMSHAPE = (3456, 5184)
SHAPE = (300, 300)
OFFSET = tuple((np.array(IMSHAPE) - np.array(SHAPE))/2)
TRANS_PX = 2

def get_imgs_paths(par_dir, ext='.JPG'):    # TODO: JPG, jpg
    paths = glob.glob(os.path.join(par_dir, '*'+ext))
    return paths

def sample_small_img(src_img, offset, shape):
    y, x = offset
    rows, cols = shape
    dst_img = src_img[y:y+rows, x:x+cols]
    return dst_img

def get_dst_path(src_path, dst_dir):
    _, fname = os.path.split(src_path)
    dst_path = os.path.join(dst_dir, fname)
    return dst_path


def main():
    if len(sys.argv) < 3:
        print('Usage: ./trim.py src_dir dst_dir')
        return
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    
    paths = get_imgs_paths(src_dir)
    sample_dicts = list()
    for i, path in enumerate(paths):
        src_img = cv2.imread(path, IMREAD_FLAG)
        if i == 0:
            dst_img = sample_small_img(src_img, OFFSET, SHAPE)
        else:
            target_img = sample_dicts[0]['img']
            diff_arr = list()
            for y_trans in range(-TRANS_PX, TRANS_PX+1):
                for x_trans in range(-TRANS_PX, TRANS_PX+1):
                    trans = np.array([y_trans, x_trans])
                    offset = np.array(OFFSET) + trans
                    dst_img = sample_small_img(src_img, offset, SHAPE)
                    diff_img = dst_img.astype(np.int) - target_img.astype(np.int)
                    diff = np.sum(np.abs(diff_img))
                    diff_arr.append({
                        'diff': diff, 'trans': tuple(trans), 'img': np.copy(dst_img)
                    })
            min_diff_idx = np.array([d['diff'] for d in diff_arr]).argmin()
            diff_dict = diff_arr[min_diff_idx]
            print('trans: {0}'.format(diff_dict['trans']))
            dst_img = np.copy(diff_dict['img'])

        dst_path = get_dst_path(path, dst_dir)
        cv2.imwrite(dst_path, dst_img)
        sample_dicts.append({'path': dst_path, 'img': np.copy(dst_img)})

#     for i, sample_dict in enumerate(sample_dicts):
#         plt.subplot(1, 2, i+1)
#         dst_img = sample_dict['img']
#         plt.imshow(cv2.cvtColor(dst_img, cv2.COLOR_GRAY2RGB))
#     plt.show()
    imgs_arr = [s_dict['img'].astype(np.uint16) for s_dict in sample_dicts]
    mean_img = (np.array(imgs_arr).sum(axis=0) / len(imgs_arr)).astype(np.uint8)
    mean_img_path = os.path.join(dst_dir, MEAN_IMG_FNAME)
    cv2.imwrite(mean_img_path, mean_img)
    print('Wrote: {0}'.format(mean_img_path))

    noise_arr = list()
    for sample_dict in sample_dicts:
        diff_img = sample_dict['img'].astype(int) - mean_img.astype(int)
        noise_arr.extend(diff_img.flatten())

    return noise_arr
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.hist(noise_arr, bins=100)
#     ax.set_xlabel('diff')
#     ax.set_ylabel('freq')
#     fig.show()

if __name__ == '__main__':
    noise_arr = main()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(noise_arr, bins=100)
    ax.set_xlabel('diff')
    ax.set_ylabel('freq')
    fig.show()
    while True:
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
