#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

MEAN_IMG_FNAME = 'mean.jpg'
IMREAD_FLAG = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH
IMSHAPE = (3456, 5184)
SHAPE = (300, 300)
OFFSET = tuple((np.array(IMSHAPE) - np.array(SHAPE))/2)
TRANS_PX = 2
BINS = 200
SAMPLE_FLG = False
MEAN_FLG = False

def get_imgs_paths(par_dir, exts=['.jpg', '.JPG']):
    paths = list()
    for ext in exts:
        paths.extend(glob.glob(os.path.join(par_dir, '*' + ext)))
    return paths

def sample_small_img(src_img, offset, shape):
    y, x = offset
    rows, cols = shape
    dst_img = src_img[y:y+rows, x:x+cols]
    return dst_img

def sample_with_trans(src_img, target_img, offset, shape, trans_px):
    diff_arr = list()
    for y_trans in range(-trans_px, trans_px+1):
        for x_trans in range(-trans_px, trans_px+1):
            trans = np.array([y_trans, x_trans])
            adjusted_offset = np.array(offset) + trans
            candidate_img = sample_small_img(src_img, adjusted_offset, shape)
            diff_img = candidate_img.astype(np.int) - target_img.astype(np.int)
            diff = np.sum(np.abs(diff_img))
            diff_arr.append({
                'diff': diff, 'trans': tuple(trans), 'img': np.copy(candidate_img)
            })
    min_diff_idx = np.array([d['diff'] for d in diff_arr]).argmin()
    diff_dict = diff_arr[min_diff_idx]
    print('trans: {0}'.format(diff_dict['trans']))
    dst_img = np.copy(diff_dict['img'])
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
    
    sample_dicts = list()
    if SAMPLE_FLG:
        paths = get_imgs_paths(src_dir)
        for i, path in enumerate(paths):
            src_img = cv2.imread(path, IMREAD_FLAG)
            if i == 0:
                dst_img = sample_small_img(src_img, OFFSET, SHAPE)
            else:
                target_img = sample_dicts[0]['img']
                dst_img = sample_with_trans(src_img, target_img, OFFSET, SHAPE, TRANS_PX)

            dst_path = get_dst_path(path, dst_dir)
            cv2.imwrite(dst_path, dst_img)
            print('Wrote: {0}'.format(dst_path))
            sample_dicts.append({'path': dst_path, 'img': np.copy(dst_img)})
    else:
        paths = get_imgs_paths(dst_dir)
        for path in paths:
            # TODO: use os.path module
            if MEAN_IMG_FNAME in path:
                continue
            dst_img = cv2.imread(path, IMREAD_FLAG)
            if dst_img is None:
                print('Failed to read a sample image: {0}'.format(path))
                return
            else:
                print('Read: {0}'.format(path))
            sample_dicts.append({'path': path, 'img': np.copy(dst_img)})

    if MEAN_FLG:
        imgs_arr = [s_dict['img'].astype(np.uint16) for s_dict in sample_dicts]
        mean_img = (np.array(imgs_arr).sum(axis=0) / len(imgs_arr)).astype(np.uint8)
        mean_img_path = os.path.join(dst_dir, MEAN_IMG_FNAME)
        cv2.imwrite(mean_img_path, mean_img)
        print('Wrote: {0}'.format(mean_img_path))
    else:
        mean_img_path = os.path.join(dst_dir, MEAN_IMG_FNAME)
        mean_img = cv2.imread(mean_img_path, IMREAD_FLAG)
        if mean_img is None:
            print('Failed to read a mean image: {0}'.format(mean_img_path))
            return
        else:
            print('Read: {0}'.format(mean_img_path))

    noise_arr = list()
    for sample_dict in sample_dicts:
        diff_img = sample_dict['img'].astype(int) - mean_img.astype(int)
        noise_arr.extend(diff_img.flatten())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(noise_arr, bins=BINS)
    ax.set_xlabel('diff')
    ax.set_ylabel('freq')
#     ax.set_xlim((-20, 20))
    plt.show()

if __name__ == '__main__':
    main()
