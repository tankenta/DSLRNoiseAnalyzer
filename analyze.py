#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import *


def get_dst_dir_path(src_dir, dst_dir_prefix):
    if src_dir == '':
        print('src_dir must not be empty. exit.')
        sys.exit(1)
    if src_dir[-1] == '/' or src_dir[-1] == '\\':
        src_dir = src_dir[:-1]
    par_dir, child_dir = os.path.split(src_dir)
    dst_dir = os.path.join(par_dir, dst_dir_prefix + child_dir)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
        print('mkdir: {0}'.format(dst_dir))
    return dst_dir

def get_imgs_paths(par_dir, exts=['.jpg', '.JPG']):
    paths = set()
    for ext in exts:
        paths |= set(glob.glob(os.path.join(par_dir, '*' + ext)))
    return sorted(paths)

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
    fname = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, fname)
    return dst_path

def compute_noise(src_dir, dst_dir):
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
    return noise_arr, sample_dicts, np.copy(mean_img)

def print_representative_val(noise_arr, plt_label):
        noise_np = np.array(noise_arr)
        print('# {0}'.format(plt_label))
        print('mean, median, min, max, variance, SD')
        print('{0:.3f}, {1:.1f}, {2}, {3}, {4:.3f}, {5:.3f}\n'.format(
            noise_np.mean(), np.median(noise_np), noise_np.min(),
            noise_np.max(), noise_np.var(), noise_np.std()
        ))
        hist = np.histogram(noise_np)
        print(hist[0])
        print(hist[1])

def draw_histogram(fig, noise_arrs, bins, plt_labels, plt_colors, plt_alpha):
    ax_hist = fig.add_subplot(1, 1, 1)
    calc_bins = bins is None
    for noise_arr, plt_label, plt_color in zip(noise_arrs, plt_labels, plt_colors):
        if calc_bins:
            bins = np.arange(np.min(noise_arr), np.max(noise_arr) + 2) - 0.5
        ax_hist.hist(
                noise_arr, bins=bins, alpha=plt_alpha,
                histtype='stepfilled', color=plt_color, label=plt_label
        )
    ax_hist.legend()
    ax_hist.set_xlabel('diff')
    ax_hist.set_ylabel('freq')
#     ax_hist.set_xlim((-50, 30))
    fig.tight_layout()

def save_frequency_table(noise_arr, dst_dir, fname_prefix):
    bins = np.arange(np.min(noise_arr), np.max(noise_arr) + 2) - 0.5
    freq, boundary = np.histogram(noise_arr, bins=bins)
    varmin = boundary[:-1]
    varsup = boundary[1:]
    dst_data = np.vstack((varmin, varsup, freq)).T
    dst_path = os.path.join(dst_dir, '{0}freq.csv'.format(fname_prefix))
    header = 'varmin,varsup,freq'
    np.savetxt(dst_path, dst_data, delimiter=',', header=header)
    print('Wrote: {0}'.format(dst_path))

def show_sample_and_mean_imgs(fig, sample_dicts_arr, mean_img_arr, plt_labels):
    for i, (sample_dicts, mean_img) in enumerate(zip(sample_dicts_arr, mean_img_arr)):
        sample_img = sample_dicts[0]['img']
        ax_s = fig.add_subplot(2, len(mean_img_arr), 2*i+1)
        ax_s.imshow(cv2.cvtColor(sample_img, cv2.COLOR_GRAY2RGB))
        ax_s.set_xlabel('sample ({0})'.format(plt_labels[i]))
        ax_m = fig.add_subplot(2, len(mean_img_arr), 2*i+2)
        ax_m.imshow(cv2.cvtColor(mean_img, cv2.COLOR_GRAY2RGB))
        ax_m.set_xlabel('mean of {0} imgs ({1})'.format(len(sample_dicts), plt_labels[i]))
    fig.tight_layout()

def main():
    # compute noise
    noise_arrs = list()
    sample_dicts_arr = list()
    mean_img_arr = list()
    for src_dir, fname_prefix in zip(SRC_DIRS, FNAME_PREFIXES):
        dst_dir = get_dst_dir_path(src_dir, DST_DIR_PREFIX)
        noise_arr, sample_dicts, mean_img = compute_noise(src_dir, dst_dir)
        noise_arrs.append(noise_arr)
        sample_dicts_arr.append(sample_dicts)
        mean_img_arr.append(np.copy(mean_img))
        # save frequency table
        save_frequency_table(noise_arr, dst_dir, fname_prefix)

    # print statistic representative value
    for noise_arr, plt_label in zip(noise_arrs, PLT_LABELS):
        print_representative_val(noise_arr, plt_label)

    # draw histgram
    fig_hist = plt.figure(1)
    draw_histogram(fig_hist, noise_arrs, BINS, PLT_LABELS, PLT_COLORS, PLT_ALPHA)

    # show samples and mean imgs
    fig_imgs = plt.figure(2)
    show_sample_and_mean_imgs(fig_imgs, sample_dicts_arr, mean_img_arr, PLT_LABELS)

    plt.show()


if __name__ == '__main__':
    main()
