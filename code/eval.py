#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:30:12 2023

@author: rouge
"""

import csv
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import nibabel as nib

from skimage.morphology import skeletonize
from skimage.morphology import remove_small_objects
from skimage.measure import label


parser = argparse.ArgumentParser()
parser.add_argument('--dir_inputs', type=str, default='../../../Thèse_Rougé_Pierre/res_semi_supervised/UA-MT/UA-MT-Bullitt_18_labeled_post', help='Path')
parser.add_argument('--postprocessing', type=bool,  default=False, help='prostprocessing or not')
parser.add_argument('--topo', type=bool, default=False, help='Compute topological metrics')

args = parser.parse_args()

dir_inputs = args.dir_inputs +'/*_pred.nii.gz'
postprocessing = args.postprocessing

# Metrics

def dice_numpy(y_true, y_pred):
    """
    Compute dice on numpy array

    Parameters
    ----------
    y_true : Numpy array of size (dim1, dim2, dim3)
        Ground truth
    y_pred : Numpy array of size (dim1, dim2, dim3)
         Predicted segmentation

    Returns
    -------
    dice : Float
        Value of dice

    """
    epsilon = 1e-5
    numerator = 2 * (np.sum(y_true * y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    dice = (numerator + epsilon) / (denominator + epsilon)
    return dice


def tprec_numpy(y_true, y_pred):
    epsilon = 1e-5
    y_pred = skeletonize(y_pred) / 255
    numerator = np.sum(y_true * y_pred)
    denominator = np.sum(y_pred)
    tprec = (numerator + epsilon) / (denominator + epsilon)
    return tprec


def tsens_numpy(y_true, y_pred):
    epsilon = 1e-5
    y_true = skeletonize(y_true) / 255
    numerator = (np.sum(y_true * y_pred))
    denominator = np.sum(y_true)
    tsens = (numerator + epsilon) / (denominator + epsilon)
    return tsens


def cldice_numpy(y_true, y_pred):
    tprec = tprec_numpy(y_true, y_pred)
    tsens = tsens_numpy(y_true, y_pred)
    numerator = 2 * tprec * tsens
    denominator = tprec + tsens
    cldice = numerator / denominator
    return cldice


def sensitivity_specificity_precision(y_true, y_pred):
    tp = np.sum(y_pred * y_true)
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    sens = tp / np.sum(y_true)
    spec = tn / (tn + fp)   
    prec = tp / (tp + fp)
    
    return sens, spec, prec

def euler_number_numpy(y):
    shape_ = y.shape
    new_shape = (shape_[0] * 2 + 1, shape_[1] * 2 + 1, shape_[2] * 2 + 1)
    CW = np.zeros(new_shape)
    for i in range(0, shape_[0]):
        for j in range(0, shape_[1]):
            for k in range(0, shape_[2]):
                a, b, c = 2*i + 1, 2*j + 1, 2*k + 1
                CW[a, b, c] = y[i, j, k]
                if CW[a, b, c] == 1:
                    for p in [-1, 0, 1]:
                        for q in [-1, 0, 1]:
                            for s in [-1, 0, 1]:
                                CW[a + p,b + q, c + s] = 1
              
    f0 = 0
    f1 = 0
    f2 = 0
    f3 = 0                             
    for i in range(0, shape_[0] * 2 + 1):
        for j in range(0, shape_[1] * 2 + 1):
            for k in range(0, shape_[2] * 2 +1):
                if (i % 2 == 0) and (j % 2 == 0)  and (k % 2 == 0):
                    f0 += CW[i, j, k]
                elif (i % 2 == 1) and (j % 2 == 1)  and (k % 2 == 1):
                    f3 += CW[i, j, k]
                elif ((i % 2 == 0) and (j % 2 == 1)  and (k % 2 == 1)) or ((i % 2 == 1) and (j % 2 == 0)  and (k % 2 == 1)) or ((i % 2 == 1) and (j % 2 == 1)  and (k % 2 == 0)):
                    f2 += CW[i, j, k]
                elif ((i % 2 == 1) and (j % 2 == 0)  and (k % 2 == 0)) or ((i % 2 == 0) and (j % 2 == 1)  and (k % 2 == 0)) or ((i % 2 == 0) and (j % 2 == 0)  and (k % 2 == 1)):   
                    f1 += CW[i, j, k]
    euler = f0 - f1 + f2 -f3
        
    return euler

def euler_number_error_numpy(y_true, y_pred, method='difference'):
    euler_number_true = euler_number_numpy(y_true)
    euler_number_pred = euler_number_numpy(y_pred)
    
    if method == 'difference' :
        euler_number_error = np.absolute(euler_number_true - euler_number_pred)
    
    elif method == 'relative' :
        euler_number_error = np.absolute(np.absolute(euler_number_true - euler_number_pred) / euler_number_true)
    
    return euler_number_error, euler_number_true, euler_number_pred
    

def b0_error_numpy(y_true, y_pred, method='difference'):
    _, ncc_true = label(y_true, return_num=True)
    _, ncc_pred = label(y_pred, return_num=True)
    
    b0_true= ncc_true - 1
    b0_pred = ncc_pred - 1
    
    if method == 'difference' :
        b0_error = np.absolute(b0_true - b0_pred)
    elif method == 'relative' :
       b0_error = np.absolute(b0_true - b0_pred) / b0_true
    
    return b0_error, b0_true, b0_pred

def b1_error_numpy(y_true, y_pred, method='difference'):
    
    euler_number_true = euler_number_numpy(y_true)
    euler_number_pred = euler_number_numpy(y_pred)
    
    _, ncc_true = label(y_true, return_num=True)
    _, ncc_pred = label(y_pred, return_num=True)
    
    b0_true= ncc_true - 1
    b0_pred = ncc_pred - 1
    
    y_true_inverse = np.ones(y_true.shape) - y_true
    y_pred_inverse = np.ones(y_pred.shape) - y_pred
    
    _, ncc_true = label(y_true_inverse, return_num=True)
    _, ncc_pred = label(y_pred_inverse, return_num=True)
    
    b2_true= ncc_true - 1
    b2_pred = ncc_pred - 1
    
    b1_true = b0_true + b2_true - euler_number_true
    b1_pred = b0_pred + b2_pred - euler_number_pred
    
    
    if method == 'difference' :
        b1_error = np.absolute(b1_true - b1_pred)
    elif method == 'relative' :
        b1_error = np.absolute(b1_true - b1_pred) / b1_true
    
    return b1_error, b1_true, b1_pred

def b2_error_numpy(y_true, y_pred, method='difference'):
    y_true_inverse = np.ones(y_true.shape) - y_true
    y_pred_inverse = np.ones(y_pred.shape) - y_pred
    
    _, ncc_true = label(y_true_inverse, return_num=True)
    _, ncc_pred = label(y_pred_inverse, return_num=True)
    
    b2_true= ncc_true - 1
    b2_pred = ncc_pred - 1
    
    if method == 'difference' :
        b2_error = np.absolute(b2_true - b2_pred)
        
    elif method == 'relative' :
         b2_error = np.absolute(b2_true - b2_pred) / b2_true
         
    return b2_error, b2_true, b2_pred

listt = glob(dir_inputs)

exp = dir_inputs.split('/')[-2]
res = open(dir_inputs.replace('/*_pred.nii.gz', '') + '/../res_' + exp + '.csv', 'w')
if args.topo:
    fieldnames = ['Patient', 'Dice', 'clDice', 'Precision', 'Sensitivity', 'Euler Number error', 'B0 error', 'B1 error', 'B2 error']
else:
    fieldnames = ['Patient', 'Dice', 'clDice', 'Precision', 'Sensitivity']
writer = csv.DictWriter(res, fieldnames=fieldnames)
writer.writeheader()

dice_list = []
cldice_list = []
prec_list = []
sens_list = []
euler_number_list = []
b0_list = []
b1_list = []
b2_list = []
for item in tqdm(listt):
    pred = nib.load(item)
    gt = nib.load(item.replace('_pred.nii.gz', '_gt.nii.gz'))
    name = item.split('/')[-1]
    name = name.split('_')[0]
    
    pred = pred.get_fdata()
    gt = gt.get_fdata()
    
    if postprocessing:
        pred = remove_small_objects(np.array(pred, dtype=bool), min_size=100)
        gt = remove_small_objects(np.array(gt, dtype=bool), min_size=100)
    
    dice = dice_numpy(gt, pred)
    cldice = cldice_numpy(gt, pred)
    sens, spec, prec = sensitivity_specificity_precision(gt, pred)
    
    if args.topo:
        euler_number_error, _, _ = euler_number_error_numpy(gt, pred)
        b0_error, _, _ = b0_error_numpy(gt, pred)
        b1_error, _, _ = b1_error_numpy(gt, pred)
        b2_error, _, _ = b2_error_numpy(gt, pred)
    
    dice_list.append(dice)
    cldice_list.append(cldice)
    prec_list.append(prec)
    sens_list.append(sens)
    
    if args.topo:
        euler_number_list.append(euler_number_error)
        b0_list.append(b0_error)
        b1_list.append(b1_error)
        b2_list.append(b2_error)
    
        dict_csv = {'Patient': name,
                    'Dice': dice,
                    'clDice': cldice,
                    'Precision': prec,
                    'Sensitivity': sens,
                    "Euler Number error": euler_number_error,
                    "B0 error": b0_error,
                    "B1 error": b1_error,
                    "B2 error": b2_error}
    else:
        dict_csv = {'Patient': name,
                    'Dice': dice,
                    'clDice': cldice,
                    'Precision': prec,
                    'Sensitivity': sens,
                    }
    
    dict_csv = {'Patient': name,
                'Dice': dice,
                'clDice': cldice,
                'Precision': prec,
                'Sensitivity': sens,
                "Euler Number error": euler_number_error,
                "B0 error": b0_error,
                "B1 error": b1_error,
                "B2 error": b2_error}
    
    writer.writerow(dict_csv)
    
    print()
    print()
    print(f"Dice:{dice}")
    
if args.topo:
    dict_csv = {'Patient': "Mean",
                'Dice': np.mean(dice_list),
                'clDice': np.mean(cldice_list),
                'Precision': np.mean(prec_list),
                'Sensitivity': np.mean(sens_list),
                "Euler Number error": np.mean(euler_number_list),
                "B0 error": np.mean(b0_list),
                "B1 error": np.mean(b1_list),
                "B2 error": np.mean(b2_list),
                }
    
    writer.writerow(dict_csv)
    
    dict_csv = {'Patient': "Std",
                'Dice': np.std(dice_list),
                'clDice': np.std(cldice_list),
                'Precision': np.std(prec_list),
                'Sensitivity': np.std(sens_list),
                "Euler Number error": np.std(euler_number_list),
                "B0 error": np.std(b0_list),
                "B1 error": np.std(b1_list),
                "B2 error": np.std(b2_list),}
    
    writer.writerow(dict_csv)
    
else:
    dict_csv = {'Patient': "Mean",
                'Dice': np.mean(dice_list),
                'clDice': np.mean(cldice_list),
                'Precision': np.mean(prec_list),
                'Sensitivity': np.mean(sens_list)}
    
    writer.writerow(dict_csv)
    
    dict_csv = {'Patient': "Std",
                'Dice': np.std(dice_list),
                'clDice': np.std(cldice_list),
                'Precision': np.std(prec_list),
                'Sensitivity': np.std(sens_list)}
    
    writer.writerow(dict_csv)
   
print()
print('Moyenne Dice')
print(np.mean(dice_list)) 
print('Moyenne clDice')
print(np.mean(cldice_list))
print('Precision Mean')
print(np.mean(prec_list))
print('Sensitiviy Mean')
print(np.mean(sens_list))
res.close()

    
    
    
    