import argparse
from copy import deepcopy
import os
import math
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import sys
from sklearn.metrics import roc_curve

def statistics(data):
    if not data:
        print("List is empty")
        return

    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    maximum = max(data)
    minimum = min(data)
    
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Max: {maximum}")
    print(f"Min: {minimum}")

def read_csv_files(file1, file2):
    # Read the CSV files using pandas
    df1 = pd.read_csv(file1, nrows=200000)
    df2 = pd.read_csv(file2, nrows=200000)
    return df1, df2

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Read two CSV files.')
    parser.add_argument('--real', type=str, help='file containing scores of the real dataset')
    parser.add_argument('--fake', type=str, help='file containing scores of the fake dataset')
    parser.add_argument('--ix', type=int, default=2, help='number of models to evaluate')
    # Parse the arguments
    args = parser.parse_args()
    thresholds = {}

    df_real, df_fake = read_csv_files(args.real, args.fake) # Read the CSV files using pandas
    for model in df_real.columns[-args.ix:]:
        thresholds[model] = 0.50 # Set thresholds to 0.5
    print('THRESHOLD: ', thresholds)
    print('-----------------------------------------------------------------------------------')
    for model in df_real.columns[-args.ix:]:
        # Take Sigmoid of the scores to get probability of fakeness
        real = F.sigmoid(torch.tensor(df_real[model].tolist()))
        real = [x for x in real if not math.isnan(x)]
        fake = F.sigmoid(torch.tensor(df_fake[model].tolist()))
        fake = [x for x in fake if not math.isnan(x)]
        scores = real + fake
        #Print Statistics of the probabilities
        print('-----------------------------------------------------------------------------------')
        print('real')
        statistics(data=real)
        print('fake')
        statistics(data=fake)
        print('-----------------------------------------------------------------------------------')
        preds = (np.array(scores) >= thresholds[model]).astype(int)
        r_true = [0] * len(real) 
        f_true = [1] * len(fake)
        y_true = r_true + f_true
        ap = average_precision_score(y_true, scores)
        print(model,' RACC: ', accuracy_score(r_true, (np.array(real) >= thresholds[model]).astype(int)))
        print(model,' FACC: ', accuracy_score(f_true, (np.array(fake) >= thresholds[model]).astype(int)))
        print(model,' ACC: ', accuracy_score(y_true, preds))
        print(model,' AP: ', ap)
        print('-----------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()

