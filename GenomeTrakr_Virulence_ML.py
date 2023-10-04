#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:34:18 2021
@author: algm

This script uses ML to predict Listeria monocytogenes virulence
form genomic features. 

"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import argparse
import time
import math
from scipy.stats import entropy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


from imblearn.over_sampling import SMOTE

# statistical testing
import scipy.stats as stats
from scipy.stats import uniform, loguniform, randint
from sklearn.decomposition import PCA

# Transformers
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import label_binarize

# Pipelines
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

# Model Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold,GroupShuffleSplit, StratifiedGroupKFold, GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import make_scorer

# Machine Learning
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier       


def grid_search_2(classifier_tuple, param_grid, cv, prepp_, prep_val_, scoring=None):
    print(f"doing grid search{classifier_tuple}")
    # creating the GridSearchCV pipeline
    # using the make_pipline function to create the tuple of (name, transform) 
    # so we don't need to supply the classifier name manually
    
    if prepp_=='var_cut' or prepp_== "var_thresh":
        #search with upsampling    
        search=GridSearchCV(Pipeline([('preprocess',VarianceThreshold(threshold=prep_val_)), ('over', SMOTE(random_state=3 ,k_neighbors=3)) ,classifier_tuple]),
                            param_grid, cv=cv, n_jobs=-1, scoring=scoring, return_train_score=True)
    
    if prepp_=='pca':
        #search with upsampling and PCA
        search=GridSearchCV(Pipeline([('preprocess',PCA(n_components=(prep_val_))), ('over', SMOTE(random_state=3 ,k_neighbors=3)) ,classifier_tuple]),
                            param_grid, cv=cv, n_jobs=-1, scoring=scoring, return_train_score=True)
    
    if prepp_==None or prepp_=="n_snps" or prepp_=="n_pang":
        #search with upsampling 
        search=GridSearchCV(Pipeline([('over', SMOTE(random_state=3 ,k_neighbors=1)) ,classifier_tuple]),
                            param_grid, cv=cv, n_jobs=-1, scoring=scoring, return_train_score=True)
    
    
    # get trainig and validation indices
    # fit the model on the training data and predict on the training data
    # predict on the validation data
    search.fit(X_train.values, np.ravel(y_train))

    return search


def rand_search(classifier_tuple, param_grid, cv, prepp_, prep_val_, scoring=None):
    print(f"doing rand search{classifier_tuple}")
    # creating the GridSearchCV pipeline
    # using the make_pipline function to create the tuple of (name, transform) 
    # so we don't need to supply the classifier name manually
    
    if prepp_=='var_cut' or prepp_== "var_thresh":
        #search with upsampling
        search=RandomizedSearchCV(Pipeline([('preprocess',VarianceThreshold(threshold=prep_val_)), ('over', SMOTE(random_state=3 ,k_neighbors=3)) ,classifier_tuple]),
                            param_grid, cv=cv, n_jobs=-1, n_iter=60, random_state=33, scoring=scoring, return_train_score=True)
    
    if prepp_=='pca':
        #search with upsampling and PCA
        search=RandomizedSearchCV(Pipeline([('preprocess',PCA(n_components=(prep_val_))), ('over', SMOTE(random_state=3 ,k_neighbors=3)) ,classifier_tuple]),
                            param_grid, cv=cv, n_jobs=-1, n_iter=60, random_state=33, scoring=scoring, return_train_score=True)
    
    if prepp_==None or prepp_=="n_snps" or prepp_=="n_pang":
        #search with only upsampling
        search=RandomizedSearchCV(Pipeline([('over', SMOTE(random_state=3 ,k_neighbors=1)) ,classifier_tuple]), 
                            param_grid, cv=cv, n_jobs=-1, n_iter=50, random_state=33, scoring=scoring, return_train_score=True)
    

    # get trainig and validation indices
    # fit the model on the training data and predict on the training data
    # predict on the validation data
    if classifier_tuple[0] == "nn":
        search.fit(X_train.values, np.ravel(y_train))
    else:
        search.fit(X_train, np.ravel(y_train))


    return search

def plot_aucs_individual(est, fold, y_test, y_pred_prob, n_class):
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}
    
    fig, ax = plt.subplots()
    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test[:,i], y_pred_prob[:,i])
        auc_=auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], linestyle='--', label=f'Class {i+1} vs Rest (AUC: {round(auc_,2)})')
    
    plt.title(f'Multiclass ROC curve for {est} / {fold}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outd,'Multiclass ROC'),dpi=300) 
    
    
def plot_mean_rocs(mean_fpr, tprs, aucs, n_classes):
    
    for est_name in tprs.keys():
        
        fig, ax = plt.subplots()
        means_list=[]
        
        # read diagonal which is by Chance
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=.8)
        
        for c in range(n_classes):
            mean_tpr = np.mean(tprs[est_name][c], axis=0)
            mean_tpr[-1] = 1.0
            means_list.append(mean_tpr)
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs[est_name][c])
            ax.plot(mean_fpr, mean_tpr,
                    label=f'Mean ROC Class {c+1} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
                    lw=1, alpha=.8)
        
        macro_tpr=np.mean(means_list, axis=0)
        macro_auc=auc(mean_fpr, macro_tpr)
        
        ax.plot(mean_fpr, macro_tpr, label=f'Macro-average ROC (AUC = {macro_auc:.2f})',
                 color='deeppink', linestyle='--', linewidth=2)
        
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f"ROC for {est_name}", xlabel='False Positive Rate', ylabel='True Positive Rate' )
        ax.legend(loc="lower right")
        plt.savefig(os.path.join(outd,f'ROC_plot_{est_name}'),dpi=300) 
        
        
def box_plot(perf_dict, perf_meassure):
    name={'accuracy':"Accuracy", 'precision':'Precision', 'recall':'Recall', 'f1':'F1'}
    fig,ax=plt.subplots()
    total=[]
    labels=[]
    for key, value in perf_dict.items():
            total.append(value[perf_meassure])
            labels.append(key)

    ax.boxplot(total, labels=labels)
    ax.set_title(f'{name[perf_meassure]} Boxplot for different Estimators')
    ax.set_ylabel(f'{name[perf_meassure]}')

    fig.savefig(os.path.join(outd,f'{name[perf_meassure]}_boxplot'),dpi=300)
    

def hist_normality_check(perf_dict, perf_measure):
    name={'accuracy':"Accuracy", 'precision':'Precision', 'recall':'Recall', 'f1':'F1'}
    est_name={'svc_lin': 'SVM (linear kernel)', 'svc_rad': 'SVM (radial kernel)','rf': 'Random Forrest',
              'adab': 'AdaBoost','nn': 'Neural Network', 'logit': 'Gradient Boost'}
    
    for est, perf_val in perf_dict.items():
        fig, ax = plt.subplots()
        mu, sigma = np.mean(perf_dict['svc_lin']['accuracy']), np.std(perf_dict['svc_lin']['accuracy'])
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        
        sns.distplot(perf_val[perf_measure], hist=True, kde=True, ax=ax, kde_kws={'label':'density_func'})
        #sns.distplot(normd, hist=False, kde=True, ax=ax)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), label='theor. norm_dist')
        ax.set_title(f'{name[perf_measure]} histogram for {est_name[est]}')
        ax.set_xlabel(f'{name[perf_measure]}')
        ax.set_ylabel('Density')
        ax.legend(loc='upper left')
        
def bootstrap_estimator(perf_dict, estimator, n_datapoints, n=100 ):
    perf_array=perf_dict[estimator]
    boot_sample_means=[]
    deci=5
    
    for b in range(0,n):
        np.random.seed(b)
        boot_sample_means.append(np.mean(np.random.choice(perf_array, n_datapoints)))
    
    return np.average(boot_sample_means).round(deci), np.std(boot_sample_means).round(deci), np.percentile(boot_sample_means, [2.5,97.5]).round(deci)


def get_varthres(X, percent_cut):

    # create the variance df
    var=pd.DataFrame(X.var(), columns=['var'])
    var_sorted=var.sort_values('var', ascending=False)
    
    
    # var thresh
    var_cutoff = var['var'].sum()*(1-percent_cut)
    
    var_filt=var_sorted[var_sorted.cumsum()<var_cutoff].dropna()

    return np.round(var_filt.iloc[-1]['var'],4)


def get_variance(X):
    
    # create the variance df
    var=pd.DataFrame(X.var(), columns=['var'])
    var_sorted=var.sort_values('var', ascending=False)

    return var_sorted


def get_entropy(X, nsnps):

    entrpy=[]
    
    ##get the entropy values per columns
    for i in X.columns:        
        entrpy.append(entropy(X[i].value_counts()))

    print("sorting")

    # make a dataframe with the column names and entropy values. 
    entrpy_df=pd.DataFrame(entrpy, index=X.columns)
    
    return entrpy_df
    

def get_entropy_features(X, nsnps, cores):
    
    n_columns=X.shape[1]
    chunk=math.ceil(n_columns/cores)
    chunk_list=[0]
    counter=0
    
    for i in range(cores):
        # print(i)
        if i == cores-1:
            counter=X.shape[1]
        else:
            counter+=chunk
        
        chunk_list.append(counter)
        
    results=Parallel(n_jobs=cores)(delayed(get_entropy)(x, 0.9) for x in [X[X.columns[chunk_list[i]:chunk_list[i+1]]] for i in range(cores)])
    
    ent_df=pd.concat(results).sort_values(0, ascending=(False))[:nsnps]
    
    # return X_ent
    return list(ent_df.index)


def get_variance_features(X, nfeat):
    
    # ent_df=pd.concat(results).sort_values(0, ascending=(False))[:nsnps]
    var_df=get_variance(X)[:nfeat]
        
    # return X_ent
    return list(var_df.index)

def cvfilterloop(X_bin, y_bin, groups, outer=False, inner=False, n_splits=3, cv_folds=10):
    '''
    This function goes through StratifiedGroupKFold splits and sorts out the ones that 
    don't meet certain criteria
    '''
    
    final_train_idxs=[]
    final_test_idxs=[]
    
    # random_state for reproducibility
    d=11
    stp=False
    
    while stp == False:
        
        if outer==True:
            cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=d)
            
        
        if inner==True:
            cv=GroupShuffleSplit(n_splits=50, test_size=0.10, random_state=d)
        
        for train_idxs, test_idxs in cv.split(X_bin, y_bin, groups):
            
            y_array=y_bin.iloc[test_idxs]['ClinFreq'].to_numpy()
            
            if np.count_nonzero(y_array==1) < 1 or np.count_nonzero(y_array==2) < 1 or np.count_nonzero(y_array==3) < 1:
                continue
            else:
                final_train_idxs.append(train_idxs)
                final_test_idxs.append(test_idxs)
                
            
            if len(final_train_idxs) == cv_folds:
                return tuple(zip(final_train_idxs, final_test_idxs))
            
        # increment d
        d=d*2



    
if __name__ == '__main__':
    
    start=time.time()
    
    print(os.path.basename(sys.argv[0]))
    
    ####   PARSE ARGUMENT   ####
    
    parser = argparse.ArgumentParser(description='Input arguments')
    parser.add_argument('-i', action='store', dest='indf', type=str, required=True, 
                        help="Absolute or relative path to inputfile")
    parser.add_argument('-o', action='store', dest='outd', type=str, required=True,
                        help="Absolute path where output files should be stored. Will be created if not yet exists")
    parser.add_argument('-enc', action='store', dest='enc', choices=['absid', 'bins'],
                        help='''Defines how the percent identities are encoded.
                        It can be:
                        absid: converts the percentages in absolute values (eg., 45%% -> 0.45)
                        bins: converts the percentages into 7 discrete bins''')
    parser.add_argument('-pca', action='store', dest='pca', type=float,
                        help='''Percent of variance explained e.g. 0.9 (default). 
                        Number of components are used so that the threshold is met.''')
    parser.add_argument('-var_thresh', action='store', dest='var_thresh', type=float,
                        help='''Defines the variance threshhold for simple Variance Thresholding''')
    parser.add_argument('-npang', action='store', dest='npang', type=int,
                        help='''Defines the number of pangenes used in prediction''')
    parser.add_argument('-nsnps', action='store', dest='nsnps', type=int,
                        help='''Defines how many of the snps columns (with highest entropy) should be used''')
    parser.add_argument('-nreps', action='store', dest='nreps', type=int, default=30,
                        help='''Number of repeated train, test split in the outer two-layer cross-validation''')
                        
    args = parser.parse_args()
    
    indf = os.path.abspath(args.indf)
    outd = args.outd
    enc = args.enc
    n_pca = args.pca
    var_thresh = args.var_thresh
    n_pang= args.npang
    n_snps=args.nsnps
    clustf="data/kma_clustering/GenomeTrakr_cluster_assignments_0.98.txt"
    
    if not os.path.isdir(outd):
        os.makedirs(outd)
    
    clust=pd.read_csv(clustf, sep='\t', index_col=(0), usecols=([0,1]), names=["Genome", "Cluster"])
    clust=clust.rename(index = lambda x: x.split('_genomic.fna')[0]).sort_index()
    
    
    ## load input ##
    
    if "virgenes" in os.path.basename(indf):
        
        level = "Virulence genes"
        
        inp=pd.read_csv(indf, sep=';', index_col=(0), decimal=',').fillna(0)
        
        ### hyperparameters for virgenes ###
        ## adjusted with large_hyperp_search_absid.py
        
        grid_estimators={
            'svc_lin': [SVC(probability=True, max_iter=500000, random_state=(33)),{'svc_lin__kernel': ['linear'],
                                               'svc_lin__C': [0.01,0.05,0.1,0.2,0.5]}]
            }

        rand_estimators={
            'svc_rad': [SVC(probability=True, random_state=(33)),{'svc_rad__kernel': ['rbf'],
                                               'svc_rad__gamma': loguniform(1e-6,1e-1),
                                               'svc_rad__C': loguniform(1e-2,1e1)}],

            'rf': [RandomForestClassifier(random_state=3), {'rf__n_estimators': randint(210,350),
                                                            'rf__criterion': ['gini', 'entropy'],
                                                            'rf__max_features': ['sqrt', 'log2'],
                                                            'rf__max_depth': randint(25,55),
                                                            'rf__min_samples_split': randint(15,40),
                                                            'rf__min_samples_leaf': randint(2,15),
                                                            'rf__bootstrap': [True, False]}],

            'nn': [MLPClassifier(max_iter=10000, early_stopping=True, random_state=1333), {'nn__hidden_layer_sizes': randint(10,40),
                                                                                           'nn__activation': ['identity', 'logistic', 'tanh', 'relu'],
                                                                                           'nn__solver': ['lbfgs', 'adam'],
                                                                                           'nn__alpha': loguniform(1e-4, 1e-1),
                                                                                           'nn__tol': loguniform(1e-3,1e-1),
                                                                                           'nn__n_iter_no_change': randint(400,800)}],

            'logit': [GradientBoostingClassifier(random_state=(33)), {'logit__n_estimators': randint(70,110),
                                                     'logit__criterion': ['friedman_mse', 'squared_error'],
                                                     'logit__learning_rate': [0.01],
                                                     'logit__loss': ['log_loss'],
                                                     'logit__max_features': ['sqrt', 'log2'],
                                                     'logit__max_depth': randint(25,55),
                                                     'logit__min_samples_split': randint(25,50),
                                                     'logit__min_samples_leaf': randint(10,35)}]
            }
    
        
    if "pangenes" in os.path.basename(indf):
        
        level = "Pangenome genes"
        
        inp=pd.read_csv(indf, sep=';', index_col=(0), decimal=',').fillna(0)
        
        
        ### hyperparameters for virgenes ###
        ## adjusted with large_hyperp_search_absid.py
        
        grid_estimators={
            'svc_lin': [SVC(probability=True, max_iter=500000, random_state=(33)),{'svc_lin__kernel': ['linear'], 
                                               'svc_lin__C': [0.0005,0.001,0.05,0.1]}]
            }
    
        rand_estimators={
            'svc_rad': [SVC(probability=True, random_state=(33)),{'svc_rad__kernel': ['rbf'], 
                                               'svc_rad__gamma': loguniform(1e-6,1e-1), 
                                               'svc_rad__C': loguniform(1e5,1e10)}],
        
            'rf': [RandomForestClassifier(random_state=3), {'rf__n_estimators': randint(210,350),
                                                            'rf__criterion': ['gini', 'entropy'],
                                                            'rf__max_features': ['sqrt', 'log2'],
                                                            'rf__max_depth': randint(30,90),
                                                            'rf__min_samples_split': randint(2,20),
                                                            'rf__min_samples_leaf': randint(1,15),
                                                            'rf__bootstrap': [True, False]}],
            
            'nn': [MLPClassifier(max_iter=10000, early_stopping=True, random_state=1333), {'nn__hidden_layer_sizes': randint(20,170),
                                                                                           'nn__activation': ['identity', 'logistic', 'tanh', 'relu'],
                                                                                           'nn__solver': ['lbfgs', 'sgd', 'adam'],
                                                                                           'nn__alpha': loguniform(1e-7, 1e-1),
                                                                                           'nn__tol': loguniform(1e-3,1e-1),
                                                                                           'nn__n_iter_no_change': randint(40,80)}],
            
            'logit': [GradientBoostingClassifier(random_state=(33)), {'logit__n_estimators': randint(10,30),
                                                     'logit__criterion': ['friedman_mse', 'squared_error'],
                                                     'logit__learning_rate': uniform(0.01,0.99),
                                                     'logit__loss': ['log_loss'],
                                                     'logit__max_features': ['sqrt', 'log2'],
                                                     'logit__max_depth': randint(50,110),
                                                     'logit__min_samples_split': randint(15,50),
                                                     'logit__min_samples_leaf': randint(7,40)}]
            }
    
    
    if "snps" in os.path.basename(indf):
        
        level = "snps"
        
        inp=pd.read_csv(indf, sep=';', index_col=(0), header=(0), decimal=',')
        
        n_snps=args.nsnps
        inp=inp[inp.columns[:n_snps]]
        
        X_bin=inp.drop("ClinFreq", axis=1, inplace=False)
        
        
        ### hyperparameters for snps ###
        ## adjusted with large_hyperp_search_binary_hypdist_pangenes_snps.py
        
        grid_estimators={
            'svc_lin': [SVC(probability=True, max_iter=50000, random_state=(33)),{'svc_lin__kernel': ['linear'], 
                                               'svc_lin__C': [0.005,0.01,0.015,0.02]}]
            }
    
        rand_estimators={
            'svc_rad': [SVC(probability=True, random_state=(33)),{'svc_rad__kernel': ['rbf'], 
                                               'svc_rad__gamma': ['auto','scale'], 
                                               'svc_rad__C': loguniform(1e-1,1e0)}],
        
            'rf': [RandomForestClassifier(random_state=3), {'rf__n_estimators': randint(120,180),
                                                            'rf__criterion': ['gini', 'entropy'],
                                                            'rf__max_features': ['sqrt', 'log2'],
                                                            'rf__max_depth': randint(2,10),
                                                            'rf__min_samples_split': randint(10,20),
                                                            'rf__min_samples_leaf': randint(5,20),
                                                            'rf__bootstrap': [True, False]}],
            
            'nn': [MLPClassifier(max_iter=10000, early_stopping=True, random_state=1333), {'nn__hidden_layer_sizes': randint(10,13),
                                                                                           'nn__activation': ['identity', 'logistic', 'tanh', 'relu'],
                                                                                           'nn__solver': ['lbfgs', 'sgd', 'adam'],
                                                                                           'nn__alpha': loguniform(1e-16, 1e-14),
                                                                                           'nn__tol': loguniform(1e-7,1e-4),
                                                                                           'nn__n_iter_no_change': randint(300,600)}],
            
            'logit': [GradientBoostingClassifier(random_state=(33)), {'logit__n_estimators': randint(100,150),
                                                     'logit__criterion': ['friedman_mse', 'squared_error'],
                                                     'logit__learning_rate': [0.01],
                                                     'logit__loss': ['log_loss'],
                                                     'logit__max_features': ['sqrt', 'log2'],
                                                     'logit__max_depth': randint(5,10),
                                                     'logit__min_samples_split': randint(15,25),
                                                     'logit__min_samples_leaf': randint(7,20)}]
            }
        
                                           
    
    ####   DATA EXPLORATION   ####
    
    # create a histogram for the clinical frequency
    bins = np.array([0, 50, 70, 100])
    ax=inp['ClinFreq'].hist(bins=bins,grid=False)
    ax.set_title(f'Clinical Frequency Histogram ({level})')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Clinical Frequency')
    plt.xticks(bins)
    plt.xlim(0,100)
    plt.savefig(os.path.join(outd,'barplot_input_data'),dpi=300) 
    
    
    
    ####   DATA PRE-PROCESSING   ####
    
    if enc == 'bins':
        Xbins=np.array([0,50,60,70,80,90,101])
        X_bin=pd.DataFrame(np.digitize(inp.values[:,1:], Xbins), index=(inp.index))
        
    if enc == 'absid':
        # drop the first row with the clinical freqs and
        # divide the rest by 100 to get absolutes
        X_bin=inp.drop("ClinFreq", axis=1, inplace=False)/100
        
    # bin clinical frequencies
    y_bin=pd.DataFrame(np.digitize(inp.values[:,:1], bins=bins), index=inp.index, columns=["ClinFreq"])
    

    ## define mapper ##
        
    if "blast" in os.path.basename(indf):
        mapper="blast"
    
    if "kma" in os.path.basename(indf):
        mapper = "kma"
    
    if "snps" in os.path.basename(indf):
        mapper = "snps (kma)"
    
    # create a heatmap
    fig, ax2 = plt.subplots(figsize=(15,10))
    sns.heatmap(X_bin, cmap="CMRmap_r", ax=ax2)
    ax2.set_title(f'Binned percent identity heatmap ({mapper})')
    ax2.set_ylabel('Samples')
    ax2.set_xlabel(level)
    fig.savefig(os.path.join(outd,'heatmap_identity'),dpi=300)
    
    
    ####   MODEL BUILDING  ####
    
    ### hyperparameters defined above ###

    # aucs mean over the different folds see example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    # the structure will be a dict with keys for the different estimators and
    # values will be a dictionary for the different classes where the values are
    # the different interp_tpr arrays
    tprs={}
    aucs={}
    mean_fpr = np.linspace(0, 1, 100)
    
    n_classes = len(np.unique(y_bin))
    
    perf_dict = dict()
    best_estimators = dict()
    tuning_searches = dict()
    print_buff=[]
    features=[]
        
    # create custom f1 scoring function
    custom_f1 = make_scorer(f1_score, greater_is_better=True, average="weighted")
    
    n_reps = int(args.nreps)
    
    print_buff.append(' '.join(sys.argv))
    
    groups = clust["Cluster"].to_numpy()
       
    # define outer CV loop
    cv_outer_filt = cvfilterloop(X_bin, y_bin,  groups, outer=True, n_splits=5, cv_folds=n_reps)

    
    for d,(train_idxs, test_idxs) in enumerate(cv_outer_filt):
        
        print("Working on Test_train_split %s..."%(d+1))
        print(f"d={d}")
        print_buff.append(f"Working on Test_train_split {d+1}...")
        # print(y_bin.iloc[test_idxs].value_counts())
        # print(len(train_idxs),len(test_idxs))
        # print("test ids counts ", y_bin.iloc[test_idxs].value_counts())
        
        # set default for current d test-train split
        best_estimators.setdefault(d, dict())
        tuning_searches.setdefault(d, dict())
        
        
        X_train=X_bin.iloc[train_idxs]
        y_train=y_bin.iloc[train_idxs]
        grps_train=groups[train_idxs]
        
        X_test=X_bin.iloc[test_idxs]
        y_test=y_bin.iloc[test_idxs]
        
        y_test_enc = label_binarize(y_test, classes=[1,2,3])
    
        print_buff.append(str(np.unique(y_train, return_counts=True)))
        
        
        #This needs to be here so the functions for filtering happen 
        # in the loop
        # define preprocessing and threshold for different inputs
        if n_pca:
            prepp='pca'
            prep_val=n_pca
        elif var_thresh:
            prepp='var_thresh'
            prep_val=args.var_thresh
        elif n_pang:
            print("n_pang")
            prepp='n_pang'
            prep_val=get_variance_features(X_train, n_pang)
            features.append(prep_val)
            X_train=X_train[prep_val]
            X_test=X_test[prep_val]
        elif n_snps:
            prepp='n_snps'
            prep_val=get_entropy_features(X_train, n_snps, 4)
            features.append(prep_val)
            #X_train is transformed in the search loop
            X_train=X_train[prep_val]
            X_test=X_test[prep_val]
        else:
            prepp=None
            prep_val=None
            features.append(X_bin.columns)
        
        # define inner CV loop
        cv_inner_filt=cvfilterloop(X_train, y_train,  grps_train, inner=True, cv_folds=8)
        

        for est,val in grid_estimators.items():
            tmp_search=grid_search_2((est,val[0]), val[1], cv_inner_filt, prepp_=prepp, prep_val_=prep_val, scoring=custom_f1)
            best_estimators[d].setdefault(est, tmp_search.best_estimator_)
            tuning_searches[d].setdefault(est, tmp_search)
            
    
        for est,val in rand_estimators.items():
            tmp_search=rand_search((est,val[0]), val[1], cv_inner_filt, prepp_=prepp, prep_val_=prep_val, scoring=custom_f1)
            best_estimators[d].setdefault(est, tmp_search.best_estimator_)
            tuning_searches[d].setdefault(est, tmp_search)

            
        # adding the Voting classifier
        # key is voting_class
        # values is a list of tuple (name, estimator/pipeline)
        voting_class=VotingClassifier([x for x in best_estimators[d].items()], voting='soft')
        print("doing voting classifier")
        voting_class.fit(X_train.values, np.ravel(y_train.values))
        
        best_estimators[d].setdefault('maj_voting', voting_class)

            
        for key,est in best_estimators[d].items():
            
            #initialize performance dict
            perf_dict.setdefault(key, dict())
            
            # initialize estimator in tpr,aucs dict
            tprs.setdefault(key, {})
            aucs.setdefault(key, {})
            

            # predict probabilities
            y_pred = est.predict(X_test.values)
            y_pred_enc = label_binarize(y_pred, classes=range(1,n_classes+1))
            y_pred_proba = est.predict_proba(X_test.values)
            
            
            # get info for roc_auc plot
            # loop over classes to get tpr, auc etc.
            for c in range(n_classes):
                
                # function outputs first fpr then tpr
                fpr_tpr = roc_curve(y_test_enc[:,c], y_pred_proba[:,c])
                
                
                interp_tpr = np.interp(mean_fpr, fpr_tpr[0], fpr_tpr[1])
                interp_tpr[0] = 0.0
                # set default for class and append interp_tpr
                tprs[key].setdefault(c, []).append(interp_tpr)
                aucs[key].setdefault(c, []).append(auc(fpr_tpr[0], fpr_tpr[1]))
                

            # performance meassures for each fold
            perf_dict[key].setdefault('accuracy',[]).append(accuracy_score(y_test, y_pred))
            perf_dict[key].setdefault('precision',[]).append(precision_score(y_test, y_pred, average='weighted', zero_division=1))
            perf_dict[key].setdefault('recall',[]).append(recall_score(y_test, y_pred, average='weighted', zero_division=1))
            perf_dict[key].setdefault('f1',[]).append(f1_score(y_test, y_pred, average='weighted', zero_division=1))
            perf_dict[key].setdefault('auc',[]).append(np.mean([x[1][d] for x in aucs[key].items()]))
            perf_dict[key].setdefault('mcc',[]).append(matthews_corrcoef(y_test, y_pred))
                
            # add number of features used:
            print_buff.append(f"Number of features: {best_estimators[d]['svc_lin'].named_steps['svc_lin'].n_features_in_}")
             
            # print training, test, and validation scores
            if key == 'maj_voting':
                print_buff.append(f"{key}:\n valid_f1: {perf_dict[key]['f1'][d]}\n\n")
            else:
                print_buff.append(f"{key}:\n train_f1: {tuning_searches[d][key].score(X_train.values,y_train)}\n valid_f1: {tuning_searches[d][key].cv_results_['mean_test_score'][tuning_searches[d][key].best_index_]}\n test_f1: {perf_dict[key]['f1'][d]}")

        
        # Number of pca components used
        if n_pca:
            print_buff.append(f"# pca components used ({n_pca}): {best_estimators[d]['svc_rad'].named_steps['preprocess'].n_components_}")
    
    
    
    print_buff.append("/n/nThe training of the models took %ss"%(time.time()-start))
    
    # open filehandle for logfile
    log_outf = open(os.path.join(outd,"training_testing_log.txt"), "w")
    log_outf.seek(0)
    log_outf.write("\n".join(print_buff))
    log_outf.truncate() 

    #close log filehandle
    log_outf.close()
    
    
    ## plot mean ROC-AUC curves
    plot_mean_rocs(mean_fpr, tprs, aucs, n_classes)
    
    ## plot boxplot
    box_plot(perf_dict, 'accuracy')
    box_plot(perf_dict, 'f1')
    
    
    ## save important files
    best_est_outf = open(os.path.join(outd,"best_estimators_dict.pickle"), "wb")
    performance_outf = open(os.path.join(outd,"performance_dict.pickle"), "wb")
    
    pickle.dump(best_estimators, best_est_outf)
    pickle.dump(perf_dict, performance_outf)
    
    best_est_outf.close()
    performance_outf.close()

    print("This took %ss"%(time.time()-start))

    
    # finally get the best of the best models (best of best aka, bob) and then fit them on entire dataset
    
    final_trained={}
    perfm="f1"
    
    for model, perfv in perf_dict.items():
        
        bobest_index = perfv[perfm].index(max(perfv[perfm]))
        bobest_feats = features[bobest_index]
        bobest_fit = best_estimators[bobest_index][model].fit(X_bin[bobest_feats], np.ravel(y_bin))
        final_trained.setdefault(model, [bobest_fit, bobest_feats])
    
    # save best of best trained
    with open(os.path.join(outd, f'final_trained_models_{perfm}.pickle'), 'wb') as final_trained_outf:
        pickle.dump(final_trained, final_trained_outf)
    
    
    ### Statistical testing ###
    # Bootstrapping
    
    ## get statistics dataframe
    statistics=pd.DataFrame(index=['accuracy', 'accuracy_std', 'accuracy_CI', 'precision', 'precision_std', 'precision_CI', 'recall', 'recall_std', 'recall_CI', 'f1', 'f1_std', 'f1_CI', 'auc', 'auc_std', 'auc_CI', 'mcc', 'mcc_std', 'mcc_CI'], dtype="float32")
    
    # fill dataframe
    for key, val in perf_dict.items():
        add={}
        for k, v in val.items():
            _boot_res = bootstrap_estimator(perf_dict[key], k , n_reps, 100)
            add.setdefault(k, _boot_res[0])
            add.setdefault(k+'_std', _boot_res[1])
            add.setdefault(k+'_CI', _boot_res[2])
        statistics[key]=pd.Series(add)
        
    
    # save statistics dataframe to txt file
    with open(os.path.join(outd, 'statistics_report.txt'), 'w') as stats_outf:
        statistics.to_string(stats_outf)
        
    # save statistics dataframe to pickle 
    with open(os.path.join(outd, 'statistics_report.pickle'), 'wb') as stats_pickle_outf:
        pickle.dump(statistics, stats_pickle_outf)
        
    # save features list of lists
    with open(os.path.join(outd, 'features.pickle'), 'wb') as feats_outf:
        pickle.dump(features, feats_outf)

