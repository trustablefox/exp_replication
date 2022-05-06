#!/usr/bin/env python
#-*- coding:utf-8 -*-

#
#==============================================================================
from __future__ import print_function
import statistics
import os
import sys
import numpy as np
from pyexplainer.pyexplainer_pyexplainer import PyExplainer as pyexp
from pyexplainer import pyexplainer_pyexplainer
import pandas as pd
import pickle
import time
import resource
import csv
import random
import lime
import lime.lime_tabular
import shap
#from anchor import utils
from anchor import anchor_tabular

#
#==============================================================================

class HExplainer(object):
    #HeuristicExplainer
    def __init__(self, global_model_name, appr, X_train, y_train, model):
        self.global_model_name = global_model_name
        self.appr = appr
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.init_explainer(appr)

    def init_explainer(self, appr):
        if appr.lower() == 'pyexplainer':
            self.explainer = pyexp(X_train=self.X_train,
                              y_train=self.y_train,
                              indep=self.X_train.columns,
                              dep='defect',
                              blackbox_model=self.model)
        elif appr.lower() == 'lime':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                               # feature_names=self.X_train.columns,
                                                               discretize_continuous=False)
        elif appr.lower() == 'shap':
            self.explainer = shap.Explainer(self.model, self.X_train)

        elif appr.lower() == 'anchor':
            self.explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=[False, True],
                feature_names=self.X_train.columns,
                train_data=self.X_train.values,
                categorical_names={})
        else:
            print('Wrong approach input')
            exit(1)

    def explain(self, X, y):
        pred = self.model.predict(X)[0]

        inst = X.iloc[0]
        preamble = []
        for fid, f in enumerate(inst.index):
            preamble.append(f'{f} = {inst[fid]}')

        print('\n  Explaining: IF {} THEN defect = {}'.format(' AND '.join(preamble), pred))

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

        if self.appr.lower() == 'pyexplainer':
            self.pyexplainer_explain(X, y, pred)
        elif self.appr.lower() == 'lime':
            self.lime_explain(X, y, pred)
        elif self.appr.lower() == 'shap':
            self.shap_explain(X, y, pred)
        elif self.appr.lower() == 'anchor':
            self.anchor_explain(X, y, pred)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        print(f'  time: {self.time}\n')

    def pyexplainer_explain(self, X, y, pred):
        top_k = 3
        rules = self.explainer.explain(X_explain=X, y_explain=y, top_k=top_k, search_function='CrossoverInterpolation')
        top_k_positive_rules = rules['top_k_positive_rules']['rule'].to_list()
        top_k_positive_importance = rules['top_k_positive_rules']['importance'].to_list()
        top_k_negative_rules = rules['top_k_negative_rules']['rule'].to_list()
        top_k_negative_importance = rules['top_k_negative_rules']['importance'].to_list()

        local_rulefit_model = rules['local_rulefit_model']
        all_rules = local_rulefit_model.get_rules()
        all_rules = all_rules[all_rules.coef != 0].sort_values("importance", ascending=False, kind='mergesort')

        if len(top_k_positive_rules) > top_k:
            top_k_positive_rules = top_k_positive_rules[:top_k]
            top_k_positive_importance = top_k_positive_importance[:top_k]

        if len(top_k_negative_rules) > top_k:
            top_k_negative_rules = top_k_negative_rules[:top_k]
            top_k_negative_importance = top_k_negative_importance[:top_k]

        # print(top_k_positive_rules)
        # print(top_k_positive_importance)
        for k, rule in enumerate(top_k_positive_rules):
            print(f'  Top {k} pos expl: {rule}\nimportance: {top_k_positive_importance[k]}')
            print('  size: {0}'.format(len(rule.split(' & '))))
        for k, rule in enumerate(top_k_negative_rules):
            print(f'  Top {k} neg expl: {rule}\nimportance: {top_k_negative_importance[k]}')
            print('  size: {0}'.format(len(rule.split(' & '))))

        print(f'  nof rules: {len(all_rules)}')

    def lime_explain(self, X, y, pred):
        #predict_fn = lambda x: self.model.predict_proba(x).astype(float)

        expl = self.explainer.explain_instance(X.iloc[0, :],
                                          self.model.predict_proba,
                                          # num_features=10, 10 is the default value
                                          top_labels=1)

        prob0, prob1 = self.model.predict_proba(X)[0]
        pred = False if prob0 > prob1 else True
        expl = sorted(expl.as_list(label=int(pred)), key=lambda l: int(l[0]))

        if prob0 == prob1:
            # Reverse the direction of feature importance
            # Since when prob0 = prob1, the target class value is class 1 in the explainer,
            # where the predicted value in the global model is class 0
            expl = list(map(lambda l: (l[0], -l[1]), expl))


        y_expl = list(filter(lambda l: l[1] >= 0, expl))
        ynot_expl = list(filter(lambda l: l[1] < 0, expl))
        print('  expl(pos class):', [(self.X_train.columns[int(f)], imprt) for f, imprt in y_expl])
        print('  size:', len(y_expl))
        print('  expl(neg class):', [(self.X_train.columns[int(f)], imprt) for f, imprt in ynot_expl])
        print('  size:', len(ynot_expl))

        # print('  explanation: IF {0} THEN defect = {1}'. format(' AND '.join(preamble), pred))
        # print('  importance:', importance)
        #print('  size: {0}'.format(len(preamble)))

        #if prob0 == prob1:
        #    exit()

    def shap_explain(self, X, y, pred):
        shap_values = self.explainer.shap_values(X)
        shap_values_sample = shap_values[int(pred)][0] if self.global_model_name == 'RF' else shap_values[0]

        predicted_value = [round(self.explainer.expected_value[idx] + np.sum(shap_values[idx]), 3)
                           for idx in range(len(self.explainer.expected_value))] \
            if self.global_model_name == 'RF' else np.sum(shap_values_sample) + self.explainer.expected_value

        print("base_value = {}, predicted_value = {}".format(self.explainer.expected_value, predicted_value))
        expl = [(idx, shap_values_sample[idx]) for idx in range(len(shap_values_sample))]

        y_expl = list(filter(lambda l: l[1] >= 0, expl))
        ynot_expl = list(filter(lambda l: l[1] < 0, expl))
        print('  expl(pos class):', [(self.X_train.columns[int(f)], imprt) for f, imprt in y_expl])
        print('  size:', len(y_expl))
        print('  expl(neg class):', [(self.X_train.columns[int(f)], imprt) for f, imprt in ynot_expl])
        print('  size:', len(ynot_expl))

    def anchor_explain(self, X, y, pred):
        exp = self.explainer.explain_instance(X.values[0], self.model.predict, threshold=0.95)

        # explanation
        expl = [name for f, name in sorted(zip(exp.features(), exp.names()))]

        preamble = ' AND '.join(expl)

        print('  expl: IF {0} THEN defect = {1}'.format(preamble, pred))
        print('  size:', len(expl))
        #print('  Anchor: %s' % (' AND '.join(exp.names())))
        #print('  Precision: %.2f' % exp.precision())
        #print('  Coverage: %.2f' % exp.coverage())


if __name__ == '__main__':
    proj_name = sys.argv[1]
    global_model_name = sys.argv[2]
    appr = sys.argv[3]
    nof_inst = int(sys.argv[4])
    #batch = int(sys.argv[-1])
    #print('batch:', batch)
    #print('Computing explanations using', appr)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    k2name = {'X_train': './dataset/{}_X_train.csv'.format(proj_name),
              'y_train': './dataset/{}_y_train.csv'.format(proj_name),
              'X_test': './dataset/{}_X_test.csv'.format(proj_name),
              'y_test': './dataset/{}_y_test.csv'.format(proj_name)}

    path_X_train = './dataset/{}_X_train.csv'.format(proj_name)
    path_y_train = './dataset/{}_y_train.csv'.format(proj_name)
    X_train = pd.read_csv(path_X_train)
    y_train = pd.read_csv(path_y_train).iloc[:, 0]
    indep = X_train.columns
    dep = 'defect'

    path_X_explain = './dataset/{}_X_test.csv'.format(proj_name)
    path_y_explain = './dataset/{}_y_test.csv'.format(proj_name)
    X_explain = pd.read_csv(path_X_explain)
    y_explain = pd.read_csv(path_y_explain).iloc[:, 0]

    if global_model_name == 'RF':
        path_model = './global_model/{}_RF_30estimators_global_model.pkl'.format(proj_name)
    else:
        path_model = './global_model/{}_LR_global_model.pkl'.format(proj_name)

    with open(path_model, 'rb') as f:
        model = pickle.load(f)
        
    explainer = HExplainer(global_model_name, appr, X_train, y_train, model)

    """
    
    Explaining
    
    """

    selected_ids = set(range(len(X_explain)))

    if len(X_explain) > nof_inst:
        random.seed(1000)
        selected_ids = set(random.sample(range(len(X_explain)), nof_inst))

    #selected_ids = set(filter(lambda l: l % 90 == batch, range(len(X_explain))))
    #random.seed()

    times = []
    nof_inst = 0

    preds = model.predict(X_explain)

    for i in range(len(X_explain)):

        if i not in selected_ids:
            continue

        nof_inst += 1

        if i < len(X_explain) - 1:
            X = X_explain.iloc[i: i+1,]
            y = y_explain.iloc[i: i+1,]
        else:
            X = X_explain.iloc[i: , ]
            y = y_explain.iloc[i: , ]

        explainer.explain(X, y)

        times.append(explainer.time)

    #print(f'times: {times}\n')
    print()
    print('# of insts:', nof_inst)
    print(f'tot time: {sum(times)}')

    exit()
