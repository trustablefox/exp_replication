#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## xprf.py
##


#
#==============================================================================
from __future__ import print_function
from options import Options
import os
import sys
import pickle
from rndmforest  import XRF, Dataset
import numpy as np
import statistics
from train_global_model import prepare_data, train_global_model, eval_global_model, load_change_metrics_df

#
#==============================================================================
def show_info():
    """
        Print info message.
    """
    print("c RFxp: Random Forest explainer.")
    print('c')

    
#
#==============================================================================
def pickle_save_file(filename, data):
    try:
        f =  open(filename, "wb")
        pickle.dump(data, f)
        f.close()
    except:
        print("Cannot save to file", filename)
        exit()

def pickle_load_file(filename):
    try:
        f =  open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data
    except Exception as e:
        print(e)
        print("Cannot load from file", filename)
        exit()    
        
    
#
#==============================================================================
data_path = './dataset/'

if __name__ == '__main__':
    options = Options(sys.argv)
    options.global_model_name = 'RF'

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    if options.train:
        x_train, x_test, y_train, y_test = prepare_data(options.proj_name, mode='all')
        train_global_model(options.proj_name, x_train, y_train, options.global_model_name)
        eval_global_model(options.proj_name, x_test, y_test, options.global_model_name)

    ### explaining
    if options.xtype:
        print('Explaining the {0} model...\n'.format('logistic regression' if options == 'LR' else 'random forest'))
        # here are some stats
        nofex, minex, maxex, avgex, times = {xtype: [] for xtype in ['abd', 'con']}, \
                                            {xtype: [] for xtype in ['abd', 'con']}, \
                                            {xtype: [] for xtype in ['abd', 'con']}, \
                                            {xtype: [] for xtype in ['abd', 'con']}, []

        # Explain data
        change_metrics, bug_label = load_change_metrics_df(options.proj_name)

        with open(data_path + options.proj_name + '_non_correlated_metrics.txt', 'r') as f:
            metrics = f.read()

        metrics_list = metrics.split('\n')
        non_correlated_change_metrics = change_metrics[metrics_list]

        if not os.path.isdir(data_path+options.proj_name+'.csv'):
            non_correlated_change_metrics['defect'] = bug_label
            non_correlated_change_metrics.to_csv(data_path+options.proj_name+'.csv', index=False)
            non_correlated_change_metrics = non_correlated_change_metrics.drop(['defect'], axis=1)

        data = Dataset(filename=data_path+options.proj_name+'.csv', mapfile=options.mapfile,
                       separator=options.separator, use_categorical=options.use_categorical)

        insts = data.X

        if options.inst is not None:
            inst = np.asarray([float(v.strip()) for v in options.inst.split(',')])
            insts = np.asarray([inst])

        for id, inst in enumerate(insts):
            explainer = XRF(data, options)
            expls = explainer.explain(inst)

            xtypes = ['abd' if options.xtype in ['abd', 'abductive'] else 'con']

            for xtype in xtypes:
                nofex[xtype].append(len(expls[xtype]))
                minex[xtype].append(min([len(e) for e in expls[xtype]]))
                maxex[xtype].append(max([len(e) for e in expls[xtype]]))
                avgex[xtype].append(statistics.mean([len(e) for e in expls[xtype]]))
            times.append(explainer.time)

        if options.verb > 0:
            print('# of insts:', len(insts))
            print('tot # of {0} expls: {1}'.format(xtype, sum(nofex[xtype])))
            print('min # of {0} expls: {1}'.format(xtype, min(nofex[xtype])))
            print('avg # of {0} expls: {1:.2f}'.format(xtype, statistics.mean(nofex[xtype])))
            print('max # of {0} expls: {1}'.format(xtype, max(nofex[xtype])))
            print('')
            print('Min {0} expl sz: {1}'.format(xtype, min(minex[xtype])))
            print('min {0} expl sz: {1:.2f}'.format(xtype, statistics.mean(minex[xtype])))
            print('avg {0} expl sz: {1:.2f}'.format(xtype, statistics.mean(avgex[xtype])))
            print('max {0} expl sz: {1:.2f}'.format(xtype, statistics.mean(maxex[xtype])))
            print('Max {0} expl sz: {1}'.format(xtype, max(maxex[xtype])))
            print('')
            print('tot time: {0:.2f}'.format(sum(times)))
            print('min time: {0:.2f}'.format(min(times)))
            print('avg time: {0:.2f}'.format(statistics.mean(times)))
            print('max time: {0:.2f}'.format(max(times)))
          
            