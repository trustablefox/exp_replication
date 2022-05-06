from __future__ import print_function
import os
import glob
import json
import csv
from correctness_lr import LRExplainer
from options import Options
import sys
from correctness_rf import XRF, Dataset
import numpy as np
import pandas as pd

def correct(j, dtn2m):
    global_model_name = j.rsplit('/', 1)[-1].rsplit('_', 1)[-1].split('.json')[0]
    explainer = j.rsplit('/', 1)[-1].split('_')[1]

    with open(j, 'r') as f:
        info_dict = json.load(f)

    stats = info_dict['stats']

    openstack_X_test = pd.read_csv('./dataset/openstack_X_test.csv')
    qt_X_test = pd.read_csv('./dataset/qt_X_test.csv')
    dt2df = {'openstack': openstack_X_test,
             'qt': qt_X_test}

    rows = check_correct(explainer, stats, dtn2m, global_model_name)

    correct_fn = j.replace('../expls/', './stats/correctness/').replace('.json', '.csv')

    crt_dir = correct_fn[ : correct_fn.rfind('/') + 1]
    if not os.path.isdir(crt_dir):
        os.makedirs(crt_dir)

    with open(correct_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(rows[0])
        write.writerows(rows[1:])

def check_correct(appr, stats, dtn2m, global_model_name):
    rows = []
    rows.append(['dataset', 'inst', 'predrule', 'isaxp'])
    i = 0

    data_path = './dataset/'
    proj_name2data = {proj_name: Dataset(filename=data_path + proj_name + '.csv') for proj_name in ['openstack', 'qt']}
    proj_name2insts = {proj_name: pd.read_csv(data_path + proj_name + '_X_test.csv') for proj_name in ['openstack', 'qt']}

    insts = sorted(stats.keys())

    for id , inst in enumerate(insts):

        dt = inst.split('_')[0]

        args = ['exp_results.py', '-vv', '-C', dtn2m[(dt, global_model_name)], dt, global_model_name]
        options = Options(args)

        data = proj_name2data[options.proj_name]

        if options.global_model_name == 'RF':
            explainer = XRF(data, options)
        else:
            explainer = LRExplainer(data, options)

        hexpl_ = stats[inst]['predrule'] if appr != 'formal' else stats[inst]['abd']
        pred = stats[inst]['pred']
        hexpl_ = hexpl_.split(' AND ')
        hexpl_ = list(map(lambda l: l.split(), hexpl_))
        hexpl = {}
        columns = list(proj_name2insts[options.proj_name].columns)
        for h in hexpl_:
            val = h[:]
            if len(h) == 3:
                f = 'f{0}'.format(columns.index(h[0]))
                val[0] = f
                val[-1] = float(val[-1])
            else:
                assert len(h) == 5
                f = 'f{0}'.format(columns.index(h[2]))
                val[2] = f
                val[0] = float(val[0])
                val[-1] = float(val[-1])
            hexpl[f] = val

        isAXp = explainer.isAXp(stats[inst]['inst'], hexpl, columns, pred)

        if appr != 'formal':
            row = [dt, stats[inst]['inst'], stats[inst]['predrule'], isAXp]
        else:
            row = [dt, stats[inst]['inst'], stats[inst]['abd'], isAXp]
        rows.append(row)
    return rows

def robustness(j):
    #global_model_name = j.rsplit('/', 1)[-1].rsplit('_', 1)[-1].split('.json')[0]
    explainer = j.rsplit('/', 1)[-1].split('_')[1]

    with open(j, 'r') as f:
        stats_0 = json.load(f)['stats']

    with open(j.replace('/0_', '/1_'), 'r') as f:
        stats_1 = json.load(f)['stats']

    if explainer == 'pyexplainer':
        rows = robust_pyexplainer(stats_0, stats_1)
    elif explainer == 'lime':
        rows = robust_lime(stats_0, stats_1)
    elif explainer == 'shap':
        rows = robust_shap(stats_0, stats_1)
    elif explainer == 'anchor':
        rows = robust_anchor(stats_0, stats_1)
    elif explainer == 'formal':
        rows = robust_formal(stats_0, stats_1)

    robust_fn = j.replace('../expls/0_', './stats/robust/').replace('.json', '.csv')
    robust_dir = robust_fn[: robust_fn.rfind('/') + 1]
    if not os.path.isdir(robust_dir):
        os.makedirs(robust_dir)

    with open(robust_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(rows[0])
        write.writerows(rows[1:])

def robust_pyexplainer(stats_0, stats_1):
    rows = []
    rows.append(['dataset', 'predrule0', 'predimprt0',
                 'predrule1', 'predimprt1', 'same'])

    for inst in stats_0:
        dt = inst.split('_')[0]

        predrule0 = stats_0[inst]['predrule']
        predrule1 = stats_1[inst]['predrule']
        pred0 = stats_0[inst]['pred']
        pred1 = stats_1[inst]['pred']
        try:
            predimprt0 = [stats_0[inst]['predimprt']]
            predimprt1 = [stats_1[inst]['predimprt']]
        except:
            predimprt0 = [1]
            predimprt1 = [1]

        assert pred0 == pred1

        row = []
        row.append(dt)
        row.append(f'IF {predrule0} THEN defect = {pred0}')
        row.append(predimprt0)
        row.append(f'IF {predrule1} THEN defect = {pred1}')
        row.append(predimprt1)
        row.append(rules_same(predrule0, predrule1, predimprt0, predimprt1))
        rows.append(row)

    return rows

def robust_lime(stats_0, stats_1):
    rows = []
    rows.append(['dataset', 'predrule0', 'predimprt0',
                 'predrule1', 'predimprt1', 'same'])

    for inst in stats_0:
        dt = inst.split('_')[0]

        predrule0 = stats_0[inst]['predrule']
        predrule1 = stats_1[inst]['predrule']
        pred0 = stats_0[inst]['pred']
        pred1 = stats_1[inst]['pred']
        try:
            predimprt0 = stats_0[inst]['predimprt']
            predimprt1 = stats_1[inst]['predimprt']
        except:
            predimprt0 = [1]
            predimprt1 = [1]

        assert pred0 == pred1

        row = []
        row.append(dt)
        row.append(f'IF {predrule0} THEN defect = {pred0}')
        row.append(predimprt0)
        row.append(f'IF {predrule1} THEN defect = {pred1}')
        row.append(predimprt1)
        row.append(rules_same(predrule0, predrule1, predimprt0, predimprt1))
        rows.append(row)

    return rows

def robust_shap(stats_0, stats_1):
    return robust_lime(stats_0, stats_1)

def robust_anchor(stats_0, stats_1):
    return robust_lime(stats_0, stats_1)

def robust_formal(stats_0, stats_1):
    rows = []
    rows.append(['dataset', 'axp0', 'axp1', 'axpsame', 'cxp0', 'cxp1', 'cxpsame'])

    for inst in stats_0:
        dt = inst.split('_')[0]

        row = []
        row.append(dt)
        for xtype in ['abd', 'con']:
            predrule0 = stats_0[inst][xtype]
            predrule1 = stats_1[inst][xtype]
            pred0 = stats_0[inst]['pred']
            pred1 = stats_1[inst]['pred']

            assert pred0 == pred1

            row.append(f'IF {predrule0} THEN defect {"=" if xtype == "abd" else "!="} {pred0}')
            row.append(f'IF {predrule1} THEN defect {"=" if xtype == "abd" else "!="} {pred1}')
            row.append(rules_same(predrule0, predrule1, [1], [1]))
            rows.append(row)

    return rows

def rules_same(rule0, rule1, imprt0, imprt1):

    rule0_ = sorted(map(lambda l: l.strip(), rule0.split(' AND ')))
    rule1_ = sorted(map(lambda l: l.strip(), rule1.split(' AND ')))

    if len(rule0_) != len(rule1_):
        return False

    rule0_ = list(map(lambda l: l.split(), rule0_))
    rule1_ = list(map(lambda l: l.split(), rule1_))
    for i in range(len(rule0_)):
        r0 = rule0_[i]
        r1 = rule1_[i]

        if len(r0) != len(r1):
            return False

        string_idx = [0, 1] if len(r0) == 3 else [1, 2, 3]

        for j in range(len(r0)):
            if j in string_idx:
                if r0[j] != r1[j]:
                    return False
            else:
                if not np.isclose([float(r0[-1])], [float(r1[-1])])[0]:
                    return False

    close = np.isclose(imprt0, imprt1)

    for c in close:
        if c == False:
            return False
    return True


def rtime(j):
    explainer = j.rsplit('/', 1)[-1].split('_')[1]

    with open(j, 'r') as f:
        stats_0 = json.load(f)['stats']

    with open(j.replace('/0_', '/1_'), 'r') as f:
        stats_1 = json.load(f)['stats']

    if explainer == 'pyexplainer':
        rows = rtime_pyexplainer(stats_0, stats_1)
        pass
    elif explainer == 'lime':
        rows = rtime_lime(stats_0, stats_1)
        pass
    elif explainer == 'shap':
        rows = rtime_shap(stats_0, stats_1)
        pass
    elif explainer == 'anchor':
        rows = rtime_anchor(stats_0, stats_1)
        pass
    elif explainer == 'formal':
        rows = rtime_formal(stats_0, stats_1)
        pass

    rtime_fn = j.replace('../expls/0_', './stats/rtime/').replace('.json', '.csv')
    rtime_dir = rtime_fn[: rtime_fn.rfind('/') + 1]
    if not os.path.isdir(rtime_dir):
        os.makedirs(rtime_dir)

    with open(rtime_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(rows[0])
        write.writerows(rows[1:])

def rtime_pyexplainer(stats_0, stats_1):
    rows = []
    header = ['dataset', 'rtimeinst', 'rtime3', 'rtime2000' ]
    rows.append(header)

    for inst in stats_0:
        dt = inst.split('_')[0]
        rtime0 =stats_0[inst]['rtime']
        rtime1 =stats_1[inst]['rtime']
        nof_rules0 = stats_0[inst]['nofrules']
        nof_rules1 = stats_1[inst]['nofrules']

        row = []
        row.append(dt)
        row.append(round((rtime0 + rtime1) / 2, 6))
        row.append(round(((rtime0 / 3) + (rtime1 / 3)) / 2, 6))
        row.append(round(((rtime0 / nof_rules0) + (rtime1 / nof_rules1)) / 2, 6))
        rows.append(row)
    return rows

def rtime_lime(stats_0, stats_1):
    rows = []
    header = ['dataset', 'rtimeinst', 'rtime']
    rows.append(header)

    for inst in stats_0:
        dt = inst.split('_')[0]
        rtime0 =stats_0[inst]['rtime']
        rtime1 =stats_1[inst]['rtime']

        row = []
        row.append(dt)
        row.append(round((rtime0 + rtime1) / 2, 6))
        row.append(round((rtime0 + rtime1) / 2, 6))
        rows.append(row)
    return rows

def rtime_shap(stats_0, stats_1):
    return rtime_lime(stats_0, stats_1)

def rtime_anchor(stats_0, stats_1):
    return rtime_lime(stats_0, stats_1)

def rtime_formal(stats_0, stats_1):
    rows = []
    header = ['dataset', 'axprtimeinst', 'axprtime', 'cxprtimeinst', 'cxprtime']
    rows.append(header)

    for inst in stats_0:
        dt = inst.split('_')[0]

        abdrtime0 = stats_0[inst]['abdrtime']
        abdrtime1 = stats_1[inst]['abdrtime']
        nof_abds0 = stats_0[inst]['nofabds']
        nof_abds1 = stats_1[inst]['nofabds']

        conrtime0 = stats_0[inst]['conrtime']
        conrtime1 = stats_1[inst]['conrtime']
        nof_cons0 = stats_0[inst]['nofcons']
        nof_cons1 = stats_1[inst]['nofcons']

        row = []
        row.append(dt)
        row.append(round((abdrtime0 + abdrtime1 ) / 2, 6))
        row.append(round(((abdrtime0 / nof_abds0) + (abdrtime1 / nof_abds1)) / 2, 6))
        row.append(round((conrtime0 + conrtime1 ) / 2, 6))
        row.append(round(((conrtime0 / nof_cons0) + (conrtime1 / nof_cons1)) / 2, 6))
        rows.append(row)
    return rows

def size(j):
    explainer = j.rsplit('/', 1)[-1].split('_')[1]

    with open(j, 'r') as f:
        stats_0 = json.load(f)['stats']

    with open(j.replace('/0_', '/1_'), 'r') as f:
        stats_1 = json.load(f)['stats']

    if explainer == 'pyexplainer':
        rows = size_pyexplainer(stats_0, stats_1)
        pass
    elif explainer == 'lime':
        rows = size_pyexplainer(stats_0, stats_1)
        pass
    elif explainer == 'shap':
        rows = size_pyexplainer(stats_0, stats_1)
        pass
    elif explainer == 'anchor':
        rows = size_pyexplainer(stats_0, stats_1)
        pass
    elif explainer == 'formal':
        rows = size_formal(stats_0, stats_1)
        pass

    rtime_fn = j.replace('log_json/0_', 'stats/size/0_').replace('.json', '.csv')

    with open(rtime_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(rows[0])
        write.writerows(rows[1:])

def size_pyexplainer(stats_0, stats_1):
    rows = []
    header = ['dataset', 'size' ]
    rows.append(header)

    for inst in stats_0:
        dt = inst.split('_')[0]
        size_0 =stats_0[inst]['predlits']

        row = []
        row.append(dt)
        row.append(size_0)
        rows.append(row)
    return rows

def size_formal(stats_0, stats_1):
    rows = []
    header = ['dataset', 'axpsize', 'cxpsize' ]
    rows.append(header)

    for inst in stats_0:
        dt = inst.split('_')[0]
        axpsize_0 =stats_0[inst]['abdlits']
        cxpsize_0 =stats_0[inst]['conlits']

        row = []
        row.append(dt)
        row.append(axpsize_0)
        row.append(cxpsize_0)
        rows.append(row)
    return rows

if __name__ == '__main__':
    json_path = '../expls'
    jsons = []

    rq = sys.argv[1].strip().lower()

    jsons = glob.glob(json_path + '/*.json')

    dtn2m = {}

    for dt in ['openstack', 'qt']:
        dtn2m[(dt, 'LR')] = './global_model/{0}_LR_global_model.pkl'.format(dt)
        dtn2m[(dt, 'RF')] = './global_model/{0}_RF_30estimators_global_model.pkl'.format(dt)


    explainers = ['pyexplainer', 'lime', 'shap', 'anchor', 'formal']
    global_model_names = ['LR', 'RF']

    if rq in ('correct', 'correctness'):
        print('correctness')
        # correctness
        for j in jsons:
            time = j.rsplit('/', 1)[-1][0]
            if time != '0':
                continue
            correct(j, dtn2m)

        df_correct = pd.DataFrame([], columns=['dataset', 'correctness', 'model', 'approach'])
        for explainer in explainers:
            for model in global_model_names:
                file = './stats/correctness/0_{0}_{1}.csv'.format(explainer, model)
                df_f = pd.read_csv(file)
                df_f['correctness'] = df_f['isaxp'].astype(int)
                df_f = df_f.groupby('dataset').mean()
                df_f = df_f.reset_index()
                df_f['model'] = model
                df_f['approach'] = explainer if explainer != 'formal' else 'FoX'
                df_correct = df_correct.append(df_f[['dataset', 'correctness', 'model', 'approach']])

        if not os.path.isdir('../res/csv'):
            os.makedirs('../res/csv')

        df_correct.to_csv('../res/csv/rq1_correctness.csv', index=False)

    elif rq in ('robust', 'robustness'):
        print('robustness')
        # robustness
        for j in jsons:
            time = j.rsplit('/', 1)[-1][0]
            if time != '0':
                continue
            robustness(j)

        df_robust = pd.DataFrame([], columns=['dataset', 'robustness', 'model', 'approach'])
        for explainer in explainers:
            for model in global_model_names:
                file = './stats/robust/{0}_{1}.csv'.format(explainer, model)
                df_f = pd.read_csv(file)
                df_f = df_f.groupby('dataset').mean()
                if explainer != 'formal':
                    df_f.loc[df_f['same'] == False, 'same'] = 0
                    df_f.loc[df_f['same'] == True, 'same'] = 1
                    df_f['robustness'] = df_f['same']
                else:
                    df_f.loc[df_f['axpsame'] == False, 'axpsame'] = 0
                    df_f.loc[df_f['axpsame'] == True, 'axpsame'] = 1
                    df_f['robustness'] = df_f['axpsame']
                df_f['model'] = model
                df_f['approach'] = explainer if explainer != 'formal' else 'FoX'
                df_f = df_f.reset_index()
                df_robust = df_robust.append(df_f[['dataset', 'robustness', 'model', 'approach']])

        if not os.path.isdir('../res/csv'):
            os.makedirs('../res/csv')

        df_robust.to_csv('../res/csv/rq2_robust.csv', index=False)
    else:
        print('runtime')
        # runtime
        for j in jsons:
            time = j.rsplit('/', 1)[-1][0]
            if time != '0':
                continue
            rtime(j)

        df_rtime_expl = pd.DataFrame([], columns=['dataset', 'rtime', 'model', 'approach'])
        for explainer in explainers:
            for model in global_model_names:
                file = './stats/rtime/{0}_{1}.csv'.format(explainer, model)
                df_f = pd.read_csv(file)
                df_f['model'] = model
                if explainer == 'pyexplainer':
                    df_f['rtime'] = df_f['rtime2000']
                    df_f['approach'] = 'pyexplainer'
                    df_rtime_expl = df_rtime_expl.append(df_f[['dataset', 'rtime', 'model', 'approach']])

                elif explainer != 'formal':
                    df_f['approach'] = explainer
                    df_rtime_expl = df_rtime_expl.append(df_f[['dataset', 'rtime', 'model', 'approach']])

                else:
                    xtypes = ['axp', 'cxp']
                    for xtype in xtypes:
                        df_f['rtime'] = df_f['{0}rtime'.format(xtype)]
                        df_f['approach'] = '{0}_{1}'.format(explainer, xtype)
                        df_rtime_expl = df_rtime_expl.append(df_f[['dataset', 'rtime', 'model', 'approach']])

        df_rtime_expl.loc[(df_rtime_expl['approach'] == 'formal_axp'), 'approach'] = 'FoX_AXP'

        df_rtime_expl.loc[(df_rtime_expl['approach'] == 'formal_cxp'), 'approach'] = 'FoX_CXP'

        if not os.path.isdir('../res/csv'):
            os.makedirs('../res/csv')

        df_rtime_expl.to_csv('../res/csv/rq3_runtime.csv', index=False)

