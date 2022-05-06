from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
from imblearn.over_sampling import SMOTE

import sys, os,  pickle
from datetime import datetime

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

data_path = './dataset/'
global_model_path = './global_model/'

if not os.path.exists(global_model_path):
    os.makedirs(global_model_path)


def split_train_test_data(feature_df, label, percent_split=70):
    _p_percent_len = int(len(feature_df) * (percent_split / 100))
    x_train = feature_df.iloc[:_p_percent_len]
    y_train = label.iloc[:_p_percent_len]

    x_test = feature_df.iloc[_p_percent_len:]
    y_test = label.iloc[_p_percent_len:]

    return x_train, x_test, y_train, y_test

def prepare_data(proj_name, mode='all'):
    if mode not in ['train', 'test', 'all']:
        print('this function accepts "train","test","all" mode only')
        return

    dt = pd.read_csv(data_path+proj_name+'.csv')
    bug_label = dt['defect']
    dt = dt.drop(['defect'], axis=1)

    x_train, x_test, y_train, y_test = split_train_test_data(dt, bug_label, percent_split=70)

    if mode == 'train':
        return x_train, y_train
    elif mode == 'test':
        return x_test, y_test
    elif mode == 'all':
        return x_train, x_test, y_train, y_test

def train_global_model(proj_name, x_train,y_train, global_model_name = 'RF', n_estimators=30):
    global_model_name = global_model_name.upper()
    if global_model_name not in ['RF','LR']:
        print('wrong global model name. the global model name must be RF or LR')
        return

    smt = SMOTE(k_neighbors=5, random_state=42, n_jobs=24)
    new_x_train, new_y_train = smt.fit_resample(x_train, y_train)
    if global_model_name == 'RF':
        global_model = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=-1)
    elif global_model_name == 'LR':
        global_model = LogisticRegression(random_state=0, n_jobs=-1)

    global_model.fit(new_x_train, new_y_train)

    if global_model_name != 'RF':
        pickle.dump(global_model, open(os.path.join(global_model_path,proj_name+'_'+global_model_name+'_global_model.pkl'),'wb'))
    else:
        pickle.dump(global_model, open(os.path.join(global_model_path,proj_name+'_'+global_model_name+f'_{n_estimators}estimators_global_model.pkl'),'wb'))

def eval_global_model(proj_name, x_test,y_test, global_model_name = 'RF', n_estimators=100):
    global_model_name = global_model_name.upper()
    if global_model_name not in ['RF','LR']:
        print('wrong global model name. the global model name must be RF or LR')
        return

    if global_model_name != 'RF':
        global_model = pickle.load(open(os.path.join(global_model_path,proj_name+'_'+global_model_name+'_global_model.pkl'),'rb'))
    else:
        global_model = pickle.load(open(os.path.join(global_model_path,proj_name+'_'+global_model_name+f'_{n_estimators}estimators_global_model.pkl'),'rb'))

    pred = global_model.predict(x_test)

    prob = global_model.predict_proba(x_test)[:,1]

    auc = roc_auc_score(y_test, prob)
    f1 = f1_score(y_test, pred)

    print('AUC: {}, F1: {}'.format(auc,f1))

if __name__ == '__main__':
    proj_name = sys.argv[1]
    global_model_name = sys.argv[2]

    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    k2df = {'X_train': x_train,
            'y_train': y_train,
            'X_test': x_test,
            'y_test': y_test}

    k2name = {'X_train': './dataset/{}_X_train.csv'.format(proj_name),
            'y_train': './dataset/{}_y_train.csv'.format(proj_name),
            'X_test': './dataset/{}_X_test.csv'.format(proj_name),
            'y_test': './dataset/{}_y_test.csv'.format(proj_name)}

    for k in k2df:
        path = k2name[k]

        if not os.path.isfile(path):
            df = k2df[k]
            df.to_csv(path, index=False)

    train_global_model(proj_name, x_train, y_train,global_model_name)
    eval_global_model(proj_name, x_test,y_test, global_model_name)
