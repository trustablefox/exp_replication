#!/bin/sh

# generate explanation for both two models by our approach

python3 explain.py -vv -C ./global_model/openstack_LR_global_model.pkl -i ./dataset/openstack_X_test.csv --nof-inst 500 -N all openstack LR > ../logs/1_openstack_LR_formal_all.log
python3 explain.py -vv -C ./global_model/qt_LR_global_model.pkl -i ./dataset/qt_X_test.csv --nof-inst 500 -N all qt LR > ../logs/1_qt_LR_formal_all.log
python3 explain.py -vv -C ./global_model/openstack_RF_30estimators_global_model.pkl -i ./dataset/openstack_X_test.csv --nof-inst 500 -N all openstack RF > ../logs/1_openstack_RF_formal_all.log
python3 explain.py -vv -C ./global_model/qt_RF_30estimators_global_model.pkl -i ./dataset/qt_X_test.csv --nof-inst 500 -N all qt RF > ../logs/1_qt_RF_formal_all.log
