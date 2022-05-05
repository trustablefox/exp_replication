#!/bin/sh

# generate explanations for LR models by LIME, SHAP, Anchor and PyExplainer

python3 hexp.py openstack LR lime 500 > ../logs/0_openstack_LR_lime.log
python3 hexp.py openstack LR shap 500 > ../logs/0_openstack_LR_shap.log
python3 hexp.py openstack LR anchor 500 > ../logs/0_openstack_LR_anchor.log
python3 hexp.py openstack LR pyexplainer 500 > ../logs/0_openstack_LR_pyexplainer.log
python3 hexp.py qt LR lime 500 > ../logs/0_qt_LR_lime.log
python3 hexp.py qt LR shap 500 > ../logs/0_qt_LR_shap.log
python3 hexp.py qt LR anchor 500 > ../logs/0_qt_LR_anchor.log
python3 hexp.py qt LR pyexplainer 500 > ../logs/0_qt_LR_pyexplainer.log

# generate explanations for RF models by LIME, SHAP, Anchor and PyExplainer

python3 hexp.py openstack RF lime 500 > ../logs/0_openstack_RF_lime.log
python3 hexp.py openstack RF shap 500 > ../logs/0_openstack_RF_shap.log
python3 hexp.py openstack RF anchor 500 > ../logs/0_openstack_RF_anchor.log
python3 hexp.py openstack RF pyexplainer 500 > ../logs/0_openstack_RF_pyexplainer.log
python3 hexp.py qt RF lime 500 > ../logs/0_qt_RF_lime.log
python3 hexp.py qt RF shap 500 > ../logs/0_qt_RF_shap.log
python3 hexp.py qt RF anchor 500 > ../logs/0_qt_RF_anchor.log
python3 hexp.py qt RF pyexplainer 500 > ../logs/0_qt_RF_pyexplainer.log
