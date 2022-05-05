# JIT-defect-models-explainer

###Train models
Train logistic regression models for 2 datasets
```
$ python explain.py -t openstack  LR
```
```
$ python explain.py -t qt  LR
```

Train random forests

```
$ python explain.py -t openstack  RF
```
```
$ python explain.py -t qt  RF
```

### Generate explanations
#### Logistic regression model
Compute an AXp
```
$ python explain.py -x abd -v -C global_model/openstack_LR_global_model.pkl openstack LR
```
```
$ python explain.py -x abd -v -C global_model/qt_LR_global_model.pkl qt LR
```
Compute a CXp:
```
$ python explain.py -x con -v -C global_model/openstack_LR_global_model.pkl openstack LR
```
```
$ python explain.py -x con -v -C global_model/qt_LR_global_model.pkl qt LR
```

Enumerate AXps and CXps for two datasets 
```
$ python explain.py -x abd -N all -vv -C global_model/openstack_LR_global_model.pkl openstack LR
```
```
$ python explain.py -x abd -N all -vv -C global_model/qt_LR_global_model.pkl qt LR
```

#### Random forest model
Compute an AXp
```
$ python explain.py -x abd -v -C global_model/openstack_RF_30estimators_global_model.pkl openstack RF
```
```
$ python explain.py -x abd -v -C global_model/qt_RF_30estimators_global_model.pkl qt RF
```

Compute a CXp
```
$ python explain.py -x con -v -C global_model/openstack_RF_30estimators_global_model.pkl openstack RF
```
```
$ python explain.py -x con -v -C global_model/qt_RF_30estimators_global_model.pkl qt RF
```

Enumerate AXps
```
$ python explain.py -x abd -N all -v -C global_model/openstack_RF_30estimators_global_model.pkl openstack RF
```
```
$ python explain.py -x abd -N all -v -C global_model/qt_RF_30estimators_global_model.pkl qt RF
```

Enumerate CXps
```
$ python explain.py -x con -N all -v -C global_model/openstack_RF_30estimators_global_model.pkl openstack RF
```
```
$ python explain.py -x con -N all -v -C global_model/qt_RF_30estimators_global_model.pkl qt RF
```
