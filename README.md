<p align="center">
<img src="img/logo.png" width="100" height="100">
 
<div align="center">
<h1>
    <b>
     FoX: Trustable Just-In-Time Explanations
    </b>
</h1>
</div>

</p>

## Table of Contents
* **[Replication of Experiments](#replication-of-experiments)**
  * [Install Prerequisites](#install-prerequisites)
  * [Replication of Enumeration Explanations 0](#replication-of-enumeration-explanations-0)
  * [Replication of Enumeration Explanations 1](#replication-of-enumeration-explanations-1)
  * [Parse Logs](#parse-logs)
  * [Replicate RQ1 (Correctness)](#replicate-rq1-correctness)
  * [Replicate RQ2 (Robustness)](#replicate-rq2-robustness)
  * [Replicate RQ3 (Time)](#replicate-rq3-runtime)

## Replication of Experiments

### Install Prerequisites
To install the required packages, please run:
```bash
pip3 install -r requirements.txt
```

### Train LR and RF Models for Openstack and Qt
```bash
cd exp_replication/src
./experiment/experiment_train.sh
```

### Replication of Enumeration Explanations 0
First time to run the set of experiments. All logs are saved in *logs*. Note that the experiments take a while
```bash
cd exp_replication/src
./experiment/experiment_fox_0.sh && ./experiment/experiment_other_0.sh
```

### Replication of Enumeration Explanations 1
Second time to run the set of experiments. All logs are saved in *exp_replication/logs*. Note that the experiments take a while
```bash
cd exp_replication/src
./experiment/experiment_fox_1.sh && ./experiment/experiment_other_1.sh
```

### Parse logs
Parsing the logs. Explanations in each log are stored in a json file in *exp_replication/expls*
```bash
cd exp_replication/src
./experiment/experiment_plogs.sh
```

### Replicate RQ1 Correctness
The results are saved in *exp_replication/res/csv/rq1_correctness.csv*. Note that this will take a while
```bash
cd exp_replication/src
./experiment/experiment_rq1.sh
```

### Replicate RQ2 Robustness
The results are saved in *exp_replication/res/csv/rq2_robust.csv*. Note that this will take a while
```bash
cd exp_replication/src
./experiment/experiment_rq2.sh
```

### Replicate RQ3 Runtime
The results are saved in *exp_replication/res/csv/rq3_runtime.csv*. Note that this will take a while
```bash
cd exp_replication/src
./experiment/experiment_rq3.sh
```

For more information of training and explaining each model, please click [here](https://github.com/foxplainer/foxplainer/tree/main/exp_replication/src).
