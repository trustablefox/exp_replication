#!/bin/sh
python3 explain.py -t openstack LR
python3 explain.py -t qt LR
python3 explain.py -t openstack RF
python3 explain.py -t qt RF
