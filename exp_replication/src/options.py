#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## options.py
##


#
#==============================================================================
from __future__ import print_function
import getopt
import math
import os
import sys


#
#==============================================================================
class Options(object):
    """
        Class for representing command-line options.
    """

    def __init__(self, command):
        """
            Constructor.
        """

        self.accmin = 0.95
        self.classifier = None
        self.encode = 'none'
        self.files = None
        self.inst = None
        self.mapfile = None
        self.maxdepth = 3
        self.n_estimators = 30
        self.nof_inst = None
        self.output = 'global_model'
        self.reduce = 'lin'
        self.seed = 7
        self.separator = ','
        self.smallest = False
        self.solver = 'g3'
        self.testsplit = 0.3
        self.train = False
        self.unit_mcs = False
        self.use_categorical = False
        self.validate = False
        self.verb = 0
        self.xnum = 1
        self.xtype = None

        if command:
            self.parse(command)

        if self.xnum != 1:
            self.xtype = 'abd'

        if self.inst and self.xtype is None:
            self.xtype = 'abd'

        if self.xtype and self.inst is None:
            print('Please indicate the dataset to be explained')
            exit(1)


    def parse(self, command):
        """
            Parser.
        """
        self.command = command

        try:
            opts, args = getopt.getopt(command[1:],
                                    'a:cC:d:e:hi:I:m:Mn:N:ho:r:s:tu:vVx:',
                                    ['accmin=',
                                     'classifier=',
                                     'encode=',
                                     'help',
                                     'inst=',
                                     'nof-inst=',
                                     'map-file=',
                                     'use-categorical=',
                                     'maxdepth=',
                                     'minimum',
                                     'nbestims=',
                                     'output=',
                                     'seed=',
                                     'sep=',
                                     'solver=',
                                     'testsplit=',
                                     'train',
                                     'unit-mcs',
                                     'validate',
                                     'verbose',
                                     'xnum'
                                     ])
        except getopt.GetoptError as err:
            sys.stderr.write(str(err).capitalize())
            self.usage()
            sys.exit(1)

        for opt, arg in opts:
            if opt in ('-a', '--accmin'):
                self.accmin = float(arg)
            elif opt in ('-c', '--use-categorical'):
                self.use_categorical = True
            elif opt in ('-C', '--classifier'):
                self.classifier = str(arg)
            elif opt in ('-d', '--maxdepth'):
                self.maxdepth = int(arg)
            elif opt in ('-e', '--encode'):
                self.encode = str(arg)
            elif opt in ('-h', '--help'):
                self.usage()
                sys.exit(0)
            elif opt in ('-i', '--inst'):
                self.inst = str(arg)
            elif opt in ('-I', '--nof-inst'):
                self.nof_inst = int(arg)
            elif opt in ('-m', '--map-file'):
                self.mapfile = str(arg)
            elif opt in ('-M', '--minimum'):
                self.smallest = True
            elif opt in ('-n', '--nbestims'):
                self.n_estimators = int(arg)
            elif opt in ('-N', '--xnum'):
                self.xnum = str(arg)
                self.xnum = -1 if self.xnum == 'all' else int(arg)
            elif opt in ('-o', '--output'):
                self.output = str(arg)
            elif opt in ('-r', '--reduce'):
                self.reduce = str(arg)
            elif opt == '--seed':
                self.seed = int(arg)
            elif opt == '--sep':
                self.separator = str(arg)
            elif opt in ('-s', '--solver'):
                self.solver = str(arg)
            elif opt == '--testsplit':
                self.testsplit = float(arg)
            elif opt in ('-t', '--train'):
                self.train = True
            elif opt in ('-u', '--unit-mcs'):
                self.unit_mcs = True
            elif opt in ('-v', '--verbose'):
                self.verb += 1
            elif opt in ('-V', '--validate'):
                self.validate = True
            elif opt in ('-x', '--xtype'):
                self.xtype = str(arg)

            else:
                assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

        if self.encode == 'none':
            self.encode = None
        self.files = args
        self.proj_name = args[0]
        self.global_model_name = args[1]

    def usage(self):
        """
            Print usage message.
        """

        print('Usage: ' + os.path.basename(self.command[0]) + ' [options] input-file')
        print('Options:')
        #print('        -a, --accmin=<float>       Minimal accuracy')
        #print('                                   Available values: [0.0, 1.0] (default = 0.95)')
        #print('        -c, --use-categorical      Treat categorical features as categorical (with categorical features info if available)')
        print('        -C, --classifier=<str>     Path to the global model')
        #print('        -d, --maxdepth=<int>       Maximal depth of a tree')
        #print('                                   Available values: [1, INT_MAX] (default = 3)')
        #print('        -e, --encode=<smt>         Encode a previously trained model')
        #print('                                   Available values: sat, maxsat, none (default = none)')
        print('        -h, --help                 Show this message')
        print('        -i, --inst=<str>           Path to the test/explain data')
        print('        -I, --nof-inst=<int>       The number of instances being explained')
        print('                                   Available values: [1, INT_MAX] (default: None, i.e. all)')
        #print('        -m, --map-file=<str>       Path to a file containing a mapping to original feature values. (default: none)')
        #print('        -M, --minimum              Compute a smallest size explanation (instead of a subset-minimal one)')
        print('        -n, --nbestims=<int>       Number of trees per class')
        print('                                   Available values: [1, INT_MAX] (default = 30)')
        print('        -N, --xnum=<int>           Number of explanations to compute')
        print('                                   Available values: [1, INT_MAX], all (default = 1)')
        #print('        -o, --output=<string>      Directory where output files will be stored (default: \'temp\')')
        #print('        -r, --reduce=<string>      Extract an MUS from each unsatisfiable core')
        #print('                                   Available values: lin, none, qxp (default = none)')
        #print('        --seed=<int>               Seed for random splitting')
        #print('                                   Available values: [1, INT_MAX] (default = 7)')
        print('        --sep=<string>             Field separator used in input file (default = \',\')')
        #print('        -s, --solver=<string>      A SAT oracle to use')
        #print('                                   Available values: glucose3, minisat (default = g3)')
        print('        -t, --train                Train a model of a given dataset')
        print('        --testsplit=<float>        Training and test sets split')
        print('                                   Available values: [0.0, 1.0] (default = 0.3)')
        #print('        -u, --unit-mcs             Detect and block unit-size MCSes')
        print('        -v, --verbose              Increase verbosity level')
        print('        -x, --xtype=<string>       Type of explanation to compute: abductive or contrastive')
        print('                                   Available values: abd, con (default = None)')
