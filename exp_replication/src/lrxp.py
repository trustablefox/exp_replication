#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pandas as pd
from rndmforest  import Dataset
import sys, os,  pickle
from options import Options
from pysat.solvers import Solver
import resource
import statistics
from train_global_model import train_global_model, eval_global_model
import pandas as pd
import numpy as np

class LRExplainer(object):
    def __init__(self, data, options):

        with open(options.classifier, 'rb') as f:
            self.model = pickle.load(f)
        self.options = options
        self.fnames = data.feature_names
        self.label = data.names[-1]
        self.data = data
        self.extract_bounds()

    def extract_bound(self, i):
        values = list(map(lambda l: l[i], self.data.X))
        return max(values), min(values)

    def extract_bounds(self):
        self.lbounds = []
        self.ubounds = []
        coefs = self.model.coef_[0]
        for i in range(len(self.data.extended_feature_names_as_array_strings)):
            coef = coefs[i]
            max_value, min_value = self.extract_bound(i)

            if coef >= 0:
                self.lbounds.append(min_value)
                self.ubounds.append(max_value)
            else:
                self.lbounds.append(max_value)
                self.ubounds.append(min_value)

            #print('coef: {}; min: {}; max: {}'.format(coef, self.lbounds[-1], self.ubounds[-1]))
            #print('from {} to {}\n'.format(self.lbounds[-1] * coef, self.ubounds[-1] * coef))

        self.lbounds = pd.Series(self.lbounds, index=self.fnames)
        self.ubounds = pd.Series(self.ubounds, index=self.fnames)

    def free_attr(self, i, inst, lbounds, ubounds, deset, inset):
        lbounds[i] = self.lbounds[i]
        ubounds[i] = self.ubounds[i]
        deset.remove(i)
        inset.add(i)

    def fix_attr(self, i, inst, lbounds, ubounds, deset, inset):

        lbounds[i] = inst[i]
        ubounds[i] = inst[i]
        deset.remove(i)
        inset.add(i)

    def equal_pred(self, lbounds, ubounds):
        return self.model.predict([lbounds])[0] == self.model.predict([ubounds])[0]

    def explain(self, inst):

        self.hypos = list(range(len(inst)))
        pred = self.model.predict([inst])[0]

        self.time = {'abd': 0, 'con': 0}
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

        self.exps = {'abd': [], 'con': []}

        if self.options.xnum == 1: # need to check options.py
            if self.options.xtype in ['abd', 'abductive']:
                xtype = 'abd'
                self.exps['abd'].append(self.extract_AXp(inst))
            else:
                xtype = 'con'
                self.exps['con'].append(self.extract_CXp(inst))

            # record the time for computing an AXp or CXp
            self.time[xtype] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
        else:
            self.exps = self.enumrate(inst)

        if self.options.validate:
            for xtype in self.exps:
                for exp in self.exps[xtype]:
                    assert self.validate(inst, exp, xtype)

        if self.options.verb > 0:
            preamble = ['{0} = {1}'.format(self.fnames[i], inst[i]) for i in self.hypos]
            print('\n  Explaining: IF {0} THEN {1} = {2}'.format(' AND '.join(preamble), self.label, pred))

            xtype = 'abd' if self.options.xtype in ['abd', 'abductive'] else 'con'
            for exp in self.exps[xtype]:
                preamble = ['{0} {1} {2}'.format(self.fnames[i], '=' if xtype == 'abd' else '!=', inst[i])
                            for i in sorted(exp)]

                print('  {}: IF {} THEN {} {} {}'.format(xtype,
                                                         ' AND '.join(preamble),
                                                         self.label,
                                                         '=' if xtype == 'abd' else '!=',
                                                         pred))
                print('  # size: {0}'.format(len(exp)))

            xtype_ = 'abd' if xtype == 'con' else 'con'
            for exp in self.exps[xtype_]:
                preamble = ['{0} {1} {2}'.format(self.fnames[i], '=' if xtype_ == 'abd' else '!=', inst[i])
                            for i in sorted(exp)]

                print('  {}: IF {} THEN {} {} {}'.format(xtype_,
                                                     ' AND '.join(preamble),
                                                         self.label,
                                                     '=' if xtype_ == 'abd' else '!=',
                                                     pred))
                print('  # size: {0}'.format(len(exp)))

            xtypes = ['abd', 'con'] if self.options.xnum != 1 else [xtype]

            for xtype in xtypes:
                print('  {0} time: {1}'.format(xtype, self.time[xtype]))

        return self.exps, self.time

    def extract_AXp(self, inst, seed=set()):
        lbounds = inst.copy()
        ubounds = inst.copy()
        candidate, drop, pick = set(self.hypos), set(), set()

        for i in seed:
            self.free_attr(i, inst, lbounds, ubounds, candidate, drop)


        potential = list(filter(lambda l: l not in seed, self.hypos))

        for i in potential:
            self.free_attr(i, inst, lbounds, ubounds, candidate, drop)
            if not self.equal_pred(lbounds, ubounds):
                self.fix_attr(i, inst, lbounds, ubounds, drop, pick)
        return pick

    def extract_CXp(self, inst, seed=set()):
        lbounds = self.lbounds.copy()
        ubounds = self.ubounds.copy()
        candidate, drop, pick = set(self.hypos), set(), set()

        for i in seed:
            self.fix_attr(i, inst, lbounds, ubounds, candidate, drop)

        potential = list(filter(lambda l: l not in seed, self.hypos))
        for i in potential:
            self.fix_attr(i, inst, lbounds, ubounds, candidate, drop)
            if self.equal_pred(lbounds, ubounds):
                self.free_attr(i, inst, lbounds, ubounds, drop, pick)
        return pick

    def enumrate(self, inst):
        oracle = Solver(name=self.options.solver)
        exps = {'abd': [], 'con': []}

        self.hit = set()

        while True:

            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                   resource.getrusage(resource.RUSAGE_SELF).ru_utime

            if not oracle.solve():
                return exps

            assignment = oracle.get_model()

            lbounds = self.lbounds.copy()
            ubounds = self.ubounds.copy()

            for i in self.hit:
                if assignment[i] > 0:
                    lbounds[i] = inst[i]
                    ubounds[i] = inst[i]

            if self.equal_pred(lbounds, ubounds):
                seed = set(self.hypos).difference(set(filter(lambda i: assignment[i] > 0, self.hit)))
                exp = self.extract_AXp(inst, seed)
                exps['abd'].append(exp)
                oracle.add_clause([-(i + 1) for i in sorted(exp)])
                xtype = 'abd'

            else:
                seed = set(filter(lambda i: assignment[i] > 0, self.hit))
                exp = self.extract_CXp(inst, seed)
                exps['con'].append(exp)
                oracle.add_clause([i + 1 for i in sorted(exp)])
                xtype = 'con'

            self.hit.update(exp)
            # count runtime for an axp or cxp
            self.time[xtype] += resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

    def validate(self, inst, exp, xtype):
        lbounds = inst.copy()
        ubounds = inst.copy()

        if xtype == 'abd':
            for i in self.hypos:
                if i not in exp:
                    lbounds[i] = self.lbounds[i]
                    ubounds[i] = self.ubounds[i]
            return self.equal_pred(lbounds, ubounds) == True
        else:
            for i in self.hypos:
                if i in exp:
                    lbounds[i] = self.lbounds[i]
                    ubounds[i] = self.ubounds[i]
            return self.equal_pred(lbounds, ubounds) == False
