#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
##

import collections
import math
import itertools
try:
    from .rbo import rbo_dict, rbo
except:
    from rbo import rbo_dict, rbo
from scipy.special import rel_entr


def normalise(lit2immprt, min_v=0):
    if lit2immprt:
        max_v = abs(max(lit2immprt.values(), key=abs))
        return {lit: (imprt - min_v) / (max_v - min_v) for lit, imprt in lit2immprt.items()}
    else:
        return lit2immprt

def axp_stats(axps_):
    """
    unweighted feature attribution
    """
    lit_count = collections.defaultdict(lambda: 0)
    nof_axps = len(axps_)
    for axp in axps_:
        for lit in axp:
            lit_count[lit] += 1
    lit_count = {lit: cnt/nof_axps for lit, cnt in lit_count.items()}
    return lit_count

def axp_stats2(axps_):
    """
    weighted feature attribution
    """
    lit_count = collections.defaultdict(lambda: 0)
    nof_axps = len(axps_)
    for axp in axps_:
        for lit in axp:
            lit_count[lit] += 1/len(axp)
    lit_count = {lit: cnt/nof_axps for lit, cnt in lit_count.items()}
    return lit_count

def measure_dist(cnt0, cnt1, shape=None, metric='manhattan', avg=False):
    assert metric in ('euclidean', 'manhattan')
    pixels = set(cnt0.keys()).union(cnt1.keys())
    for p in pixels:
        assert isinstance(p, int)
    cnt0 = {abs(lit): abs(imprt) for lit, imprt in cnt0.items()}
    cnt1 = {abs(lit): abs(imprt) for lit, imprt in cnt1.items()}
    error = {p: abs(cnt0.get(p, 0) - cnt1.get(p, 0)) for p in pixels}
    # error =
    #print(metric)
    #print()
    #print(cnt0)
    #print()
    #print(cnt1)
    #print()
    #print(error)
    #print()
    #print('pixels:', pixels)
    #print()
    if metric == 'euclidean':
        error = math.sqrt(sum([e ** 2 for e in error.values()]))
    else:
        error = sum(error.values())
    #print('error:', error)
    #print()
    if avg and shape:
        return error / (shape[0] * shape[1])
    else:
        return error

def compare_lists(cnt0, cnt_gt, metric='kendall_tau', inst_id=1, p=0.9):
    #if inst_id == 2:
    #    print('cnt_gt:', cnt_gt)
    #print()
    #if metric == 'rbo':
    #    print('rbo')
    #else:
    #    print('tau')
    #print(cnt0)
    #print()
    for lit in cnt0:
        assert isinstance(lit, int)
    for lit in cnt_gt:
        assert isinstance(lit, int)
    cnt0 = {abs(lit): abs(imprt) for lit, imprt in cnt0.items()}
    cnt_gt = {abs(lit): abs(imprt) for lit, imprt in cnt_gt.items()}
    cnt0_sort = sort_cnt(cnt0, reverse=True)
    #print(cnt0_sort)
    #print()
    #print()
    cnt_gt_sort = sort_cnt(cnt_gt, reverse=True)

    #print(cnt_gt)
    #print()
    #print(cnt_gt_sort)
    #print('cnt_gt:', cnt_gt)
    #print()
    #print('cnt_gt_sort:', cnt_gt_sort)
    #print()

    if metric in ('kendall_tau', 'kendalltau'):
        #Scipy library
        # C: Concordant pairs
        # D: Discordant pairs
        #M = (C - D)
        #tau = M/(C+D) [-1, 1]
        coef = kendalltau(cnt0_sort['pix2rank'].copy(),
                          cnt_gt_sort['pix2rank'].copy())
    elif metric == 'rbo':
        # Rank Biased Overlap
        coef = rank_biased_overlap(cnt0_sort['pix2rank'].copy(),
                                   cnt_gt_sort['pix2rank'].copy(),
                                   p=p)
    else:
        assert False, 'incorrect metric: {}'.format(metric)
    #if inst_id == 2:
    #    print('coef:', coef)
    #print()
    #print(coef)

    #kendall tau
    # Rank Biased Overlap (RBO)
    #print('coef:', coef)

    return coef

def sort_cnt(cnt, reverse=True):
    imprt2pix = collections.defaultdict(lambda : [])
    for pix, imprt in cnt.items():
        imprt2pix[imprt].append(pix)

    imprts = sorted(imprt2pix.keys(), reverse=reverse)
    pix2rank = {}
    for i, imprt in enumerate(imprts):
        for pix in imprt2pix[imprt]:
            pix2rank[pix] = i

    return {'imprt2pix': imprt2pix,
            'pix2rank': pix2rank}


    #return [(p, cnt[p]) for p in sorted(cnt.keys(),
    #                                    key=lambda l: cnt[l],
    #                                    reverse=reverse)]

def update_sort(pix2rank0, pix2rank1):
    pix2rank0 = pix2rank0.copy()
    pix2rank1 = pix2rank1.copy()
    pr0_only = set(pix2rank0.keys()).difference(pix2rank1.keys())
    lst_rank0 = max(pix2rank0.values()) if pix2rank0 else -1

    pr1_only = set(pix2rank1.keys()).difference(pix2rank0.keys())
    lst_rank1 = max(pix2rank1.values()) if pix2rank1 else -1

    for e in pr0_only:
        pix2rank1[e] = lst_rank1 + 1

    for e in pr1_only:
        pix2rank0[e] = lst_rank0 + 1

    return pix2rank0, pix2rank1

def rank_biased_overlap(pix2rank0, pix2rank1, p):
    pix2rank0, pix2rank1 = update_sort(pix2rank0, pix2rank1)
    if not pix2rank0 and not pix2rank1:
        coef = 0.0
    else:
        res = rbo_dict(pix2rank0, pix2rank1, p=p)
        coef = res.ext

    return coef

def kendalltau(pix2rank0, pix2rank1):
    pix2rank0, pix2rank1 = update_sort(pix2rank0, pix2rank1)
    all_elements = set(pix2rank0).union(pix2rank1)
    # intersection = set(pix2rank0).intersection(pix2rank1)

    assert len(pix2rank0) == len(pix2rank1) == len(all_elements)

    all_combs = itertools.combinations(all_elements, 2)

    concordants = []
    discordants = []

    for p1, p2 in all_combs:
        k11 = pix2rank0[p1]
        k12 = pix2rank0[p2]

        k21 = pix2rank1[p1]
        k22 = pix2rank1[p2]

        d1 = (k11 - k12)  # > 0
        d2 = (k21 - k22)  # > 0

        if (d1 > 0 and d2 > 0) or (d1 == 0 and d2 == 0) or (d1 < 0 and d2 < 0):
            concordants.append((p1, p2))
        else:
            discordants.append((p1, p2))

    c = len(concordants)
    d = len(discordants)
    if c == 0 and d == 0:
        tau = -1
    else:
        tau = (c - d) / (c + d)
    return tau

def kl(p, q):
    """
    p - the ground truth distribution
    q - another distribution
    """
    p = {abs(pix): dist for pix, dist in p.items()}
    q = {abs(pix): dist for pix, dist in q.items()}
    all_pixes = set(p.keys()).union(q.keys())
    divergence = 0
    for pix in all_pixes:
        divergence += rel_entr(p.get(pix, 0), q.get(pix, 0))
    #print(divergence == float('inf'))
    return divergence
