import glob
import os
import sys
import collections
import json
try:
    from .stats import axp_stats
except:
    from stats import axp_stats

def cmptmax(files, key):
    maxkey = 0
    for file in files:
        with open(file, 'r') as f:
            keydict = json.load(f)

        for dt in keydict['stats']:
            k = keydict['stats'][dt][key]
            if k > maxkey:
                maxkey = k
    return maxkey


def parse_log(log, expr, appr, model, info_dict):
    with open(log, 'r') as f:
        lines = f.readlines()

    dataset = log[log.rfind('/') + 1:].split('_')[1]
    config = f'{expr}_{appr}_{model}'

    insts_ids = []
    for i, line in enumerate(lines):
        if 'explaining: ' in line.lower():
            insts_ids.append(i)

    insts_ids.append(len(lines))

    if appr != 'formal':

        if appr.lower() == 'pyexplainer':
            # need to add tot nof rule
            parse_pyexplainer(dataset, config, insts_ids, lines, info_dict)
        elif appr.lower() == 'lime':
            parse_lime(dataset, config, insts_ids, lines, info_dict)
        elif appr.lower() == 'shap':
            parse_shap(dataset, config, insts_ids, lines, info_dict)
        elif appr.lower() == 'anchor':
            parse_anchor(dataset, config, insts_ids, lines, info_dict)
        else:
            #print('wrong log')
            #print(log)
            exit(1)
    else:
        parse_formal(dataset, config, insts_ids, lines, info_dict)


def parse_pyexplainer(dataset, config, insts_ids, lines, info_dict):

    for i, id in enumerate(insts_ids[:-1]):

        inst_name = f'{dataset}_inst{i}'
        pred = True if lines[id].rsplit(' = ')[-1].strip().lower() in ['1', 'true'] else False
        end_id = insts_ids[i + 1]
        rules = {True: [], False: []}
        for iid in range(id, end_id):
            line = lines[iid]
            if 'top ' in line.lower():
                rule = line[line.find(':') + 1:].strip().replace(' & ', ' AND ')
                lits = len(rule.split(' AND '))
                line2 = lines[iid + 1]
                assert 'importance:' in line2.lower()
                importance = float(line2.lower().split('importance:')[-1])

                dirct = True if 'pos' in line[ : line.find(':')].lower() else False
                rules[dirct].append((rule, lits, importance))

            if 'nof rules:' in line:
                nof_rules = int(line.split('nof rules:')[-1])

            if 'time:' in line and 'tot time:' not in line:
                time = float(line.split(':')[-1])


        rules[True].sort(key=lambda l: l[-1], reverse=True)
        rules[False].sort(key=lambda l: l[-1], reverse=True)

        info_dict[config]['stats'][inst_name] = {'status': True,
                                                 'inst': lines[id][lines[id].find('IF')+2: lines[id].rfind('THEN')].strip(),
                                                 'rtime': time,
                                                 'nofrules': nof_rules,
                                                 'pred': pred}

        for k in range(3):
            name = 'predrule{0}'.format(k+1 if k > 0 else '')
            namelit = 'predlits{0}'.format(k + 1 if k > 0 else '')
            nameimprt = 'predimprt{0}'.format(k + 1 if k > 0 else '')

            if len(rules[pred]) > k:
                info_dict[config]['stats'][inst_name][name] = rules[pred][k][0]
                info_dict[config]['stats'][inst_name][namelit] = rules[pred][k][1]
                info_dict[config]['stats'][inst_name][nameimprt] = rules[pred][k][2]
            else:
                info_dict[config]['stats'][inst_name][name] = None
                info_dict[config]['stats'][inst_name][namelit] = None
                info_dict[config]['stats'][inst_name][nameimprt] = None

            name = 'opprule{0}'.format(k + 1 if k > 0 else '')
            namelit = 'opplits{0}'.format(k + 1 if k > 0 else '')
            nameimprt = 'oppimprt{0}'.format(k + 1 if k > 0 else '')

            if len(rules[not pred]) > k:
                info_dict[config]['stats'][inst_name][name] = rules[not pred][k][0]
                info_dict[config]['stats'][inst_name][namelit] = rules[not pred][k][1]
                info_dict[config]['stats'][inst_name][nameimprt] = rules[not pred][k][2]
            else:
                info_dict[config]['stats'][inst_name][name] = None
                info_dict[config]['stats'][inst_name][namelit] = None
                info_dict[config]['stats'][inst_name][nameimprt] = None


def parse_lime(dataset, config, insts_ids, lines, info_dict):

    for i, id in enumerate(insts_ids[:-1]):

        inst = lines[id].split('Explaining:')[-1].split(' THEN ')[0]
        inst = inst[inst.find('IF ')+3:].split(' AND ')

        inst = {fv.split(' = ')[0]: fv.split(' = ')[-1] for fv in inst}

        inst_name = f'{dataset}_inst{i}'
        pred = True if lines[id].rsplit(' = ')[-1].strip().lower() in ['1', 'true'] else False
        end_id = insts_ids[i + 1]
        #rules = {'predrule': [], 'oppdrule': []}
        for iid in range(id, end_id):
            line = lines[iid]
            if 'expl(pos class):' in line:
                rule = line.split('expl(pos class):')[-1].strip()[1:-1].split("), (")
                rule = list(filter(lambda l: len(l) > 0, rule))
                rule = list(map(lambda l: l.strip('(').strip(')').strip().split(', '), rule))
                poslits = len(rule)
                posrule = list(map(lambda l: l[0].strip().strip("'"), rule))
                posrule = ' AND '.join(list(map(lambda l: '{0} = {1}'.format(l, inst[l]), posrule)))
                posimprt = list(map(lambda l: float(l[1]), rule))
            elif 'expl(neg class):' in line:
                rule = line.split('expl(neg class):')[-1].strip()[1:-1].split("), (")
                rule = list(filter(lambda l: len(l) > 0, rule))
                rule = list(map(lambda l: l.strip('(').strip(')').strip().split(', '), rule))
                neglits = len(rule)
                negrule = list(map(lambda l: l[0].strip().strip("'"), rule))
                negrule = ' AND '.join(list(map(lambda l: '{0} = {1}'.format(l, inst[l]), negrule)))
                negimprt = list(map(lambda l: abs(float(l[1])), rule))

            if 'time:' in line and 'tot time:' not in line:
                time = float(line.split(':')[-1])
        inst = lines[id].split(' IF ', maxsplit=1)[-1].rsplit(' THEN ', maxsplit=1)[0].split(' AND ')
        inst = {f.strip(): fid for fid, f in enumerate(inst)}
        info_dict[config]['stats'][inst_name] = {'status': True,
                                                 'inst': lines[id][lines[id].find('IF')+2: lines[id].rfind('THEN')].strip(),
                                                 'rtime': time,
                                                 'pred': pred,
                                                 'predrule': posrule if pred else negrule,
                                                 'predlits': poslits if pred else neglits,
                                                 'predimprt': posimprt if pred else negimprt,
                                                 'oppdrule': negrule if pred else posrule,
                                                 'opplits': neglits if pred else poslits,
                                                 'oppdimprt': negimprt if pred else posimprt}

        f2imprt = {}
        prule = info_dict[config]['stats'][inst_name]['predrule'].split(' AND ')
        for f, imprt in zip(prule, info_dict[config]['stats'][inst_name]['predimprt']):
            f2imprt[inst[f.strip()]] = imprt

        orule = info_dict[config]['stats'][inst_name]['oppdrule'].split(' AND ')
        for f, imprt in zip(orule, info_dict[config]['stats'][inst_name]['oppdimprt']):
            f2imprt[inst[f.strip()]] = imprt

        info_dict[config]['stats'][inst_name]['f2imprt'] = f2imprt

def parse_shap(dataset, config, insts_ids, lines, info_dict):
    return parse_lime(dataset, config, insts_ids, lines, info_dict)

def parse_anchor(dataset, config, insts_ids,  lines, info_dict):
    for i, id in enumerate(insts_ids[:-1]):
        inst = lines[id].split('Explaining:')[-1].split(' THEN ')[0]
        inst = inst[inst.find('IF ') + 3:].split(' AND ')

        inst_name = f'{dataset}_inst{i}'
        pred = True if lines[id].rsplit(' = ')[-1].strip().lower() in ['1', 'true'] else False
        end_id = insts_ids[i + 1]
        for iid in range(id, end_id):
            line = lines[iid]

            if 'expl:' in line:
                predrule = line.split('expl:')[-1].split(' THEN ')[0].strip().strip('IF ')
                predlits = len(predrule.split(' AND '))

            if 'time:' in line and 'tot time:' not in line:
                time = float(line.split(':')[-1])

        info_dict[config]['stats'][inst_name] = {'status': True,
                                                 'inst': lines[id][lines[id].find('IF')+2: lines[id].rfind('THEN')].strip(),
                                                 'rtime': time,
                                                 'pred': pred,
                                                 'predrule': predrule,
                                                 'predlits': predlits,
                                                 'oppdrule': None,
                                                 'opplits': None}

def parse_formal(dataset, config, insts_ids, lines, info_dict):

    xtypes = ['abd', 'con']

    for i, id in enumerate(insts_ids[:-1]):
        inst_name = f'{dataset}_inst{i}'
        pred = True if lines[id].rsplit(' = ')[-1].strip().lower() in ['1', 'true'] else False
        end_id = insts_ids[i + 1]

        rules = {xtype: [] for xtype in xtypes}

        for iid in range(id, end_id):
            line = lines[iid]
            if 'abd:' in line:
                abd = line.split('abd:')[-1].split(' THEN ')[0].strip().strip('"').strip('IF ')
                abd_lits = len(abd.split(' AND '))
                rules['abd'].append((abd, abd_lits))
            elif 'con:' in line:
                con = line.split('con:')[-1].split(' THEN ')[0].strip().strip('"').strip('IF ')
                con_lits = len(con.split(' AND '))
                rules['con'].append((con, con_lits))
            elif 'abd time:' in line:
                abd_time = float(line.split(':')[-1])
            elif 'con time:' in line:
                con_time = float(line.split(':')[-1])
                break


        for xtype in rules:
            rules[xtype].sort(key=lambda l: l[-1])
        inst = lines[id].split(' IF ', maxsplit=1)[-1].rsplit(' THEN ', maxsplit=1)[0].split(' AND ')
        inst = {f.strip(): fid for fid, f in enumerate(inst)}
        abd_ids = [[inst[f.strip()] for f in abd.split(" AND ")] for abd, abd_list in rules['abd']]

        ffa = axp_stats(abd_ids)

        info_dict[config]['stats'][inst_name] = {'status': True,
                                                 'inst': lines[id][lines[id].find('IF')+2: lines[id].rfind('THEN')].strip(),
                                                 'inst-ids': insts_ids,
                                                 'pred': pred,
                                                 'abdrtime': abd_time,
                                                 'abds': [r[0] for r in rules['abd']],
                                                 'abd': rules['abd'][0][0],
                                                 'abdlits': rules['abd'][0][1],
                                                 'nofabds': len(rules['abd']),
                                                 'conrtime': con_time,
                                                 'con': rules['con'][0][0],
                                                 'conlits': rules['con'][0][1],
                                                 'nofcons': len(rules['con']),
                                                  'ffa': ffa}

if __name__ == '__main__':

    """
    
    parsing logs
    
    """
    apprs = ['heuristic', 'formal']

    logs_path = {'heuristic': './logs', 'formal': './logs'}
    logs = collections.defaultdict(lambda : [])

    path = '../logs'

    all_logs = glob.glob(path + '/*.log')
    for log in all_logs:
        file = log.rsplit('/')[-1]
        expr = int(file.split('_', maxsplit=1)[0])
        appr = file.rsplit('/')[-1].split('_')[3].split('.log')[0]
        model = file.split('_')[2]
        #batch = int(file.rsplit('_')[-1].split('.')[0])
        logs[tuple([expr, appr, model])].append(log)

    """
    #save info
    #
    """

    info_dict = collections.defaultdict(
        lambda: {"preamble": {"program": None, "prog_args": None,
                              "prog_alias": None, "benchmark": None},
                 "stats": {}})

    for expr, appr, model in logs:
        logs[tuple([expr, appr, model])].sort()

        appr_logs = logs[tuple([expr, appr, model])]

        for log in appr_logs:
            parse_log(log, expr, appr, model, info_dict)

    if not os.path.isdir('../expls'):
        os.makedirs('../expls')

    files = []
    for k in info_dict:
        json_fn = f'../expls/{k}.json'
        files.append(json_fn)
        for p in info_dict[k]['preamble']:
            info_dict[k]['preamble'][p] = k.replace('_', '').replace('1', '').replace('all', 'enum')

        with open(json_fn, 'w') as f:
            json.dump(info_dict[k], f, indent=4)