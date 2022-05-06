
from sklearn.ensemble._voting import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import resource

import collections
from six.moves import range
import six
import math

from data import Data
from tree import Forest, predict_tree, build_tree
#from .encode import SATEncoder
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.examples.hitman import Hitman
import pickle

#
#==============================================================================

def pickle_load_file(filename):
    try:
        f =  open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data
    except Exception as e:
        print(e)
        print("Cannot load from file", filename)
        exit()

class Dataset(Data):
    """
        Class for representing dataset (transactions).
    """
    def __init__(self, filename=None, fpointer=None, mapfile=None,
            separator=',', use_categorical = False):
        super().__init__(filename, fpointer, mapfile, separator, use_categorical)

        # split data into X and y
        self.feature_names = self.names[:-1]
        self.nb_features = len(self.feature_names)
        self.use_categorical = use_categorical

        samples = np.asarray(self.samps)
        if not all(c.isnumeric() for c in samples[:, -1]):
            le = LabelEncoder()
            le.fit(samples[:, -1])
            samples[:, -1]= le.transform(samples[:, -1])
            self.class_names = le.classes_

        samples = np.asarray(samples, dtype=np.float32)
        self.X = samples[:, 0: self.nb_features]
        self.y = samples[:, self.nb_features]
        self.num_class = len(set(self.y))
        self.target_name = list(range(self.num_class))

        #print("c nof features: {0}".format(self.nb_features))
        #print("c nof classes: {0}".format(self.num_class))
        #print("c nof samples: {0}".format(len(self.samps)))

        # check if we have info about categorical features
        if (self.use_categorical):
            self.target_name = self.class_names

            self.binarizer = {}
            for i in self.categorical_features:
                self.binarizer.update({i: OneHotEncoder(categories='auto', sparse=False)})#,
                self.binarizer[i].fit(self.X[:,[i]])
        else:
            self.categorical_features = []
            self.categorical_names = []
            self.binarizer = []
        #feat map
        self.mapping_features()



    def train_test_split(self, test_size=0.2, seed=0):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=seed)


    def transform(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.binarizer != [])
            tx = []
            for i in range(self.nb_features):
                #self.binarizer[i].drop = None
                if (i in self.categorical_features):
                    self.binarizer[i].drop = None
                    tx_aux = self.binarizer[i].transform(x[:,[i]])
                    tx_aux = np.vstack(tx_aux)
                    tx.append(tx_aux)
                else:
                    tx.append(x[:,[i]])
            tx = np.hstack(tx)
            return tx
        else:
            return x

    def transform_inverse(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.binarizer != [])
            inverse_x = []
            for _, xi in enumerate(x):
                inverse_xi = np.zeros(self.nb_features)
                for f in range(self.nb_features):
                    if f in self.categorical_features:
                        nb_values = len(self.categorical_names[f])
                        v = xi[:nb_values]
                        v = np.expand_dims(v, axis=0)
                        iv = self.binarizer[f].inverse_transform(v)
                        inverse_xi[f] =iv
                        xi = xi[nb_values:]

                    else:
                        inverse_xi[f] = xi[0]
                        xi = xi[1:]
                inverse_x.append(inverse_xi)
            return inverse_x
        else:
            return x

    def transform_inverse_by_index(self, idx):
        if (idx in self.extended_feature_names):
            return self.extended_feature_names[idx]
        else:
            print("Warning there is no feature {} in the internal mapping".format(idx))
            return None

    def transform_by_value(self, feat_value_pair):
        if (feat_value_pair in self.extended_feature_names.values()):
            keys = (list(self.extended_feature_names.keys())[list( self.extended_feature_names.values()).index(feat_value_pair)])
            return keys
        else:
            print("Warning there is no value {} in the internal mapping".format(feat_value_pair))
            return None

    def mapping_features(self):
        self.extended_feature_names = {}
        self.extended_feature_names_as_array_strings = []
        counter = 0
        if (self.use_categorical):
            for i in range(self.nb_features):
                if (i in self.categorical_features):
                    for j, _ in enumerate(self.binarizer[i].categories_[0]):
                        self.extended_feature_names.update({counter:  (self.feature_names[i], j)})
                        self.extended_feature_names_as_array_strings.append("f{}_{}".format(i,j)) # str(self.feature_names[i]), j))
                        counter = counter + 1
                else:
                    self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                    self.extended_feature_names_as_array_strings.append("f{}".format(i)) #(self.feature_names[i])
                    counter = counter + 1
        else:
            for i in range(self.nb_features):
                self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                self.extended_feature_names_as_array_strings.append("f{}".format(i))#(self.feature_names[i])
                counter = counter + 1

    def readable_sample(self, x):
        readable_x = []
        for i, v in enumerate(x):
            if (i in self.categorical_features):
                readable_x.append(self.categorical_names[i][int(v)])
            else:
                readable_x.append(v)
        return np.asarray(readable_x)


    def test_encoding_transformes(self, X_train):
        # test encoding

        X = X_train[[0],:]

        print("Sample of length", len(X[0])," : ", X)
        enc_X = self.transform(X)
        print("Encoded sample of length", len(enc_X[0])," : ", enc_X)
        inv_X = self.transform_inverse(enc_X)
        print("Back to sample", inv_X)
        print("Readable sample", self.readable_sample(inv_X[0]))
        assert((inv_X == X).all())

        '''
        for i in range(len(self.extended_feature_names)):
            print(i, self.transform_inverse_by_index(i))
        for key, value in self.extended_feature_names.items():
            print(value, self.transform_by_value(value))
        '''
#
#==============================================================================
class VotingRF(VotingClassifier):
    """
        Majority rule classifier
    """

    def fit(self, X, y, sample_weight=None):
        print(X, " ", sample_weight)
        self.estimators_ = []
        for _, est in self.estimators:
            self.estimators_.append(est)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_


    def predict(self, X):
        """Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        #check_is_fitted(self)

        # 'hard' voting
        predictions = self._predict(X)
        predictions =  np.asarray(predictions, np.int64) #NEED TO BE CHECKED
        maj = np.apply_along_axis(
            lambda x: np.argmax(
                np.bincount(x, weights=self._weights_not_none)),
            axis=1, arr=predictions)

        maj = self.le_.inverse_transform(maj)

        return maj


#
#==============================================================================
class RF2001(object):
    """
        The main class to train Random Forest Classifier (RFC).
    """

    def __init__(self, options):
        """
            Constructor.
        """
        self.forest = None
        self.voting = None
        self.opt = options

        param_dist = {'n_estimators':options.n_estimators,
                      'max_depth':options.maxdepth,
                      #'criterion':'entropy',
                      'random_state':324089}

        param_dist = {'n_estimators': options.n_estimators,
                      'random_state': 0,
                      'n_jobs': -1}

        self.forest = RandomForestClassifier(**param_dist)



    def train(self, dataset, outfile=None):
        """
            Train a random forest.
        """
        print("outfile: ", outfile)
        X_train, X_test, y_train, y_test = dataset.train_test_split()

        if self.opt.verb:
            dataset.test_encoding_transformes(X_train)

        X_train = dataset.transform(X_train)
        X_test = dataset.transform(X_test)

        print("Build a random forest.")
        self.forest.fit(X_train,y_train)

        rtrees = [ ('dt', dt) for _, dt in enumerate(self.forest.estimators_)]
        self.voting = VotingRF(estimators=rtrees)
        self.voting.fit(X_train,y_train)

        '''
        print(X_test[[0],:])
        print("RF: ",np.asarray(self.voting.predict(X_test[[0],:])))
        for i,t in  enumerate(self.forest.estimators_):
            print("DT_{0}: {1}".format(i,np.asarray(t.predict(X_test[[0],:]))))
        '''

        self.update_trees(dataset.extended_feature_names_as_array_strings)

        train_acc = accuracy_score(self.predict(X_train), y_train)
        test_acc = accuracy_score(self.predict(X_test), y_test)


        if self.opt.verb > 1:
            self.print_acc_vote(X_train, X_test, y_train, y_test)
            self.print_acc_prob(X_train, X_test, y_train, y_test)

        return train_acc, test_acc

    def update_trees(self, feature_names):
        self.trees = [build_tree(dt.tree_, feature_names) for dt in self.forest.estimators_]

    def predict(self, X):

        majs = []
        for _, inst in enumerate(X):
            scores = [predict_tree(dt, inst) for dt in self.trees]
            scores = np.asarray(scores)
            maj = np.argmax(np.bincount(scores))
            majs.append(maj)
        majs = np.asarray(majs)

        return majs
        #return self.voting.predict(X)

    def predict_prob(self, X):
        self.forest.predict(X)

    def estimators(self):
        assert(self.forest.estimators_ is not None)
        return self.forest.estimators_

    def n_estimators(self):
        return self.forest.n_estimators

    def print_acc_vote(self, X_train, X_test, y_train, y_test):
        train_acc = accuracy_score(self.predict(X_train), y_train)
        test_acc = accuracy_score(self.predict(X_test), y_test)
        print("----------------------")
        print("RF2001:")
        print("Train accuracy RF2001: {0:.2f}".format(100. * train_acc))
        print("Test accuracy RF2001: {0:.2f}".format(100. * test_acc))
        print("----------------------")

    def print_acc_prob(self, X_train, X_test, y_train, y_test):
        train_acc = accuracy_score(self.forest.predict(X_train), y_train)
        test_acc = accuracy_score(self.forest.predict(X_test), y_test)
        print("RF-scikit:")
        print("Train accuracy RF-scikit: {0:.2f}".format(100. * train_acc))
        print("Test accuracy RF-scikit: {0:.2f}".format(100. *  test_acc))
        print("----------------------")

    def print_accuracy(self, data, X_test, y_test):
        #X_train = dataset.transform(X_train)
        X_test = data.transform(X_test)

        test_acc = accuracy_score(self.predict(X_test), y_test)
        #print("----------------------")
        #print("Train accuracy : {0:.2f}".format(100. * train_acc))
        #print("Test accuracy : {0:.2f}".format(100. * test_acc))
        print("c Cross-Validation: {0:.2f}".format(100. * test_acc))
        #print("----------------------")
#
#==============================================================================
class XRF(object):
    """
        class to encode and explain Random Forest classifiers.
    """

    def __init__(self, dataset, options):

        self.cls = RF2001(options)
        self.cls.forest = pickle_load_file(options.classifier)
        #self.cls.update_trees(feature_names=dataset.extended_feature_names_as_array_strings)

        self.data = dataset
        self.label = dataset.names[-1]
        self.verbose = options.verb
        self.options = options
        assert (options.encode in [None, "sat", "maxsat"])
        self.opt_encoding = options.encode
        self.f = Forest(self.cls, dataset.extended_feature_names_as_array_strings)
        if options.verb > 2:
            self.f.print_trees()
        #print("c RF sz:", self.f.sz)
        #print('c max-depth:', self.f.md)
        #print('c nof DTs:', len(self.f.trees))


    def encode(self, inst, pred=None):
        """
            Encode a tree ensemble trained previously.
        """
        if 'f' not in dir(self):
            self.f = Forest(self.cls, self.data.extended_feature_names_as_array_strings)
            #self.f.print_tree()

        _ = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        #if self.opt_encoding == "maxsat":
        #    self.enc = MaxSATEncoder(self.f, self.data.feature_names, self.data.num_class, \
        #                          self.data.extended_feature_names_as_array_strings)
        #else:
        self.enc = SATEncoder(self.f, self.data.feature_names, self.data.num_class, \
                                  self.data.extended_feature_names_as_array_strings)

        if pred is None:
            inst = self.data.transform(np.array(inst))[0]
        _, _, _, _ = self.enc.encode(inst, pred)

        #time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
        #        resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        #if self.verbose:
        #    print('c nof vars:', formula.nv) # number of variables
        #    print('c nof clauses:', len(formula.clauses)) # number of clauses
        #    print('c encoding time: {0:.3f}'.format(time))

    def explain(self, inst):
        """
            Explain a prediction made for a given sample with a previously
            trained RF.
        """

        if 'enc' not in dir(self):
            self.encode(inst)

        #if self.verbose:
        #    print("instance: {0}".format(np.array(inst)) )

        inpvals = self.data.readable_sample(inst)
        preamble = []
        for f, v in zip(self.data.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)

        inps = self.data.extended_feature_names_as_array_strings # input (feature value) variables
        #print("inps: {0}".format(inps))

        #if self.opt_encoding == "maxsat":
        #        self.x = MaxSATExplainer(self.enc, inps, preamble, self.data.target_name, verb=self.verbose)
        #else:
        self.x = SATExplainer(self.enc, inps, preamble, self.label, self.data.class_names, options=self.options, verb=self.verbose)
        inst = self.data.transform(np.array(inst))[0]
        expls = self.x.explain(np.array(inst))

        time = self.x.time

        return expls, time

    def isAXp(self, inst, hexpl, columns, pred):
        """
            Explain a prediction made for a given sample with a previously
            trained RF.
        """
        print("inst: ", inst)
        if 'enc' not in dir(self):
            self.encode(hexpl, pred)

        #inpvals = self.data.readable_sample(inst)
        #preamble = []
        #for f, v in zip(self.data.feature_names, inpvals):
        #    if f not in str(v):
        #        preamble.append('{0} = {1}'.format(f, v))
        #    else:
        #        preamble.append(v)
        preamble = 'IF {0} THEN defect = {1}'.format(hexpl, pred)

        inps = self.data.extended_feature_names_as_array_strings # input (feature value) variables
        #print("inps: {0}".format(inps))

        #if self.opt_encoding == "maxsat":
        #        self.x = MaxSATExplainer(self.enc, inps, preamble, self.data.target_name, verb=self.verbose)
        #else:
        self.x = SATExplainer(self.enc, inps, preamble, self.label, self.data.class_names, options=self.options, verb=self.verbose)
        #inst = self.data.transform(np.array(inst))[0]
        #expls = self.x.explain(np.array(inst))
        
        # isRedundant, expl 
        isAXp, _, _ = self.x.isAXp(hexpl, columns)

        #hexp =
        
        # time
        _ = self.x.time

        return isAXp


    def test_tree_ensemble(self):
        if 'f' not in dir(self):
            self.f = Forest(self.cls)

        _, X_test, _, y_test = self.data.train_test_split()
        X_test = self.data.transform(X_test)

        y_pred_forest = self.f.predict(X_test)
        acc = accuracy_score(y_pred_forest, y_test)
        print("Test accuracy: {0:.2f}".format(100. * acc))

        y_pred_cls = self.cls.predict(X_test)

        #print(np.asarray(y_pred_cls, np.int64))
        #print(y_pred_forest)

        assert((y_pred_cls == y_pred_forest).all())


#
#==============================================================================
class SATEncoder(object):
    """
        Encoder of Random Forest classifier into SAT.
    """

    def __init__(self, forest, feats, nof_classes, extended_feature_names, from_file=None):
        self.forest = forest
        #self.feats = {f: i for i, f in enumerate(feats)}
        self.num_class = nof_classes
        self.vpool = IDPool()
        self.extended_feature_names = extended_feature_names

        #encoding formula
        self.cnf = None

        # for interval-based encoding
        self.intvs, self.imaps, self.ivars, self.thvars = None, None, None, None


    def newVar(self, name):

        if name in self.vpool.obj2id: #var has been already created
            return self.vpool.obj2id[name]

        var = self.vpool.id('{0}'.format(name))
        return var

    def printLits(self, lits):
        print(["{0}{1}".format("-" if p<0 else "",self.vpool.obj(abs(p))) for p in lits])

    def traverse(self, tree, k, clause):
        """
            Traverse a tree and encode each node.
        """

        if tree.children:
            f = tree.name
            v = tree.threshold
            pos = neg = []
            if f in self.intvs:
                d = self.imaps[f][v]
                pos, neg = self.thvars[f][d], -self.thvars[f][d]
            else:
                var = self.newVar(tree.name)
                pos, neg = var, -var
                #print("{0} => {1}".format(tree.name, var))

            assert (pos and neg)
            self.traverse(tree.children[0], k, clause + [-neg])
            self.traverse(tree.children[1], k, clause + [-pos])
        else:  # leaf node
            cvar = self.newVar('class{0}_tr{1}'.format(tree.values,k))
            self.cnf.append(clause + [cvar])
            #self.printLits(clause + [cvar])

    def compute_intervals(self):
        """
            Traverse all trees in the ensemble and extract intervals for each
            feature.

            At this point, the method only works for numerical datasets!
        """

        def traverse_intervals(tree):
            """
                Auxiliary function. Recursive tree traversal.
            """

            if tree.children:
                f = tree.name
                v = tree.threshold
                if f in self.intvs:
                    self.intvs[f].add(v)

                traverse_intervals(tree.children[0])
                traverse_intervals(tree.children[1])

        # initializing the intervals
        self.intvs = {'{0}'.format(f): set([]) for f in self.extended_feature_names if '_' not in f}

        for tree in self.forest.trees:
            traverse_intervals(tree)

        # OK, we got all intervals; let's sort the values
        self.intvs = {f: sorted(self.intvs[f]) + ([math.inf] if len(self.intvs[f]) else []) for f in six.iterkeys(self.intvs)}

        def int_feat(intvs):
            for intv in intvs[:-1]:
                if intv % 0.5 != 0:
                    return False
            else:
                return True

        self.imaps, self.ivars = {}, {}
        self.invalid_ivars = {}
        self.thvars = {}
        for feat, intvs in six.iteritems(self.intvs):
            self.imaps[feat] = {}
            self.ivars[feat] = []
            self.thvars[feat] = []

            is_int_feat = int_feat(intvs)
            if is_int_feat:
                self.invalid_ivars[feat] = set()

            for i, ub in enumerate(intvs):
                self.imaps[feat][ub] = i

                ivar = self.newVar('{0}_intv{1}'.format(feat, i))
                self.ivars[feat].append(ivar)
                #print('{0}_intv{1}'.format(feat, i))

                if is_int_feat and ub % 1 == 0.5 and i > 0:
                    # non-existing ivars due to integer data types
                    if ub - intvs[i - 1] == 0.5:
                        self.cnf.append([-ivar])
                        self.invalid_ivars[feat].add(ub)

                if ub != math.inf:
                    #assert(i < len(intvs)-1)
                    thvar = self.newVar('{0}_th{1}'.format(feat, i))
                    self.thvars[feat].append(thvar)
                    #print('{0}_th{1}'.format(feat, i))


    def maj_vote_const(self, ctvars):
        """
            capture majority class vote with cardinality constraints (Pseudo Boolean..)
        """
        # define Tautology var
        vtaut = self.newVar('Tautology')
        num_tree = len(self.forest.trees)
        if(self.num_class == 2):
            rhs = math.floor(num_tree / 2) + 1
            if(self.cmaj==1 and not num_tree%2):
                rhs = math.floor(num_tree / 2)

            lhs = [ctvars[k][1 - self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits = lhs, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
        else:
            zvars = []
            zvars.append([self.newVar('z_0_{0}'.format(k)) for k in range (num_tree) ])
            zvars.append([self.newVar('z_1_{0}'.format(k)) for k in range (num_tree) ])
            ##
            rhs = num_tree
            lhs0 = zvars[0] + [ - ctvars[k][self.cmaj] for k in range(num_tree)]
            ##self.printLits(lhs0)
            atls = CardEnc.atleast(lits = lhs0, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
            ##
            #rhs = num_tree - 1
            rhs = num_tree + 1
            ###########
            lhs1 =  zvars[1] + [ - ctvars[k][self.cmaj] for k in range(num_tree)]
            ##self.printLits(lhs1)
            atls = CardEnc.atleast(lits = lhs1, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
            #
            pvars = [self.newVar('p_{0}'.format(k)) for k in range(self.num_class + 1)]
            ##self.printLits(pvars)
            for k,p in enumerate(pvars):
                for i in range(num_tree):
                    if k == 0:
                        z = zvars[0][i]
                        #self.cnf.append([-p, -z, vtaut])
                        self.cnf.append([-p, z, -vtaut])
                        #self.printLits([-p, z, -vtaut])
                        #print()
                    elif k == self.cmaj+1:
                        z = zvars[1][i]
                        self.cnf.append([-p, z, -vtaut])

                        #self.printLits([-p, z, -vtaut])
                        #print()

                    else:
                        z = zvars[0][i] if (k<self.cmaj+1) else zvars[1][i]
                        self.cnf.append([-p, -z, ctvars[i][k-1] ])
                        self.cnf.append([-p, z, -ctvars[i][k-1] ])

                        #self.printLits([-p, -z, ctvars[i][k-1] ])
                        #self.printLits([-p, z, -ctvars[i][k-1] ])
                        #print()

            #
            self.cnf.append([-pvars[0], -pvars[self.cmaj+1]])
            ##
            lhs1 =  pvars[:(self.cmaj+1)]
            ##self.printLits(lhs1)
            eqls = CardEnc.equals(lits = lhs1, bound = 1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(eqls)


            lhs2 = pvars[(self.cmaj + 1):]
            ##self.printLits(lhs2)
            eqls = CardEnc.equals(lits = lhs2, bound = 1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(eqls)




    def encode(self, sample, pred=None):
        """
            Do the job.
        """

        ###print('Encode RF into SAT ...')

        self.cnf = CNF()
        # getting a tree ensemble
        #self.forest = Forest(self.model, self.extended_feature_names)
        num_tree = len(self.forest.trees)
        #self.forest.predict_inst(sample)
        #introducing class variables
        #cvars = [self.newVar('class{0}'.format(i)) for i in range(self.num_class)]


        # introducing class-tree variables
        ctvars = [[] for t in range(num_tree)]
        for k in range(num_tree):
            for j in range(self.num_class):

                var = self.newVar('class{0}_tr{1}'.format(j,k))
                ctvars[k].append(var)

        # traverse all trees and extract all possible intervals
        # for each feature
        ###print("compute intervarls ...")
        self.compute_intervals()

        #print(self.intvs)
        #print([len(self.intvs[f]) for f in self.intvs])
        #print(self.imaps)
        #print(self.ivars)
        #print(self.thvars)
        #print(ctvars)

        ##print("encode trees ...")
        # traversing and encoding each tree
        for k, tree in enumerate(self.forest.trees):
            #print("Encode tree#{0}".format(k))
            # encoding the tree
            self.traverse(tree, k, [])
            # exactly one class var is true
            #self.printLits(ctvars[k])
            card = CardEnc.atmost(lits=ctvars[k], vpool=self.vpool,encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)

        if pred is None:
            # calculate the majority class
            self.cmaj = self.forest.predict_inst(sample)
        else:
            self.cmaj = 1 if pred == True else 0

        ##print("encode majority class ...")
        #Cardinality constraint AtMostK to capture a j_th class
        self.maj_vote_const(ctvars)


        ##print("exactly-one feat const ...")
        # enforce exactly one of the feature values to be chosen
        # (for categorical features)
        categories = collections.defaultdict(lambda: [])
        for f in self.extended_feature_names:
            if '_' in f:
                categories[f.split('_')[0]].append(self.newVar(f))
        for c, feats in six.iteritems(categories):
            # exactly-one feat is True
            self.cnf.append(feats)
            card = CardEnc.atmost(lits=feats, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)
        # lits of intervals
        for f, intvs in six.iteritems(self.ivars):
            if not len(intvs):
                continue
            self.cnf.append(intvs)
            card = CardEnc.atmost(lits=intvs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)
            #self.printLits(intvs)

        for f, threshold in six.iteritems(self.thvars):
            for j, thvar in enumerate(threshold):
                d = j+1
                pos, neg = self.ivars[f][d:], self.ivars[f][:d]

                if j == 0:
                    assert(len(neg) == 1)
                    self.cnf.append([thvar, neg[-1]])
                    self.cnf.append([-thvar, -neg[-1]])

                else:
                    self.cnf.append([thvar, neg[-1], -threshold[j-1]])
                    self.cnf.append([-thvar, threshold[j-1]])
                    self.cnf.append([-thvar, -neg[-1]])

                if j == len(threshold) - 1:
                    assert(len(pos) == 1)
                    self.cnf.append([-thvar, pos[0]])
                    self.cnf.append([thvar, -pos[0]])
                else:
                    self.cnf.append([-thvar, pos[0], threshold[j+1]])
                    self.cnf.append([thvar, -pos[0]])
                    self.cnf.append([thvar, -threshold[j+1]])

                #if j > 0:
                #    self.cnf.append([-thvar, threshold[j - 1]])
                #if j < len(threshold) - 1:
                #    self.cnf.append([thvar, -threshold[j + 1]])


        return self.cnf, self.intvs, self.imaps, self.ivars


#
#==============================================================================
class MaxSATEncoder(SATEncoder):

    def __init__(self, forest, feats, nof_classes, extended_feature_names,  from_file=None):
        super(MaxSATEncoder, self).__init__(forest, feats, nof_classes, extended_feature_names, from_file)


    def maj_vote_const(self, ctvars):
        """
            Use MaxSAT to capture majority class vote
        """
        num_tree = len(self.forest.trees)
        self.soft = dict()
        for j in range(self.num_class):
            self.soft[j] = [ctvars[i][j] for i in range(num_tree)]
            assert any([(f"class{j}" in self.vpool.obj(abs(v))) for v in self.soft[j]])
            # if j == self.cmaj:
            #     self.soft[j] = [-v for v in self.soft[j]]
            #self.printLits(self.soft[j])

#
#==============================================================================
class SATExplainer(object):
    """
        An SAT-inspired minimal explanation extractor for Random Forest models.
    """

    def __init__(self, sat_enc, inps, preamble, label, target_name, options, verb=1):
        """
            Constructor.
        """

        self.enc = sat_enc
        self.label = label

        self.inps = inps  # input (feature value) variables
        self.target_name = target_name
        self.preamble = preamble
        self.options = options
        self.verbose = verb

        self.slv = None
        ##self.slv = Solver(name=options.solver)
        ##self.slv = Solver(name="minisat22")
        #self.slv = Solver(name="glucose3")
        # CNF formula
        #self.slv.append_formula(self.enc.cnf)

        # number of oracle calls
        self.calls = 0

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """
        self.assums = []  # var selectors to be used as assumptions
        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids
        self.sel2v = {} # selectors to (categorical/interval) values

        #for i in range(self.enc.num_class):
        #    self.csel.append(self.enc.newVar('class{0}'.format(i)))
        #self.csel = self.enc.newVar('class{0}'.format(self.enc.cmaj))

        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, sample), 1):
            if '_' in inp:

                assert (inp not in self.enc.intvs)

                feat = inp.split('_')[0]
                selv = self.enc.newVar('selv_{0}'.format(feat))

                self.assums.append(selv)
                if selv not in self.sel2fid:
                    self.sel2fid[selv] = int(feat[1:])
                    self.sel2vid[selv] = [i - 1]
                else:
                    self.sel2vid[selv].append(i - 1)

                p = self.enc.newVar(inp)
                if not val:
                    p = -p
                else:
                    self.sel2v[selv] = p

                self.enc.cnf.append([-selv, p])

                #self.enc.printLits([-selv, p])

            elif len(self.enc.intvs[inp]):
                #v = None
                #for intv in self.enc.intvs[inp]:
                #    if intv > val:
                #        v = intv
                #        break
                v = next((intv for intv in self.enc.intvs[inp] if intv > val), None)

                assert(v is not None)

                selv = self.enc.newVar('selv_{0}'.format(inp))

                self.assums.append(selv)

                assert (selv not in self.sel2fid)
                self.sel2fid[selv] = int(inp[1:])
                self.sel2vid[selv] = [i - 1]

                for j,p in enumerate(self.enc.ivars[inp]):
                    cl = [-selv]
                    if j == self.enc.imaps[inp][v]:
                        cl += [p]
                        self.sel2v[selv] = p
                    else:
                        cl += [-p]
                    self.enc.cnf.append(cl)
                    #self.enc.printLits(cl)


    def explain(self, sample):
        """
            Hypotheses minimization.
        """
        if self.verbose:
            print('\n  explaining:  "IF {0} THEN {1} = {2}"'.format(' AND '.join(self.preamble),
                                                                  self.label,
                                                                  self.target_name[self.enc.cmaj]))

        #create a SAT solver
        self.slv = Solver(name="glucose3")

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        self.assums = sorted(set(self.assums))

        # pass a CNF formula
        self.slv.append_formula(self.enc.cnf)

        #self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
        #            resource.getrusage(resource.RUSAGE_SELF).ru_utime

        self.time = {'abd': 0, 'con': 0}
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                        resource.getrusage(resource.RUSAGE_SELF).ru_utime

        if self.options.xtype in ('abductive', 'abd'):
            # abductive explanations => MUS computation and enumeration
            if not self.options.smallest and self.options.xnum == 1:
                self.expls = [self.extract_mus()]

                # runtime of computing an AXp
                self.time['abd'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
            else:
                self.mhs_mus_enumeration()
            xtype = 'abd'
        else:  # contrastive explanations => MCS enumeration
            self.mhs_mcs_enumeration()
            xtype = 'con'

        #self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
        #        resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        if self.options.validate:
            for expl in self.expls:
                self.validate(expl, xtype)

            if self.options.xnum != 1:
                for expl in self.duals:
                    self.validate(expl, 'con' if self.options.xtype in ('abductive', 'abd') else 'abd')

        self.expls = list(map(lambda l: sorted([self.sel2fid[h] for h in l ]), self.expls))
        #expl = sorted([self.sel2fid[h] for h in self.assums if h>0 ])

        # delete sat solver
        self.slv.delete()
        self.slv = None

        if self.verbose:
            for expl in self.expls:
                #assert len(expl), 'PI-explanation cannot be an empty-set! otherwise the RF predicts only one class'
                #print("expl-selctors: ", expl)
                if self.options.xtype in ('abductive', 'abd'):
                    preamble = [self.preamble[i] for i in expl]
                else:
                    preamble = [self.preamble[i].replace(' = ', ' != ') for i in expl]
                print('  {0}: "IF {1} THEN {2} {3} {4}"'.format(xtype,
                                                                  ' AND '.join(preamble),
                                                                self.label,
                                                                  '=' if self.options.xtype in ('abductive', 'abd') else '!=',
                                                                  self.target_name[self.enc.cmaj]))
                print('  # size:', len(expl))

        self.expls = {xtype: self.expls}

        if self.options.xnum != 1:

            xtype_ = 'con' if self.options.xtype in ('abductive', 'abd') else 'abd'

            expls_ = list(map(lambda l: sorted([self.sel2fid[h] for h in l]), self.duals))
            self.expls[xtype_] = expls_

            if self.verbose:
                for expl in expls_:
                    preamble = [self.preamble[i].replace(' = ', ' != ') for i in expl]
                    print('  {0}: "IF {1} THEN {2} {3} {4}"'.format(xtype_,
                                                                    ' AND '.join(preamble),
                                                                    self.label,
                                                                    '=' if xtype_ == 'abd' else '!=',
                                                                    self.target_name[self.enc.cmaj]))
                    print('  # size:', len(expl))


        if self.verbose:
            xtypes = ['abd', 'con'] if self.options.xnum != 1 else [xtype]

            for xtype in xtypes:
                print('  {0} time: {1:.2f}'.format(xtype, self.time[xtype]))

        return self.expls

    def prepare_hexp(self, hexpl):

        self.assums = []  # var selectors to be used as assumptions
        #self.sel2v = {}  # selectors to (categorical/interval) values
        self.selv2fv = {}

        def int_feat(intvs):
            for intv in intvs[:-1]:
                if intv % 0.5 != 0:
                    return False
            else:
                return True

        def isInt(val):
            return val % 1 == 0

        # preparing the selectors
        for inp in sorted(hexpl.keys(), key=lambda l: int(l[1:])):
            assert '_' not in inp

            hintv = hexpl[inp]

            if len(self.enc.intvs[inp]):
                selv = self.enc.newVar('selv_{0}'.format(inp))
                self.assums.append(selv)
                self.selv2fv[selv] = hintv

                is_int_feat = int_feat(self.enc.intvs[inp])

                if len(hintv) == 3:
                    val = hintv[-1]
                    eq = hintv[1]
                    if eq == '=':
                        v = next((intv for intv in self.enc.intvs[inp] if intv >= val), None)
                        assert (v is not None)

                        for j, p in enumerate(self.enc.ivars[inp]):
                            cl = [-selv]
                            if j == self.enc.imaps[inp][v]:
                                cl += [p]
                                #self.sel2v[selv] = p
                            else:
                                cl += [-p]
                            self.enc.cnf.append(cl)

                    elif eq in ('<', '<='):
                        max_val = val if eq == '<=' else val - 0.000001
                        max_val = math.floor(max_val) if is_int_feat else max_val
                        v = next((intv for intv in self.enc.intvs[inp] if intv >= max_val), None)

                        assert (v is not None)

                        for j, p in enumerate(self.enc.ivars[inp]):
                            #cl = [-selv]
                            if j <= self.enc.imaps[inp][v]:
                                #cl += [p]
                                pass
                                #self.sel2v[selv] = p
                            else:
                                #cl += [-p]
                                cl = [-selv, -p]
                                self.enc.cnf.append(cl)
                            #self.enc.cnf.append(cl)
                    elif eq in ('>=', '>'):

                        min_val = val if eq == '>=' else val + 0.000001
                        min_val = math.ceil(min_val) if is_int_feat else min_val
                        v = next((intv for intv in self.enc.intvs[inp] if intv >= min_val), None)
                        assert (v is not None)

                        for j, p in enumerate(self.enc.ivars[inp]):
                            #cl = [-selv]

                            if j >= self.enc.imaps[inp][v]:
                                #cl += [p]
                                # self.sel2v[selv] = p
                                pass
                            else:
                                #cl += [-p]
                                cl = [-selv, -p]
                                self.enc.cnf.append(cl)
                            #self.enc.cnf.append(cl)
                    else:
                        print('something worng')
                        exit(1)

                elif len(hintv) == 5:

                    val0, eq0, inp_, eq1, val1 = hintv
                    assert inp == inp_


                    min_val = val0 if eq0 == '<=' else val0 + 0.000001
                    min_val = math.ceil(min_val) if is_int_feat else min_val
                    v0 = next((intv for intv in self.enc.intvs[inp] if intv >= min_val), None)

                    max_val = val1 if eq1 == '<=' else val1 - 0.000001
                    max_val = math.floor(max_val) if is_int_feat else max_val
                    v1 = next((intv for intv in self.enc.intvs[inp] if intv >= max_val), None)

                    assert (v0 is not None)
                    assert (v1 is not None)
                    for j, p in enumerate(self.enc.ivars[inp]):
                        #cl = [-selv]
                        if j >= self.enc.imaps[inp][v0] and j <= self.enc.imaps[inp][v1]:
                            #cl += [p]
                            # self.sel2v[selv] = p
                            pass
                        else:
                            #cl += [-p]
                            cl = [-selv, -p]
                            self.enc.cnf.append(cl)

                        #self.enc.cnf.append(cl)

                else:
                    print('something worng')
                    exit(1)


    def isAXp(self, hexpl, columns):
        """
            Hypotheses minimization.
        """
        #if self.verbose:
        #    print('\n  explaining:  "IF {0} THEN {1} = {2}"'.format(' AND '.join(self.preamble),
        #                                                          self.label,
        #                                                          self.target_name[self.enc.cmaj]))

        #create a SAT solver
        self.slv = Solver(name="glucose3")

        # adapt the solver to deal with the current sample
        #self.prepare(sample)
        self.prepare_hexp(hexpl)

        self.assums = sorted(set(self.assums))

        # pass a CNF formula
        self.slv.append_formula(self.enc.cnf)

        #self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
        #            resource.getrusage(resource.RUSAGE_SELF).ru_utime

        self.time = {'abd': 0, 'con': 0}
        _ = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                        resource.getrusage(resource.RUSAGE_SELF).ru_utime



        #vtaut = self.enc.newVar('Tautology')
        #for assum in self.assums:
        #    print(self.selv2fv[assum])
        #    print(self.slv.solve(assumptions=[vtaut, assum]))
        #
        #print(self.slv.solve(assumptions=[vtaut] + self.assums))
        #exit()

        isAXp, isRedundant, expl_ = self.extract_mus()

        expl_ = [self.selv2fv[s] for s in expl_]
        expl = []

        for e in expl_:
            ee = []
            if len(e) == 3:
                ee.append(columns[int(e[0][1:])])
                ee.append(e[1])
                ee.append(str(e[2]))
            else:
                ee.append(str(e[0]))
                ee.append(e[1])
                ee.append(columns[int(e[2][1:])])
                ee.append(e[3])
                ee.append(str(e[4]))
            expl.append(' '.join(ee))

        expl = ' AND '.join(expl)

        return isAXp, isRedundant, expl

    def extract_mus(self, start_from=None):
        """
            Compute any subset-minimal explanation.
        """
        self.nsat, self.nunsat = 0, 0
        self.stimes, self.utimes = [], []

        vtaut = self.enc.newVar('Tautology')

        if self.slv.solve(assumptions=[vtaut] + self.assums) != False:
            #print(f'not satisfies axp property')
            return False, False, []

        def _do_linear(core):
            """
                Do linear search.
            """

            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    t0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                         resource.getrusage(resource.RUSAGE_SELF).ru_utime

                    self.calls += 1
                    sat = self.slv.solve(assumptions=[vtaut] + sorted(to_test))

                    t = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                        resource.getrusage(resource.RUSAGE_SELF).ru_utime - t0

                    if not sat:
                        self.nunsat += 1
                        self.utimes.append(t)

                        return False

                    to_test.add(a)

                    self.nsat += 1
                    self.stimes.append(t)

                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        if start_from is None:
            # this call must be unsatisfiable!
            assert self.slv.solve(assumptions=[vtaut] + self.assums) == False
        else:
            assert self.slv.solve(assumptions=[vtaut] + start_from) == False

        # this is our MUS over-approximation
        core = self.slv.get_core()
        #core.remove(vtaut)
        core = list(filter(lambda l: l != vtaut, core))

        expl = _do_linear(core)

        if len(expl) == len(self.assums):
            return True, False, expl
        else:
            return True, True, expl

    def mhs_mus_enumeration(self):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (contrastive) explanations
        self.duals = []

        vtaut = self.enc.newVar('Tautology')

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                        resource.getrusage(resource.RUSAGE_SELF).ru_utime

        with Hitman(bootstrap_with=[self.assums], htype='sorted' if self.options.smallest else 'lbx') as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.assums):
                self.calls += 1
                if self.slv.solve(assumptions=[vtaut] + self.assums[:i] + self.assums[(i + 1):]):
                    hitman.hit([hypo])
                    self.duals.append([hypo])

                    # count time for cxps
                    self.time['con'] += resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
                else:
                    # count time for cxps
                    self.time['abd'] += resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                        resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

                time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.options.verb > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if self.slv.solve(assumptions=[vtaut] + hset):
                    to_hit = []
                    # satisfied
                    _, unsatisfied = [], []

                    removed = list(set(self.assums).difference(set(hset)))

                    model = self.slv.get_model()
                    for h in removed:
                        if model[abs(h) - 1] != h:
                            unsatisfied.append(h)
                        else:
                            hset.append(h)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        self.calls += 1
                        if self.slv.solve(assumptions=[vtaut] + hset + [h]):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.options.verb > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    self.duals.append(to_hit)

                    self.time['con'] += resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                        resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
                else:
                    if self.options.verb > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    self.time['abd'] += resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                        resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

                    if len(self.expls) != self.options.xnum:
                        hitman.block(hset)
                    else:
                        break

                time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime

    def mhs_mcs_enumeration(self):
        """
           Enumerate subset- and cardinality-minimal contrastive explanations.
       """

        # result
        self.expls = []

        # just in case, let's save dual (abductive) explanations
        self.duals = []

        vtaut = self.enc.newVar('Tautology')

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime

        with Hitman(bootstrap_with=[self.assums], htype='sorted' if self.options.smallest else 'lbx') as hitman:
            # computing unit-size MUSes
            for i, hypo in enumerate(self.assums):
                self.calls += 1

                if not self.slv.solve(assumptions=[vtaut] + [hypo]):
                    hitman.hit([hypo])
                    self.duals.append([hypo])

                    self.time['abd'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

                elif self.options.unit_mcs and self.slv.solve(assumptions=[vtaut] + self.assums[:i] + self.assums[(i + 1):]):
                    # this is a unit-size MCS => block immediately
                    self.calls += 1
                    hitman.block([hypo])
                    self.expls.append([hypo])

                    self.time['con'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
                else:
                    pass
                time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime
            self.calls = 0
            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1
                if self.options.verb > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if not self.slv.solve(
                        assumptions=[vtaut] + sorted(set(self.assums).difference(set(hset)))):
                    to_hit = self.slv.get_core()
                    to_hit = list(filter(lambda l: l != vtaut, to_hit))

                    if len(to_hit) > 1 and self.options.reduce != 'none':
                        to_hit = self.extract_mus(start_from=to_hit)

                    self.duals.append(to_hit)

                    if self.options.verb > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    self.time['abd'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
                else:
                    if self.options.verb > 2:
                        print('expl:', hset)
                    self.expls.append(hset)

                    self.time['con'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
                    if len(self.expls) != self.options.xnum:
                        hitman.block(hset)
                    else:
                        break

                time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime

    def validate(self, expl, xtype):
        vtaut = self.enc.newVar('Tautology')

        if xtype in ('abductive', 'abd'):
            assert not self.slv.solve(assumptions=[vtaut] + expl)
        else:
            assert self.slv.solve(assumptions=[vtaut] + sorted(set(self.assums).difference(set(expl))))

    def print_sat_model(self):
        assert(self.slv.get_model())
        model = [ p for p in self.slv.get_model() if self.enc.vpool.obj(abs(p)) ]
        str_model = []
        lits = []
        for p in model:
            if self.enc.vpool.obj(abs(p)) in self.inps:
                str_model.append((p, self.enc.vpool.obj(abs(p))))

            elif ("class" in self.enc.vpool.obj(abs(p))):
                  str_model.append((p, self.enc.vpool.obj(abs(p))))

            #elif ("intv" in self.enc.vpool.obj(abs(p))) :
            #      str_model.append((p, self.enc.vpool.obj(abs(p))))

            if ("_tr" in self.enc.vpool.obj(abs(p))) :
                  lits.append(p)

            if ("p_" in self.enc.vpool.obj(abs(p))) :
                  str_model.append((p, self.enc.vpool.obj(abs(p))))
            if ("z_" in self.enc.vpool.obj(abs(p))) :
                  str_model.append((p, self.enc.vpool.obj(abs(p))))

        print("Model:", str_model)
        ###print(self.slv.get_model())
        
        # num_tree
        _ = len(self.enc.forest.trees)
        num_class = self.enc.num_class
        occ = [0]*num_class

        for p in lits:
            if p > 0:
                j = int(self.enc.vpool.obj(abs(p))[5])
                occ[j] +=1
        print(occ)

data_path = './dataset/'
