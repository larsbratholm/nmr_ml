#from __future__ import print_function

import fml
from fml import NUCLEAR_CHARGE
import sys
import numpy as np
import cPickle as pickle
import os
import nmr_ml
#import sklearn.feature_selection
import collections
import h5py
from fml.math import l2_distance, manhattan_distance, p_distance, get_l2_distance_arad
import scipy.stats as ss
import matplotlib.pyplot as plt
import sklearn.random_projection
import sklearn.cluster
import sklearn.decomposition
import sklearn.mixture
import pandas
import statsmodels.api as sm
import scipy.optimize

# inherit fml.Molecule class
class Molecule(fml.Molecule):
    def __init__(self):
        fml.Molecule.__init__(self)
        self.electron_charges = []
        self.mulliken_charges = []

    def generate_atomic_localized_coulomb_matrix(self, calc="all",size=23, cutoff=10**6, cutoff2=10**6, alpha=1):
        self.atomic_coulomb_matrix = fml.representations.fgenerate_atomic_localized_coulomb_matrix( \
                self.nuclear_charges, self.coordinates, self.natoms, size, cutoff, cutoff2, alpha)

    def generate_reduced_atomic_localized_coulomb_matrix(self, calc="all",size=23, cutoff=10**6, cutoff2=10**6, alpha=1):
        self.atomic_coulomb_matrix = fml.representations.fgenerate_reduced_atomic_localized_coulomb_matrix( \
                self.nuclear_charges, self.coordinates, self.natoms, size, cutoff, cutoff2, alpha)

    def generate_atomic_coulomb_matrix(self, calc="all",size=23, cutoff=10**6, decay=0, charge_mode = "nuclear"):
        if charge_mode == "electronic":
            charges = np.asarray(self.nuclear_charges) - self.properties[:,4].astype(float)
        elif charge_mode == "mulliken":
            charges = self.properties[:,4]
        else:
            charges = self.nuclear_charges
        self.atomic_coulomb_matrix = fml.representations.fgenerate_atomic_coulomb_matrix( \
                charges, self.coordinates, self.natoms, size, cutoff, decay)

    def read_extended_xyz(self, filename):

        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        self.natoms = int(lines[0])
        nproperties = len(lines[2].split()) - 3

        self.coordinates = np.empty((self.natoms, 3), dtype=float)
        self.properties = np.empty((self.natoms, nproperties), dtype=object)

        for i, line in enumerate(lines[2:]):
            tokens = line.split()

            if len(tokens) < 4:
                break

            self.atomtypes.append(tokens[0])
            self.nuclear_charges.append(NUCLEAR_CHARGE[tokens[0]])

            self.coordinates[i] = np.asarray(tokens[1:4])
            self.properties[i] = np.asarray(tokens[4:] + [filename])

        self.merge_labels()

    # merge labels for symmetric groups
    def merge_labels(self):
        labels = self.properties[:,3]
        for i, l in enumerate(labels):
            if l[1:] in ("5", "6"):
                labels[i] = labels[i][0] + "4"
            if l[1:] in ("14", "15"):
                labels[i] = labels[i][0] + "13"
            elif l[1:] in (str(self.natoms), str(self.natoms - 1)):
                labels[i] = labels[i][0] + str(self.natoms-2)
            elif l[1:] in (str(self.natoms - 7), str(self.natoms - 8)):
                labels[i] = labels[i][0] + str(self.natoms-9)
            elif l in ("A24", "A25"):
                labels[i] = "A23"
            elif l == "D28":
                labels[i] = "D27"
            elif l == "D31":
                labels[i] = "D30"
            elif l == "F32":
                labels[i] = "F29"
            elif l == "F35":
                labels[i] = "F32"
            elif l == "F28":
                labels[i] = "F27"
            elif l == "F31":
                labels[i] = "F30"
            elif l in ("I28", "I29"):
                labels[i] = "I27"
            elif l in ("I34", "I35"):
                labels[i] = "I33"
            elif l in ("K37", "K38"):
                labels[i] = "K36"
            elif l in ("L34", "L35"):
                labels[i] = "L33"
            elif l in ("L31", "L32"):
                labels[i] = "L30"
            elif l in ("M32", "M33"):
                labels[i] = "M31"
            elif l in ("R38", "R39", "R40"):
                labels[i] = "R37"
            elif l in ("T28", "T29"):
                labels[i] = "T27"
            elif l in ("V31", "V32"):
                labels[i] = "V30"
            elif l in ("V28", "V29"):
                labels[i] = "V27"
            elif l  == "N30":
                labels[i] = "N29"
            elif l  == "Q33":
                labels[i] = "Q32"
            elif l  == "R36":
                labels[i] = "R35"
            elif l  == "Y32":
                labels[i] = "Y29"
            elif l  == "Y35":
                labels[i] = "Y33"
            elif l  == "Y28":
                labels[i] = "Y27"
            elif l  == "Y31":
                labels[i] = "Y30"

def get_mols(path = "xyz/", pickle_path = 'mols.pickle'):
    # read coordinates and properties
    if os.path.isfile(pickle_path):
        with open(pickle_path) as f:
           return pickle.load(f)
    else:
        filenames = [path+f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        mols = np.empty(len(filenames), dtype=object)
        for i, filename in enumerate(filenames):
            mol = Molecule()
            mol.read_extended_xyz(filename)
            mols[i] = mol
        with open(pickle_path, "w") as f:
            pickle.dump(mols, f, -1)
        return mols

def generate_CM(mols, max_atoms):
    # cutoffs for normal atomic CM
    for cut in np.arange(1.5,15.5,0.5):
        for mol in mols:
            mol.generate_atomic_coulomb_matrix(size=max_atoms, cutoff=cut)
        with open("descriptors/CM/cutoff_%.1f.pickle" % cut, "w") as f:
            pickle.dump(mols, f, -1)

def generate_decayed_CM(mols, max_atoms):
    # cutoffs for decayed atomic CM
    for decay in (1,2):
        for cut in np.arange(2,12,1):
            for mol in mols:
                mol.generate_atomic_coulomb_matrix(size=max_atoms, cutoff=cut, decay=decay)
            with open("descriptors/decay/decay_%d_cutoff_%.1f.pickle" % (decay, cut), "w") as f:
                pickle.dump(mols, f, -1)

def generate_sncf_CM(mols, max_atoms):
    # cutoffs for localized sncf atomic CM
    for cut in np.arange(2.0,16,1):
        for cut2 in np.arange(cut,min(16,2*cut+1),1):
            for exponent in np.arange(1,7,1):
                for mol in mols:
                    mol.generate_atomic_localized_coulomb_matrix(size=max_atoms, cutoff=cut, cutoff2=cut2, alpha=exponent)
                with open("descriptors/localized_CM/cutoff_%d_cutoff2_%d_exponent_%d.pickle" % (cut, cut2, exponent), "w") as f:
                    pickle.dump(mols, f, -1)

def generate_reduced_CM(mols, max_atoms):
    # cutoffs for reduced localized sncf atomic CM
    for cut in np.arange(2,9,1):
        for cut2 in np.arange(cut+1,min(10, 2*cut+1),1):
            for exponent in np.arange(1,7,1):
                for mol in mols:
                    mol.generate_reduced_atomic_localized_coulomb_matrix(size=max_atoms, cutoff=cut, cutoff2=cut2, alpha=exponent)
                with open("descriptors/reduced_localized_CM/cutoff_%d_cutoff2_%d_exponent_%d.pickle" % (cut, cut2, exponent), "w") as f:
                    pickle.dump(mols, f, -1)

def generate_charge_CM(mols, max_atoms):
    # cutoffs for charge_mode atomic CM
    for charge_mode in ("mulliken", "electronic"):
        for cut in np.arange(2.0,10,0.5):
            for mol in mols:
                mol.generate_atomic_coulomb_matrix(size=max_atoms, cutoff=cut, charge_mode=charge_mode)
            with open("descriptors/CM_charge_mode/%s_cutoff_%.1f.pickle" % (charge_mode, cut), "w") as f:
                pickle.dump(mols, f, -1)

def generate_ARAD(mols, max_atoms):
    for mol in mols:
        mol.generate_arad_descriptor(size=max_atoms)
        # change name for easier evaluation
        mol.atomic_coulomb_matrix = mol.arad_descriptor.copy()
        del mol.arad_descriptor
    with open("descriptors/arad.pickle", "w") as f:
        pickle.dump(mols, f, -1)

def generate_descriptors():
    mols = get_mols()
    max_atoms = max(mol.natoms for mol in mols)
    generate_CM(mols, max_atoms)
    generate_decayed_CM(mols, max_atoms)
    generate_sncf_CM(mols, max_atoms)
    generate_reduced_CM(mols, max_atoms)
    generate_charge_CM(mols, max_atoms)
    generate_ARAD(mols, max_atoms)

def make_h5_dataset(f, name, data):
    data = f.create_dataset(name, data=data, chunks=True, compression='lzf', shuffle=True, fletcher32=True)
    print(data.name)

#def generate_arad_kernels(mols, filename):
#    # create h5py object for each file
#    h5py_filename = "kernels/" + "/".join(filename.split("/")[1:])
#    h5py_filename = ".".join(h5py_filename.split(".")[:-1]) + ".h5"
#    f = h5py.File(h5py_filename,"w")
#    # iterate over atomtypes of interest
#    for element in ("C","H","N"):
#        print(element)
#        f_ele = f.require_group(element)
#        descriptors = []
#        # track where each descriptor comes from:
#        origin = []
#        properties = []
#        for n, mol in enumerate(mols[:100]):
#            w = np.where(np.asarray(mol.atomtypes) == element)[0]
#            desc = np.zeros(mol.atomic_coulomb_matrix.shape)
#            desc[:w.size, :, :w.size] = mol.atomic_coulomb_matrix[np.ix_(w,range(5),w)]
#            descriptors.append(desc)
#            assert(np.allclose(desc.shape, (56,5,56)))
#            prop = np.zeros((mol.atomic_coulomb_matrix.shape[0], mol.properties.shape[1]), dtype=object)
#            prop[:w.size,:] = mol.properties[w,:]
#            properties.append(prop)
#            assert(np.allclose(prop.shape, (56,12)))
#
#        descriptors = np.asarray(descriptors)
#        # h5py can't store objects, and converting to string defaults
#        # to string length of 8. So get the needed maximum length
#        maxlen = max(len(str(x)) for y in properties for z in y for x in z) #max(max(len(a) for a in x) for y in properties for x in y)
#        properties = np.asarray(properties, dtype='S%d' % maxlen)
#
#        # store properties
#        make_h5_dataset(f_ele, 'properties', properties)
#
#        # create ARAD kernels
#        z = properties[:,:,0].astype(float)
#        D = get_l2_distance_arad(descriptors, descriptors, z, z)
#        f_kernel = f_ele.create_group('arad')
#        make_gaussian_kernel(D, f_kernel)
#        del D
#    # close hdf5 file
#    f.close()

def generate_distance_matrices(filenames = []):
    # either traverse subdirs and do all
    if len(filenames) == 0:
        path = "descriptors/"
        filenames = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".pickle"):
                     filenames.append(os.path.join(root, f))
    else:
        # or only do given filenames
        pass

    # iterate over all saved descriptors
    for filename in filenames:
        with open(filename) as f:
           mols = pickle.load(f)

        # arad special case
        if "arad" in filename.lower():
            quit("ARAD not supported")
            continue

        # create h5py object for each file
        h5py_filename = "distances/" + "/".join(filename.split("/")[1:])
        h5py_filename = ".".join(h5py_filename.split(".")[:-1]) + ".h5"
        f = h5py.File(h5py_filename,"w")
        # iterate over atomtypes of interest
        for element in ("C","H","N"):
            print(element)
            f_ele = f.require_group(element)
            descriptors = []
            # track where each descriptor comes from:
            origin = []
            properties = []
            for n, mol in enumerate(mols):
                for i, atom in enumerate(mol.atomtypes):
                    # remove the outermost hydrogens to keep size a bit lower
                    if i in [3,4,5,mol.natoms-1,mol.natoms-2,mol.natoms-3, mol.natoms-5]:
                        # quick check that I don't do anything wrong
                        if mol.atomtypes[i] != "H":
                            quit("error :(")
                        continue
                    if atom == element:
                        descriptors.append(mol.atomic_coulomb_matrix[i])

                        origin.append((n,i))
                        properties.append(mol.properties[i])
            descriptors = np.asarray(descriptors)
            origin = np.asarray(origin)
            # h5py can't store objects, and converting to string defaults
            # to string length of 8. So get the needed maximum length
            maxlen = max(len(x) for y in properties for x in y)
            properties = np.asarray(properties, dtype='S%d' % maxlen)

            # only keep features that are not constant throughout the dataset
            idx = np.where(descriptors.var(0) > 1e-6)[0]
            descriptors = descriptors[:,idx]
            # normalize descriptors such that sigma ranges are approximately the same
            descriptors /= np.max(np.linalg.norm(descriptors, axis=1))

            # store origin and properties
            make_h5_dataset(f_ele, 'origin', origin)
            make_h5_dataset(f_ele, 'properties', properties)

            D = manhattan_distance(descriptors.T, descriptors.T)
            make_h5_dataset(f_ele, 'manhattan', D)
            del D
            D = l2_distance(descriptors.T, descriptors.T)
            make_h5_dataset(f_ele, 'l2', D)
            np.square(D, out=D)
            make_h5_dataset(f_ele, 'l2sq', D)
            del D
            D = p_distance(descriptors.T, descriptors.T, p=1.5)
            make_h5_dataset(f_ele, 'p_three_halfs', D)
            del D

            ## create kernels based on manhattan distance
            #D = manhattan_distance(descriptors.T, descriptors.T)
            #f_kernel = f_ele.create_group('laplace_l1')
            #print(f_kernel)
            #make_laplace_kernel(D, f_kernel)
            #f_kernel = f_ele.create_group('matern3_l1')
            #print(f_kernel)
            #make_matern3_kernel(D, f_kernel)
            #del D
            ## create kernels based on l2 distance
            #D = l2_distance(descriptors.T, descriptors.T)
            #f_kernel = f_ele.create_group('laplace_l2')
            #print(f_kernel)
            #make_laplace_kernel(D, f_kernel)
            #f_kernel = f_ele.create_group('matern3_l2')
            #print(f_kernel)
            #make_matern3_kernel(D, f_kernel)
            ## create kernels based on squared l2 distance
            #np.square(D, out=D)
            #f_kernel = f_ele.create_group('gaussian')
            #print(f_kernel)
            #make_gaussian_kernel(D, f_kernel)
            #del D
            ## create p=3/2 kernel
            #D = p_distance(descriptors.T, descriptors.T, p=1.5)
            #f_kernel = f_ele.create_group('mixed')
            #print(f_kernel)
            #make_three_halfs_kernel(D, f_kernel)
            #del D
        # close hdf5 file
        f.close()

def predict():
    #np.random.seed(1)
    #np.random.shuffle(mols)

    ## Make training and test sets
    #n_test  = 2000
    #n_train = 8000

    #Xall = []
    #Yall = []

    #target_type = "C"

    #for mol in mols:
    #    for i, atomtype in enumerate(mol.atomtypes):
    #        if atomtype == target_type:
    #            Xall.append(mol.atomic_coulomb_matrix[i])
    #            Yall.append(float(mol.properties[i,6])/1.128+20.49 - float(mol.properties[i,5]))
    #            #Xall.append(float(mol.properties[i,5]))
    #            #Yall.append(float(mol.properties[i,6]))

    ##import scipy.stats as ss
    ##print(ss.linregress(Xall,Yall))

    ## Vectors of descriptors for training and test sets - note transposed
    ## for enhanced speed in kernel evaluation
    #X = np.array(Xall[:n_train]).T
    #Xs = np.array(Xall[-n_test:]).T
    #mod = sklearn.feature_selection.VarianceThreshold()
    #X = mod.fit_transform(X.T).T
    #Xs = mod.transform(Xs.T).T
    #assert(Xs.shape[0] == X.shape[0])

    ## Vectors of properties for training and test sets
    #Y = np.array(Yall[:n_train], dtype=float)
    #Ys = np.array(Yall[-n_test:], dtype=float)

    ## Set hyper-parameters
    #sigma = 10**(4.5)
    #llambda = 10**(-9.0)

    ## print "Calculating K-matrix           ...",
    #print(u"Calculating: K\u1D62\u2C7C = k(q\u1D62, q\u2C7C)          ... ", end="")
    #sys.stdout.flush() 
    #start = time.time()

    ## Gaussian kernel usually better for atomic properties
    #K = laplacian_kernel(X, X, sigma)

    ## Alternatively, just calculate the L2 distance, and convert
    ## to kernel matrix (e.g. for multiple sigmas, etc):
    ##D = np.sqrt(l2_distance(X,X))
    ###D = manhattan_distance(X,X)
    ##D /= -1.0 * sigma
    ##K = np.exp(D)

    #print ("%7.2f seconds" % (time.time() - start) )

    #for i in xrange(n_train):
    #    K[i,i] += llambda

    #print( u"Calculating: \u03B1 = (K + \u03BBI)\u207B\u00B9 y         ... ", end="")
    #sys.stdout.flush()
    #start = time.time()
    #alpha = cho_solve(K,Y)
    #print ("%7.2f seconds" % (time.time() - start) )

    #print(u"Calculating: K*\u1D62\u2C7C = k(q\u1D62, q*\u2C7C)        ... ", end="")
    #sys.stdout.flush()
    #start = time.time()
    #Ks = gaussian_kernel(X, Xs, sigma)
    #print ("%7.2f seconds" % (time.time() - start) )


    #print( u"Calculating: y* = (K*)\u1D40\u00B7\u03B1             ... ", end="")
    #sys.stdout.flush()
    #start = time.time()
    #Y_tilde = np.dot(Ks.transpose(), alpha)
    #print ("%7.2f seconds" % (time.time() - start) )
    #
    #rmsd = np.sqrt(np.mean(np.square(Ys - Y_tilde)))
    #print("RMSD = %6.2f ppm" % rmsd)
    pass

def generate_preprocessing():
    def remove_mean(X,Y):
        return (Y-X) - (Y-X).mean()

    def linreg(X, Y):
        a,b = ss.linregress(X,Y)[:2]
        return (Y - (a*X + b))
    
    def weighted_linreg(X,Y,w):
        wsum = np.sum(w)
        xy = np.sum(w*X*Y)
        x = np.sum(w*X)
        y = np.sum(w*Y)
        xx = np.sum(w*X*X)

        a = (wsum*xy-x*y)/(wsum*xx-x*x)
        b = -(x*xy-xx*y)/(wsum*xx-x*x)
        return a,b


    def GM_linreg(df, n):
        def EM(w):
            # transform the data
            D_train = np.zeros(X_train.shape)
            for i in range(n):
                weights = w.reshape(atomtypes.size, n)[atom_indices,i]
                a,b = weighted_linreg(X_train,Y_train,weights)
                D_train += weights * (Y_train - (b + a*X_train))
            E = np.sum(D_train**2)/D_train.size
            return E

        def callback(w):
            print EM(w)

        def norm(w):
            return np.sum(w.reshape(atomtypes.size,n), axis=1) - 1

            ## fit the GM
            #GM = sklearn.mixture.GaussianMixture(n_components = n)
            #GM.fit(D_train_tf[:,None])
            #new_labels = current_labels.copy()
            #for i, a in enumerate(atomtypes):
            #    D_train_a = D_train_tf[atom_indices == i]
            #    prob = GM.predict_proba(D_train_a[:,None])
            #    log_sum = np.sum(np.log(prob),axis=0)/
            #    new_labels[i] = np.exp(log_sum) / np.sum(np.exp(log_sum))
            #return new_labels


        # random assignment of classes
        atomtypes, atom_indices = np.unique(df.label.values[df.train.values == True], return_inverse = True)
        current_labels = np.random.dirichlet([0.5]*n, size=atomtypes.size)
        current_labels = current_labels.ravel()
        # TODO less rubbish method
        opt = scipy.optimize.minimize(EM, current_labels, constraints={'type': 'eq', 'fun': norm}, method = "SLSQP", options={"disp":False, "maxiter": 500}, bounds=[(0,1)]*atomtypes.size*n)
        weights = opt.x.reshape(atomtypes.size,n).argmax(1)
        return weights


    def get_partition_indices(N, n):
        sets = [range(i,N,n) for i in range(n)]
        train_idx = np.concatenate(sets[:n-2]).astype(int)
        val_idx = np.asarray(sets[n-2], dtype=int)
        test_idx = np.asarray(sets[n-1], dtype=int)
        return train_idx, val_idx, test_idx

    def pca():
        pass

    mols = get_mols()
    # do partitioning to make sure there's no dataset contamination
    # 4 parts for training, 1 for validation and 1 for testing
    train_idx, val_idx, test_idx = get_partition_indices(len(mols), 6)

    import time
    t = time.time()
    T = lambda : time.time() - t


    for element in ("C","H","N"):
        print element, T()
        properties = []
        origin = []
        for n, mol in enumerate(mols):
            for i, atom in enumerate(mol.atomtypes):
                # remove the outermost hydrogens to keep size a bit lower
                if i in [3,4,5,mol.natoms-1,mol.natoms-2,mol.natoms-3, mol.natoms-5]:
                    # quick check that I don't do anything wrong
                    if mol.atomtypes[i] != "H":
                        quit("error :(")
                    continue
                if atom == element:
                    origin.append((n,i))
                    properties.append(mol.properties[i])

        origin = np.asarray(origin, dtype=int)
        properties = np.asarray(properties, dtype=object)
        X = properties[:,5].astype(float)
        Y = properties[:,6].astype(float)

        # store everything in pandas
        df = pandas.DataFrame(index=range(properties.shape[0]))
        df['molindex'] = origin[:,0]
        df['train'] = np.in1d(df.molindex.values, train_idx)
        df['val'] = np.in1d(df.molindex.values, val_idx)
        df['test'] = np.in1d(df.molindex.values, test_idx)

        df['value'] = remove_mean(X, Y)
        df['base'] = X
        df['ref'] = Y
        df['linregress'] = linreg(X, Y)
        df['label'] = properties[:,3].astype(str)

        X_train = df.base.values[df.train.values == True]
        X_val = df.base.values[df.val.values == True]
        X_test = df.base.values[df.test.values == True]
        Y_train = df.ref.values[df.train.values == True]
        Y_val = df.ref.values[df.val.values == True]
        Y_test = df.ref.values[df.test.values == True]

        # do linear regression based GM with up to 5 classes
        for n in range(2,5):
            name = 'GMLR%d' % n
            print name, T()

            df[name+"_cluster"] = 0
            labels = GM_linreg(df, n)
            atomtypes, atom_indices = np.unique(df.label.values[df.train.values == True], return_inverse = True)
            slopes, offsets = [], []
            for i in range(n):
                slope, offset = ss.linregress(X_train[labels[atom_indices] == i], Y_train[labels[atom_indices] == i])[:2]
                slopes.append(slope)
                offsets.append(offset)
            atomtypes, atype_indices = np.unique(df.label.values, return_inverse=True)
            slopes = np.asarray(slopes)
            offsets = np.asarray(offsets)
            D = Y - slopes[labels[atype_indices]] * X - offsets[labels[atype_indices]]
            df[name+"_value"] = D
            df[name+"_cluster"] = labels[atype_indices]

        #other_properties = properties[:,[0,1,2,4,5,6,7,8,9,10]].astype(float)
        other_properties = properties[:,[4,5,7,9]].astype(float)
        other_properties_train = other_properties[df.train.values == True]

        pca = sklearn.decomposition.PCA(n_components = 3, whiten=1)
        pca.fit(other_properties_train - other_properties_train.mean(0))
        train_pca = pca.transform(other_properties_train - other_properties_train.mean(0))
        all_pca = pca.transform(other_properties - other_properties_train.mean(0))
        df['pca1'] = all_pca[:,0]
        df['pca2'] = all_pca[:,1]
        df['pca3'] = all_pca[:,2]

        for eps in (0.7, 0.9, 1.1, 1.3):
            name = "DB%.1f" % eps
            print name, T()
            min_samples = 5
            labels = np.asarray([-1])
            while np.sum(labels == -1) > 0 or min_samples == 0:
                clust = sklearn.cluster.DBSCAN(eps=eps, metric="manhattan", min_samples=min_samples, n_jobs=6)
                # just set weight to zero for points not in training set
                clust.fit(all_pca, sample_weight=np.asarray([df.train.values == True], dtype=int).ravel())
                labels = clust.labels_
                min_samples -= 1
            if -1 in labels or (np.unique(labels, return_counts=True)[1] < 50).any():
                continue
            df[name+"_cluster"] = clust.labels_

            slopes, offsets = [], []
            for label in np.unique(clust.labels_):
                slope, offset = ss.linregress(X_train[clust.labels_[df.train.values == True] == label], Y_train[clust.labels_[df.train.values == True] == label])[:2]
                slopes.append(slope)
                offsets.append(offset)
            slopes = np.asarray(slopes)
            offsets = np.asarray(offsets)
            D = Y - slopes[labels] * X - offsets[labels]
            df[name+"_value"] = D


            #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
            #for i in np.unique(labels):
            #    #plt.scatter(X[labels == i],(Y-X)[labels == i])
            #    ax1.scatter(lol[:,0][labels == i], lol[:,1][labels == i],marker=".")
            #    ax2.scatter(lol[:,0][labels == i], lol[:,2][labels == i],marker=".")
            #    ax3.scatter(lol[:,1][labels == i], lol[:,2][labels == i],marker=".")
            #plt.title("eps = %.1f    min_samples = %d" % (eps, sam))
            #plt.savefig("%s_ma_%.1f_%d.png" %(element,eps,sam))
            #plt.close()
        for n in (2,3,4,5):
            name = "GM%d" % n
            print name, T()
            clust = sklearn.mixture.GaussianMixture(n_components=n, covariance_type="tied", n_init=3, init_params="kmeans", tol=1e-6, reg_covar=1e-2)
            clust.fit(train_pca)
            labels = clust.predict(all_pca)
            df[name+"_cluster"] = labels

            slopes, offsets = [], []
            for label in np.unique(labels):
                slope, offset = ss.linregress(X_train[labels[df.train.values == True] == label], Y_train[labels[df.train.values == True] == label])[:2]
                slopes.append(slope)
                offsets.append(offset)
            slopes = np.asarray(slopes)
            offsets = np.asarray(offsets)
            D = Y - slopes[labels] * X - offsets[labels]
            df[name+"_value"] = D

        df.to_pickle('target/%s\.pickle' % element)
        del df











if __name__ == "__main__":
    np.random.seed(1)
    #generate_descriptors()
    #generate_distance_matrices(filenames = sys.argv[1:])
    generate_preprocessing()

