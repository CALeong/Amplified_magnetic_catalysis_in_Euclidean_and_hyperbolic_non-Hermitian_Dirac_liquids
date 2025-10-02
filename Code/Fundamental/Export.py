import os

import numpy as np
import scipy
import sys

def npy_export_compatible_file_mathematica(npy_file, save_dir):
    data = np.load(npy_file)
    npy_file_name = npy_file.split('/')[-1]
    npy_file_name_noext = npy_file_name.split('.npy')[0]
    np.savetxt(save_dir + '/' + npy_file_name_noext + '.txt', data)

def npy_export_compatible_file_mathematica_over_directory(subdir, save_dir):
    os.chdir(subdir)
    files = os.listdir()
    for f in files:
        if 'system' in f or 'center' in f:
            path = subdir + '/' + f
            npy_export_compatible_file_mathematica(path, save_dir)

def npy_export_compatible_file_mathematica_over_directory_EIGVAL_File(subdir, save_dir):
    os.chdir(subdir)
    files = os.listdir()
    for f in files:
        if '_eigvals.npy' in f:
            path = subdir + '/' + f
            npy_export_compatible_file_mathematica(path, save_dir)

def diagonalization_result_export(ham_file,save_dir):
    ham_file_name = ham_file.split('/')[-1]
    ham_file_name_noext = ham_file_name.split('.npy')[0]
    ham = np.load(ham_file)
    eners, lsmat, rsmat = scipy.linalg.eig(ham, left=True, right=True)

    if np.all(np.imag(eners)==0):
        eners = np.real(eners)
    else:
        print('Problem: some energies are imaginary')
        print('Max imaginary value: {}'.format(np.max(np.imag(eners))))
        sys.exit(1)

    lsmat = np.conj(lsmat)

    rsmat = rsmat[:, np.argsort(eners)]
    lsmat = lsmat[:, np.argsort(eners)]
    eners = np.sort(eners)

    np.save(save_dir + '/' + 'energySpectrum_' + ham_file_name_noext, eners)
    np.save(save_dir + '/' + 'leftEigenVectorMatrix_' + ham_file_name_noext, lsmat)
    np.save(save_dir + '/' + 'rightEigenVectorMatrix_' + ham_file_name_noext, rsmat)

    npy_export_compatible_file_mathematica(save_dir + '/' + 'energySpectrum_' + ham_file_name_noext + '.npy', save_dir)
    # npy_export_compatible_file_mathematica(save_dir + '/' + 'leftEigenVectorMatrix_' + ham_file_name_noext + '.npy', save_dir)
    # npy_export_compatible_file_mathematica(save_dir + '/' + 'rightEigenVectorMatrix_' + ham_file_name_noext + '.npy', save_dir)

