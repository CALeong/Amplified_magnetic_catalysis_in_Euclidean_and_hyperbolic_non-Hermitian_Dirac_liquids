import numpy as np
import scipy

def Energy_Spectrum_Save(ham_dir, save_dir):
    ham_file_name = ham_dir.split('/')[-1]
    ham_file_name_noext = ham_file_name.split('.npy')[0]
    ham = np.load(ham_dir)
    eners = scipy.linalg.eigvals(ham)
    np.save(save_dir + '/' + ham_file_name_noext + '_EnergySpectrum', eners)

def DOS_Plot_Save(ener_spec_dir):
    espec = np.load(ener_spec_dir)
