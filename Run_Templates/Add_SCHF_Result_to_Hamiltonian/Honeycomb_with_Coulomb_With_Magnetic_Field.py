# Import dependencies
import numpy as np
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_PeierlsSub_Honeycomb
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import hartreefock_hamiltonian_addition_NonHermitian_Honeycomb

# Define system parameters
nval = 20
alpha_NH = 0.4
beta = 1.0
raw_data_location = ''
coulomb_v_value = 1.0

# Load in raw data and pick out relevant raw data for given V value given
raw_data = np.load(raw_data_location)
raw_data = raw_data[np.where(np.abs(raw_data[:, 0] - coulomb_v_value) < 0.001)[0], 1:].reshape(-1)

# form the Hamiltonian with SCHF obtained interaction term
hamiltonian = NonHermitian_PeierlsSub_Honeycomb(nval, alpha_NH, beta)
hamiltonian_with_schf_result = hartreefock_hamiltonian_addition_NonHermitian_Honeycomb(nval, hamiltonian,
                                                                                       raw_data, coulomb_v_value)


