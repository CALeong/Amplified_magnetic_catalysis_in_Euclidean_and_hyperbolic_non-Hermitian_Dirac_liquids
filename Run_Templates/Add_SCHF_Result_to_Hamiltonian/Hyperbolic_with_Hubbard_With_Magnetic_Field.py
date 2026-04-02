# Import dependencies
import numpy as np
from Fundamental.Hubbard import onsite_hubbard_tight_binding_mat_peierls_substitution
from Fundamental.Hubbard import add_onsite_hubbard_repulsion_hartree_decomposition

# Define system parameters
pval = 10
qval = 3
nval = 4
alpha_NH = 0.4
raw_data_location = ''
hubbard_u_value = 2.5
beta = 1.0

# Load in raw data and pick out relevant raw data for given U value given
raw_data = np.load(raw_data_location)
raw_data = raw_data[np.where(np.abs(raw_data[:, 0] - hubbard_u_value) < 0.001)[0], 1:].reshape(-1)

# form the Hamiltonian with SCHF obtained interaction term
hamiltonian = onsite_hubbard_tight_binding_mat_peierls_substitution(pval, qval, nval, alpha_NH, beta)
hamiltonian_with_schf_result = add_onsite_hubbard_repulsion_hartree_decomposition(pval, qval, nval, hamiltonian,
                                                                                  hubbard_u_value, raw_data)


