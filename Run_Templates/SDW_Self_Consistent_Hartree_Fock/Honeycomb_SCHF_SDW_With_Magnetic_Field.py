# Import dependencies
import numpy as np
from Fundamental.Hubbard import onsite_hubbard_tight_binding_mat_peierls_substitution_honeycomb
from Fundamental.Eigenvalue_Degeneracy_Fix import small_chaos as small_disorder
from Fundamental.Hubbard import SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput_honeycomb as SCHF_calculate

# Define system parameters
nval = 20
alpha_NH = 0.4
small_disorder_strength = 10**(-3)
onsite_hubbard_U = 1.0
beta_list = [0.1, 0.3, 0.5]

# Define variables to accumulate data through each iteration of below for loop
all_system_data = np.array([])

# Iterate over all beta values to compute
for beta in beta_list:
    # Form the spinful tight-binding Hamiltonian and add small disorder for stabilization
    hamiltonian = onsite_hubbard_tight_binding_mat_peierls_substitution_honeycomb(nval, alpha_NH, beta)
    hamiltonian_withDisorder = hamiltonian + small_disorder(small_disorder_strength, hamiltonian.shape[0])

    # Perform the SCHF computation
    raw_data, system_data = SCHF_calculate(nval, hamiltonian_withDisorder,
                                           U_list=np.array([onsite_hubbard_U]), initial_guess=0.1, tolerance=0.001)

    # Append results to appropriate variable
    all_system_data = np.append(all_system_data, system_data)

# Here may want to save all_system_data
