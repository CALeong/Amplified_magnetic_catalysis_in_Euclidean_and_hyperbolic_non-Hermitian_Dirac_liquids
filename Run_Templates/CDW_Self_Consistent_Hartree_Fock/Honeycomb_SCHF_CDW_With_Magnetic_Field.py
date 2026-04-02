# Import dependencies
import numpy as np
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_PeierlsSub_Honeycomb
from Fundamental.Eigenvalue_Degeneracy_Fix import small_chaos as small_disorder
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import selfconsist_hartreefock_NonHermitian_Honeycomb_PBC_DisorderAlreadyAdded_SaveRawOutputs as SCHF_calculate

# Define system parameters
nval = 20
alpha_NH = 0.4
small_disorder_strength = 10**(-3)
coulomb_V = 0.3
beta_list = [0.1, 0.3, 0.5]

# Define variables to accumulate data through each iteration of below for loop
all_system_data = np.array([])

# Iterate over all beta values to compute
for beta in beta_list:
    # Form the spin-polarized tight-binding Hamiltonian and add small disorder for stabilization
    hamiltonian = NonHermitian_PeierlsSub_Honeycomb(nval, alpha_NH, beta)
    hamiltonian_withDisorder = hamiltonian + small_disorder(small_disorder_strength, hamiltonian.shape[0])

    # Perform the SCHF computation
    raw_data, system_data, center_data = SCHF_calculate(nval, hamiltonian_withDisorder,
                                                        initial_guess=0.1, tolerance=0.001,
                                                        hfcoeff_list=np.array([coulomb_V]))

    # Append results to appropriate variable
    all_system_data = np.append(all_system_data, system_data)

# Here may want to save all_system_data


