# Import dependencies
import numpy as np
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_PeierlsSub_Hamiltonian
from Fundamental.Eigenvalue_Degeneracy_Fix import small_chaos as small_disorder
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import selfconsist_hartreefock_NonHermitian_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient as SCHF_calculate

# Define system parameters
pval = 10
qval = 3
nval = 4
alpha_NH = 0.4
small_disorder_strength = 10**(-3)
coulomb_V = 0.5
t = 1
beta_list = [0.1, 0.5, 1.0]

# Define variables to accumulate data through each iteration of below for loop
all_system_data = np.array([])

# Iterate over all beta values to compute
for beta in beta_list:
    # Form the spin-polarized tight-binding Hamiltonian and add small disorder for stabilization
    hamiltonian = NonHermitian_PeierlsSub_Hamiltonian(pval, qval, nval, alpha_NH, t, beta)
    hamiltonian_withDisorder = hamiltonian + small_disorder(small_disorder_strength, hamiltonian.shape[0])

    # Perform the SCHF computation
    raw_data, system_data, center_data = SCHF_calculate(pval, qval, nval, hamiltonian_withDisorder,
                                                        initial_guess=0.1, tolerance=0.001,
                                                        hfcoeff_list=np.array([coulomb_V]))

    # Append results to appropriate variable
    all_system_data = np.append(all_system_data, system_data)

# Here may want to save all_raw_data and all_system_data
