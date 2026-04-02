# Import dependencies
from Fundamental.Hamiltonian_PeierlsSubstitution import Number_Plaquets
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_PeierlsSub_Hamiltonian
from Fundamental.Hubbard import onsite_hubbard_tight_binding_mat_peierls_substitution

# Define system parameters
pval = 10
qval = 3
nval = 4
alpha_NH = 0.4
beta = 1.0
t = 1

# Generate the Hamiltonian (spin-polarized or spinful)
hamiltonian_spin_polarized = NonHermitian_PeierlsSub_Hamiltonian(pval, qval, nval, alpha_NH, t, beta)
hamiltonian_spinful = onsite_hubbard_tight_binding_mat_peierls_substitution(pval, qval, nval, alpha_NH, beta)

# Calculate the total amount of magnetic flux through the system
total_number_plaquettes = Number_Plaquets(pval, qval, nval)[1]
total_magnetic_flux = beta * total_number_plaquettes


