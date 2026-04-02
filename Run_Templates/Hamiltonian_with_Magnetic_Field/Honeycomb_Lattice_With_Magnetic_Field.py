# Import dependencies
from Fundamental.Honeycomb_Lattice import honeycomb_number_plaquets
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_PeierlsSub_Honeycomb
from Fundamental.Hubbard import onsite_hubbard_tight_binding_mat_peierls_substitution_honeycomb

# Define system parameters
nval = 20
alpha_NH = 0.4
beta = 1.0
t = 1

# Generate the Hamiltonian (spin-polarized or spinful)
hamiltonian_spin_polarized = NonHermitian_PeierlsSub_Honeycomb(nval, alpha_NH, beta)
hamiltonian_spinful = onsite_hubbard_tight_binding_mat_peierls_substitution_honeycomb(nval, alpha_NH, beta)

# Calculate the total amount of magnetic flux through the system
total_number_plaquettes = honeycomb_number_plaquets(nval)
total_magnetic_flux = beta * total_number_plaquettes


