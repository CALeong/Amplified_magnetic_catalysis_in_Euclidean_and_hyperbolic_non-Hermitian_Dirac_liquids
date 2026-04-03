# Import dependencies
import numpy as np
from Fundamental.Honeycomb_Lattice import honeycomb_points
from Fundamental.Honeycomb_Lattice import honeycomb_lattice
from Fundamental.NonHermitian_Hamiltonian import site_assignment_honeycomb

# Define functions for calculations
def get_gen_sites_sublattice_basis_spin_doubled_honeycomb(num_levels):
    sites_on_gen = {}
    points_per_level, tot_num_points = honeycomb_points(num_levels)
    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    newbasis = np.concatenate((asites, bsites))
    newbasis = np.array([int(i) for i in newbasis])
    for current_gen in range(1, num_levels+1):
        sites_on_level_indices = np.arange(np.sum(points_per_level[:current_gen-1]),
                                           np.sum(points_per_level[:current_gen]), dtype=int)
        sites_on_level_indices_newbasis = np.intersect1d(sites_on_level_indices, newbasis, return_indices=True)[2]
        new_entry = np.concatenate((sites_on_level_indices_newbasis, sites_on_level_indices_newbasis + tot_num_points))
        sites_on_gen['n={}'.format(current_gen)] = np.array([int(i) for i in new_entry])
    return(sites_on_gen)


def get_delta_sublattice_spin_vals_genlocalized_honeycomb(deltavals, numlevels):
    afm_orders = np.array([])
    mag_orders = np.array([])
    sites_on_gen = get_gen_sites_sublattice_basis_spin_doubled_honeycomb(numlevels)
    tot_num_points = honeycomb_points(numlevels)[1]
    for nl in range(1, numlevels+1):
        relsites = sites_on_gen['n={}'.format(nl)]
        aupspin = deltavals[relsites[np.where((relsites < tot_num_points/2))[0]]]
        bupspin = deltavals[relsites[np.where((relsites >= tot_num_points/2) & (relsites < tot_num_points))[0]]]
        adownspin = deltavals[relsites[np.where((relsites >= tot_num_points) & (relsites < 3*tot_num_points/2))[0]]]
        bdownspin = deltavals[relsites[np.where((relsites >= 3*tot_num_points/2))[0]]]
        afm_orders = np.append(afm_orders, 0.5*(np.abs(np.average(aupspin)) + np.abs(np.average(bupspin)) +
                                                np.abs(np.average(adownspin)) + np.abs(np.average(bdownspin))))
        mag_orders = np.append(mag_orders, 0.5*(np.abs(np.average(aupspin)) - np.abs(np.average(bupspin)) +
                                                np.abs(np.average(adownspin)) - np.abs(np.average(bdownspin))))
    return(afm_orders, mag_orders)


def magnetic_order_parameters_localgen_honeycomb(rawdata, num_gens):
    afm_orders = np.zeros((1, num_gens+1))
    magnetization_orders = np.zeros((1, num_gens+1))
    for row in range(np.size(rawdata, 0)):
        uval = rawdata[row, 0]
        deltavals = rawdata[row, 1:]
        afm_vals, mag_vals = get_delta_sublattice_spin_vals_genlocalized_honeycomb(deltavals, num_gens)
        afm_orders = np.vstack((afm_orders, np.concatenate((np.array([uval]), afm_vals))))
        magnetization_orders = np.vstack((magnetization_orders, np.concatenate((np.array([uval]), mag_vals))))
    return(afm_orders[1:,:], magnetization_orders[1:,:])


# Provide paths to the raw data file which corresponds with the larger (N number of generations)
# and smaller (N-1 number of generations) systems being considered
rawdir_larger = 'path to N generation system raw data file'
rawdir_smaller = 'path to N-1 generation system raw data file'

# Define system parameters
uval = 1
nval_large = 20
nval_small = 19

# Perform calculation to get generation trend result

rawdata_large = np.load(rawdir_larger)
reldata_large = rawdata_large[np.where(np.abs(np.real(rawdata_large[:, 0] - uval)) < 0.001)[0], :]
result_large = magnetic_order_parameters_localgen_honeycomb(reldata_large, nval_large)[0]

rawdata_small = np.load(rawdir_smaller)
reldata_small = rawdata_small[np.where(np.abs(np.real(rawdata_small[:, 0] - uval)) < 0.001)[0], :]
result_small = magnetic_order_parameters_localgen_honeycomb(reldata_small, nval_small)[0]

