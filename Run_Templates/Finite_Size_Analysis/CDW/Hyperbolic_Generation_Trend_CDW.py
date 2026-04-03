# Import dependencies
import numpy as np
from Fundamental.NonHermitian_Hamiltonian import site_assignment
from Fundamental.Hamiltonian import H0
from Fundamental.Number_Points import points

# Define functions for computation
def get_gen_sites_sublattice_basis(p, num_levels):
    sites_on_gen = {}
    points_per_level, tot_num_points = points(p, 3, num_levels)
    asites, bsites = site_assignment(p, 3, num_levels, H0(p, 3, num_levels))
    newbasis = np.concatenate((asites, bsites))
    newbasis = np.array([int(i) for i in newbasis])
    for current_gen in range(1, num_levels+1):
        sites_on_level_indices = np.arange(np.sum(points_per_level[:current_gen-1]),
                                           np.sum(points_per_level[:current_gen]), dtype=int)
        sites_on_level_indices_newbasis = np.intersect1d(sites_on_level_indices, newbasis, return_indices=True)[2]
        new_entry = sites_on_level_indices_newbasis
        sites_on_gen['n={}'.format(current_gen)] = np.array([int(i) for i in new_entry])
    return(sites_on_gen)


def get_delta_sublattice_spinless_genlocalized(deltavals, p, numlevels):
    cdw_orders = np.array([])
    sites_on_gen = get_gen_sites_sublattice_basis(p, numlevels)
    tot_num_points = points(p, 3, numlevels)[1]
    for nl in range(1, numlevels + 1):
        relsites = sites_on_gen['n={}'.format(nl)]
        asites = deltavals[relsites[np.where(relsites < tot_num_points / 2)[0]]]
        bsites = deltavals[relsites[np.where(relsites >= tot_num_points / 2)[0]]]
        cdw_orders = np.append(cdw_orders, 0.5 * (np.abs(np.average(asites)) + np.abs(np.average(bsites))))
    return (cdw_orders)


def cdw_order_parameters_localgen(rawdata, p, num_gens):
    cdw_orders = np.zeros((1, num_gens+1))
    # rawdata = np.load(rawdatadir)
    for row in range(np.size(rawdata, 0)):
        vval = rawdata[row, 0]
        deltavals = rawdata[row, 1:]
        cdw_vals = get_delta_sublattice_spinless_genlocalized(deltavals, p, num_gens)
        cdw_orders = np.vstack((cdw_orders, np.concatenate((np.array([vval]), cdw_vals))))
    return(cdw_orders[1:,:])


# Provide paths to the raw data file which corresponds with the larger (N number of generations)
# and smaller (N-1 number of generations) systems being considered
rawdir_larger = 'path to N generation system raw data file'
rawdir_smaller = 'path to N-1 generation system raw data file'

# Define system parameters
vval = 0.4
pval = 10
nval_large = 4
nval_small = 3

# Perform calculation to get generation trend result

rawdata_large = np.load(rawdir_larger)
reldata_large = rawdata_large[np.where(np.abs(np.real(rawdata_large[:, 0] - vval)) < 0.001)[0], :]
result_large = cdw_order_parameters_localgen(reldata_large, pval, nval_large)

rawdata_small = np.load(rawdir_smaller)
reldata_small = rawdata_small[np.where(np.abs(np.real(rawdata_small[:, 0] - vval)) < 0.001)[0], :]
result_small = cdw_order_parameters_localgen(reldata_small, pval, nval_small)


