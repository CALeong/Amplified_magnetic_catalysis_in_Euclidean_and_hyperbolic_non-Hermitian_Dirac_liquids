import numpy as np
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import site_assignment_NonHermitian

def calc_from_raw_delta_data_NonHermitian(p,q,n,raw_delta_data):
    a_sites, b_sites = site_assignment_NonHermitian(p, q, n)
    deltas_list = np.array([])
    deltas_center_list = np.array([])
    for row in range(np.size(raw_delta_data,0)):
        deltas = raw_delta_data[row,1:]
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        a_deltas_center = np.array([])
        b_deltas_center = np.array([])
        a_site_counter = 0
        b_site_counter = 1
        for i in range(p):
            # print(i, i%2==0)
            if i % 2 == 0:
                a_deltas_center = np.append(a_deltas_center, deltas[int(a_sites[i - 1 * a_site_counter])])
                print(a_sites[i - 1 * a_site_counter])
                a_site_counter += 1
            else:
                b_deltas_center = np.append(b_deltas_center, deltas[int(b_sites[i - 1 * b_site_counter])])
                print(b_sites[i - 1 * b_site_counter])
                b_site_counter += 1
        a_deltas_center_avg = np.abs(np.average(a_deltas_center))
        b_deltas_center_avg = np.abs(np.average(b_deltas_center))
        deltas_center_list = np.append(deltas_center_list, 0.5 * (a_deltas_center_avg + b_deltas_center_avg))
    return(deltas_list, deltas_center_list)

from Fundamental.Number_Points import points

def calculate_center_values_from_raw_SDW(rawdata, p, q, n):
    center_final_results = np.array([])
    asites, bsites = site_assignment_NonHermitian(p,q,n)
    asites = np.array([int(i) for i in asites])
    bsites = np.array([int(i) for i in bsites])
    for row in range(np.size(rawdata, 0)):
        localorderparams = rawdata[row,1:]
        centvals_a_upspin = localorderparams[asites[:int(p/2)]]
        centvals_b_upspin = localorderparams[bsites[:int(p/2)]]
        downspin_a_indices = asites[:int(p / 2)] + points(p,q,n)[1]
        downspin_b_indices = bsites[:int(p / 2)] + points(p,q,n)[1]
        downspin_a_indices = np.array([int(i) for i in downspin_a_indices])
        downspin_b_indices = np.array([int(i) for i in downspin_b_indices])
        centvals_a_downspin = localorderparams[downspin_a_indices]
        centvals_b_downspin = localorderparams[downspin_b_indices]
        center_final_results = np.append(center_final_results, 0.5*(np.abs(np.average(centvals_a_upspin)) + np.abs(np.average(centvals_b_upspin))
                                                                    + np.abs(np.average(centvals_a_downspin)) + np.abs(np.average(centvals_b_downspin))))
    return(center_final_results)

