import numpy as np
from Fundamental.Number_Points import points
from scipy.sparse import dok_matrix
from Fundamental.Hamiltonian_PeierlsSubstitution import Number_Plaquets

def number_points_q3_general(p, num_levels):
    q = 3

    if num_levels == 1:
        sites_per_level = np.array([p])
    elif num_levels == 2:
        sites_per_level = np.array([p, p * (p - 3)])
    elif num_levels >= 3:
        sites_per_level = np.array([p, p*(p-3), (p-4)*p + p*(p-5)*(p-3)])
    number_plaqs_per_level = Number_Plaquets(p, q, num_levels)[0]
    for n in range(3, num_levels):
        sites_per_level = np.append(sites_per_level, (p-5)*(number_plaqs_per_level[n-1] - number_plaqs_per_level[n-2])*(p-3)
                                    + (p-6)*number_plaqs_per_level[n-2]*(p-3)
                                    + number_plaqs_per_level[n-1]*(p-4))

    sites_per_level = [int(i) for i in sites_per_level]
    return(sites_per_level, np.sum(sites_per_level))

def general_q3_hamiltonian(p, num_levels):
    q = 3
    t = 1

    sites_per_level, tot_num_points = number_points_q3_general(p, num_levels)

    #Find index of first site on each level (not counting last level)
    sites_cumulative = np.array([0, sites_per_level[0]])
    for i in range(1, num_levels):
        sites_cumulative = np.append(sites_cumulative, sites_cumulative[-1] + sites_per_level[i])

    ham = dok_matrix((int(tot_num_points), int(tot_num_points)), dtype=int)
    # ham = np.zeros((int(tot_num_points), int(tot_num_points)), dtype=int)

    for n in range(num_levels-1): #iterate over all generations except last generation
        sion = np.arange(sites_cumulative[n], sites_cumulative[n+1]) #sion = site_indices_on_level

        intralayer_hopping_counter = 0
        four_side_counter = 0
        for i in range(len(sion)):

            #Handles nearest-neighbor on same generation hopping
            ham[sion[i], np.take(sion, i + 1, mode='wrap')] = t
            ham[np.take(sion, i + 1, mode='wrap'), sion[i]] = t
            ham[sion[i], sion[i - 1]] = t
            ham[sion[i - 1], sion[i]] = t

            #Now to handle connections to generation above
            first_point_nxt_lvl = sites_cumulative[n+1]

            if n != 0:
                if intralayer_hopping_counter % (p-3) == 0 and i != 0:
                    four_side_counter = four_side_counter + 1
                    intralayer_hopping_counter = intralayer_hopping_counter + 1
                elif i == 0:
                    intralayer_hopping_counter = intralayer_hopping_counter + 1
                else:
                    ham[sion[i], first_point_nxt_lvl + (intralayer_hopping_counter-1-2*four_side_counter)*(p - 3) + four_side_counter*(p - 4)] = t
                    ham[first_point_nxt_lvl + (intralayer_hopping_counter-1-2*four_side_counter)*(p - 3) + four_side_counter*(p - 4), sion[i]] = t
                    intralayer_hopping_counter = intralayer_hopping_counter + 1
            else:
                ham[sion[i], first_point_nxt_lvl + intralayer_hopping_counter * (p - 3)] = t
                ham[first_point_nxt_lvl + intralayer_hopping_counter * (p - 3), sion[i]] = t
                intralayer_hopping_counter = intralayer_hopping_counter + 1

    #Now to handle last level
    sion = np.arange(sites_cumulative[num_levels-1], sites_cumulative[num_levels])
    for i in range(len(sion)):
        # Handles nearest-neighbor on same generation hopping
        ham[sion[i], np.take(sion, i + 1, mode='wrap')] = t
        ham[np.take(sion, i + 1, mode='wrap'), sion[i]] = t
        ham[sion[i], sion[i - 1]] = t
        ham[sion[i - 1], sion[i]] = t

    return(ham)