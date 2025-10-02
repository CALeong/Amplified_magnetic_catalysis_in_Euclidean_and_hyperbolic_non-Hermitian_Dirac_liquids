from Fundamental.Number_Points import points
import numpy as np

def get_first_site_on_generations(p, num_levels):
    q = 3
    points_per_site = points(p, q, num_levels)[0]
    first_site_on_gen = np.array([0])
    for i in range(num_levels):
        first_site_on_gen = np.append(first_site_on_gen, first_site_on_gen[-1] + points_per_site[i])
    return(first_site_on_gen)

def get_sites_around_3side_plaquet_q3_hyperbolic_lattice(p, specific_level):
    q = 3
    first_sites_on_each_gen = get_first_site_on_generations(p, specific_level)
    if specific_level == 1:
        return(np.arange(0,p))
    elif specific_level == 2:
        return(np.concatenate((np.array([0,1]), np.arange(first_sites_on_each_gen[2], first_sites_on_each_gen[2]+(p-3)+1))))
    elif specific_level >= 3:
        previous_gen_sites_wanted = np.array([first_sites_on_each_gen[specific_level-1] + 1, first_sites_on_each_gen[specific_level-1] + 2])
        this_gen_sites_wanted = np.arange(first_sites_on_each_gen[specific_level], first_sites_on_each_gen[specific_level]+(p-3)+1)
        return(np.concatenate((previous_gen_sites_wanted, this_gen_sites_wanted)))

def get_sites_around_4side_plaquet_q3_hyperbolic_lattice(p, specific_level):
    q = 3
    first_sites_on_each_gen = get_first_site_on_generations(p, specific_level)
    if specific_level >= 3:
        previous_gen_sites_wanted = np.array([first_sites_on_each_gen[specific_level-1], first_sites_on_each_gen[specific_level-1] + 1,
                                              first_sites_on_each_gen[specific_level+1]-1
                                              ])
        this_gen_sites_wanted = np.concatenate((
            first_sites_on_each_gen[specific_level],
            np.arange(first_sites_on_each_gen[specific_level+1]-(p-4), first_sites_on_each_gen[specific_level+1]-1)
        ))
        return(np.concatenate((previous_gen_sites_wanted, this_gen_sites_wanted)))