import os
import numpy as np
import pandas as pd

def sort_slices(slice_files):
    vstarts = np.array([])
    for sf in slice_files:
        associated_vstart = float(sf.split('Vstart')[1].split('.npy')[0])
        vstarts = np.append(vstarts, associated_vstart)
    return(slice_files[np.argsort(vstarts)])

def combine_slices_data(subdir,alist,savedir):
    os.chdir(subdir)
    files = os.listdir()
    sys_files = np.array([])
    cen_files = np.array([])
    raw_files = np.array([])
    for f in files:
        if 'system' in f:
            sys_files = np.append(sys_files, f)
        elif 'center' in f:
            cen_files = np.append(cen_files, f)
        elif 'raw' in f:
            raw_files = np.append(raw_files, f)
    for a in alist:
        if a == 0:
            a = 0
        sys_speca = np.array([])
        for sf in sys_files:
            if 'a{}_'.format(a) in sf:
                sys_speca = np.append(sys_speca, sf)
        cen_speca = np.array([])
        for cf in cen_files:
            if 'a{}_'.format(a) in cf:
                cen_speca = np.append(cen_speca, cf)
        raw_speca = np.array([])
        for rf in raw_files:
            if 'a{}_'.format(a) in rf:
                raw_speca = np.append(raw_speca, rf)
        sys_speca = sort_slices(sys_speca)
        cen_speca = sort_slices(cen_speca)
        raw_speca = sort_slices(raw_speca)
        sys_data = np.array([])
        for ss in sys_speca:
            sys_data = np.concatenate((sys_data,np.load(ss)))
        cen_data = np.array([])
        for cs in cen_speca:
            cen_data = np.concatenate((cen_data, np.load(cs)))
        raw_data = np.zeros((1,np.size(np.load(raw_speca[0]),1)))
        for rs in raw_speca:
            raw_data = np.vstack((raw_data, np.load(rs)))
        raw_data = raw_data[1:,:]
        np.save(savedir+'/'+'System_a{}'.format(a), sys_data)
        np.save(savedir+'/'+'Center_a{}'.format(a), cen_data)
        np.save(savedir+'/'+'Raw_a{}'.format(a), raw_data)

def sort_slices_sdw(slice_files):
    ustarts = np.array([])
    for f in slice_files:
        ustart_val_str = f.split('_')[-2]
        ustart_val_str = ustart_val_str.split('UStart')[1]
        ustart_val = float(ustart_val_str.replace('d','.'))
        ustarts = np.append(ustarts, ustart_val)
    return(slice_files[np.argsort(ustarts)])


def combine_slices_data_sdw(datadir, alist_str, savedir):
    os.chdir(datadir)
    allfiles = os.listdir()

    sysfiles = np.array([])
    rawfiles = np.array([])
    for af in allfiles:
        if 'sysdata' in af:
            sysfiles = np.append(sysfiles, af)
        elif 'rawdata' in af:
            rawfiles = np.append(rawfiles, af)

    for a in alist_str:
        sys_spec_a = np.array([])
        raw_spec_a = np.array([])

        for sf in sysfiles:
            if a in sf:
                sys_spec_a = np.append(sys_spec_a, sf)
        for rf in rawfiles:
            if a in rf:
                raw_spec_a = np.append(raw_spec_a, rf)

        sys_spec_a = sort_slices_sdw(sys_spec_a)
        raw_spec_a = sort_slices_sdw(raw_spec_a)

        sys_spec_a_combined = np.array([])
        raw_spec_a_combined = np.zeros((1,np.size(np.load(raw_spec_a[1]),1)))
        for ssa in sys_spec_a:
            sys_spec_a_combined = np.append(sys_spec_a_combined, np.load(ssa))
        for rsa in raw_spec_a:
            raw_spec_a_combined = np.vstack((raw_spec_a_combined, np.load(rsa)))

        raw_spec_a_combined = raw_spec_a_combined[1:,:]

        np.save(savedir + '/' + 'System_{}'.format(a), sys_spec_a_combined)
        np.save(savedir + '/' + 'Raw_{}'.format(a), raw_spec_a_combined)

def order_amag_subdirs(amag_dirs):
    amag_dir_vals = np.array([])
    for i in amag_dirs:
        name = i.split('amag_')[1]
        name_val = float(name.replace('d','.'))
        amag_files_vals = np.append(amag_dir_vals, name_val)
    return(np.argsort(amag_dir_vals))





