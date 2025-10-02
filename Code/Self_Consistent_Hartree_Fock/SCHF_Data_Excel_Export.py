import pandas as pd
import numpy as np
import os

def schf_export_excel(working_dir, export_dir, vvals, system=True, center=False):
    os.chdir(working_dir)
    files = os.listdir()
    if system == True:
        sys_files = [f for f in files if 'System' in f]

        df_sys = pd.DataFrame({'V': vvals})

        for sf in sys_files:
            data = np.load(working_dir + '/' + sf)
            df_sys[sf] = data

        df_sys.to_excel(export_dir + '/' + 'system_data.xlsx')

    if center == True:
        cen_files = [f for f in files if 'Center' in f]

        df_cen = pd.DataFrame({'V': vvals})

        for cf in cen_files:
            data = np.load(working_dir + '/' + cf)
            df_cen[cf] = data

        df_cen.to_excel(export_dir + '/' + 'center_data.xlsx')
