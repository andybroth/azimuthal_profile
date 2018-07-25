"""
Combine column densities into plots

python azimuthal_profile.py parameter_file

where parameter file is formatted as:

m12i_ref11:
snapshot_600_cdens.h5

m12i_ref11_x3:
snapshot_600_cdens.h5
snapshot_600_cdens.h5
snapshot_600_cdens.h5

REMEMBER THE TRAILING WHITESPACE LINE AT THE BOTTOM OF THIS FILE.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yt
import numpy as np
import h5py as h5
import sys
from matplotlib.colors import LogNorm
from get_COS_data import get_COS_data, plot_COS_data
import sys
sys.path.insert(0, '/home/andyr/src/frb')
from radial_profile2 import *

if __name__ == '__main__':
  """
  Takes file generated by azimuthal_projections.py and creates plot
  """

  threshold = {'H_number_density' : 10**16, 'O_p5_number_density':1e14, 'density':1e-4}

  # Get observational data from file
  COS_data = get_COS_data()

  # Color cycling
  colors = 3*['black', 'cyan', 'green', 'magenta', 'yellow', 'blue', 'red']

  fn_head = sys.argv[1].split('.')[0]
  profiles_dict = read_parameter_file(sys.argv[1])

  n_bins = 100
  r_bins = np.linspace(1, 200, n_bins)
  a_bins = np.linspace(0, 90, n_bins)


# Get the list of ion_fields from the first file available
  fn = list(profiles_dict.values())[0][0]
  f = h5.File(fn, 'r')
  ion_fields = list(f.keys())
  ion_fields.remove('radius')
  ion_fields.remove('phi')
  f.close()

  # Step through each ion
    for field in ion_fields:
        # setup plot?
        for c, (k,v) in enumerate(profiles_dict.items()):
            n_files = len(v)
            cdens_arr = np.array([])
            a_arr = np.array([])
            for j in range(n_files):
                f = h5.File(v[j], 'r')
                a_arr = np.concatenate((a_arr, f['phi'].value))
                cdens_arr = np.concatenate((cdens_arr, f["%s/%s" % (field, 'edge')].value))

            profile_data = make_profiles(a_arr, cdens_arr, a_bins, field)
            # plot_hist2d(r_arr, cdens_arr, field, fn_head)
            plot_profile(a_bins, profile_data, k, colors[c])
        finish_plot(field, COS_data, fn_head)



