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
import sys
sys.path.insert(0, '/home/andyr/src/frb')
from get_COS_data import get_COS_data, plot_COS_data
from radial_profile2 import *

def make_profiles2(a_arr, r_arr, cdens_arr, a_bins, r_bins):
  """
  Splits data into three radial groups (50 kpc, 100 kpc, 150 kpc) and bins data
  with respect to azimuthal angle to find the percentiles of each raidal group.
  """
  r_bin_ids = np.digitize(r_arr, r_bins)
  a_bin_ids = np.digitize(a_arr, a_bins)
  radii = [0, 11, 21, 31]
  profile_data = [np.zeros([3, len(a_bins)]), np.zeros([3, len(a_bins)]), \
                  np.zeros([3, len(a_bins)])]
  for i, a_bin_id in enumerate(np.arange(len(a_bins))):
    for j in range(len(radii)-1):
      # ids = np.logical_and(a_bin_ids == a_bin_id, r_bin_ids > radii[j])
      # ids = np.logical_and(ids, r_bin_ids <= radii[j+1])
      sample = normalize_by_radius(cdens_arr, r_bin_ids, radii[j], a_bin_ids, a_bin_id)
      profile_data[j][0,i] = np.median(sample)
      profile_data[j][1,i] = np.percentile(sample, 25)
      profile_data[j][2,i] = np.percentile(sample, 75)
  
  return profile_data

def normalize_by_radius(cdens_arr, r_bin_ids, r, a_bin_ids, a_bin_id):
  sample = np.array([])
  r += 1
  a = r
  while a <= r+10:
    ids = np.logical_and(a_bin_ids == a_bin_id, r_bin_ids == a)
    print(True in ids)
    bin_data = cdens_arr[ids]
    sample = np.append(sample, np.sum(bin_data) / len(bin_data))
    a+=1
  return sample

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

  a_n_bins = 45
  r_n_bins = 31
  r_bins = np.linspace(0, 150, r_n_bins)
  a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
  a_bins = np.flip(a_bins, 0)

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
      r_arr = np.array([])
      for j in range(n_files):
        f = h5.File(v[j], 'r')
        a_arr = np.concatenate((a_arr, f['phi'].value))
        r_arr = np.concatenate((r_arr, f['radius'].value))
        cdens_arr = np.concatenate((cdens_arr, f["%s/%s" % (field, 'edge')].value))
      profile_data = make_profiles2(a_arr, r_arr, cdens_arr, a_bins, r_bins)
      
      # profile_data = make_profiles(a_arr, cdens_arr, a_bins, field, n_bins)
      # plot_hist2d(a_arr, cdens_arr, field, fn_head)
      for i in range(3):
        plot_profile(a_bins, profile_data[i], k, colors[c])
        finish_plot(field, COS_data, fn_head, i*50)



