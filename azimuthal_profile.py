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
  radii = [0, 10, 20]
  profile_data = [np.zeros([3, len(a_bins)]), np.zeros([3, len(a_bins)]), \
                  np.zeros([3, len(a_bins)])]
  for i, a_bin_id in enumerate(np.arange(len(a_bins))):
    for j in range(len(radii)):
      sample = normalize_by_radius(cdens_arr, r_bin_ids, radii[j], a_bin_ids, a_bin_id)
      profile_data[j][0,i] = np.median(sample)
      profile_data[j][1,i] = np.percentile(sample, 25)
      profile_data[j][2,i] = np.percentile(sample, 75)
  return profile_data

def normalize_by_radius(cdens_arr, r_bin_ids, r, a_bin_ids, a_bin_id):
  '''
  Takes a raidal bin of width 50kpc and splits it into 10 smaller bins of 5kpc.
  Takes the average column density of each bin and appends it to sample. Will
  return a len 10 array of the average cden of each 5kpc bin which is used to 
  give the percentiles. 
  '''
  sample = np.array([])
  radial_bin_id = r
  while radial_bin_id < r+10:
    ids = np.logical_and(a_bin_ids == a_bin_id, r_bin_ids == radial_bin_id)
    bin_data = cdens_arr[ids]
    avg = np.sum(bin_data) / len(bin_data)
    sample = np.append(sample, avg)
    radial_bin_id += 1
  return sample

def make_radial_profile(a_arr, r_arr, cdens_arr, a_bins, r_bins, a_n_bins, r_n_bins):
  '''
  Makes a profile based on radius with 3 bins of azimuthal angles: 0-30, 30-60, 
  60-90. 
  '''
  r_bin_ids = np.digitize(r_arr, r_bins)
  a_bin_ids = np.digitize(a_arr, a_bins)
  angle_bins = range(a_n_bins)
  profile_data = np.zeros([3, r_n_bins])
  for j in range(a_n_bins-1):
    profile_data = np.concatenate((profile_data, np.zeros([3, r_n_bins])), axis=0)
  for r_bin_id in range(r_n_bins):
    for j in range(len(profile_data)):
      ids = np.logical_and(r_bin_ids == r_bin_id, a_bin_ids == angle_bins[j])
      sample = cdens_arr[ids]
      profile_data[j][0,r_bin_id] = np.median(sample)
      profile_data[j][1,r_bin_id] = np.percentile(sample, 25)
      profile_data[j][2,r_bin_id] = np.percentile(sample, 75)
  return profile_data

def angle_profile_big(a_arr, r_arr, cdens_arr, a_bins, r_bins, a_n_bins, r_n_bins):
  '''
  Makes 3 bins for angle and radius and plots median point for each with
  error bars of 1 standard deviation.
  '''
  r_bin_ids = np.digitize(r_arr, r_bins)
  a_bin_ids = np.digitize(a_arr, a_bins)
  cden_data = [np.zeros([2, a_n_bins]), np.zeros([2, a_n_bins]), \
                  np.zeros([2, a_n_bins])]
  angle_data = [np.zeros([2, a_n_bins]), np.zeros([2, a_n_bins]), \
                np.zeros([2, a_n_bins])]
  for r_bin_id in range(r_n_bins):
    for a_bin_id in range(a_n_bins):
      ids = np.logical_and(r_bin_ids == r_bin_id, a_bin_ids == a_bin_id)
      sample = cdens_arr[ids]
      cden_data[r_bin_id][0,a_bin_id] = np.median(sample)
      cden_data[r_bin_id][1,a_bin_id] = np.std(sample)
      angle_data[r_bin_id][0,a_bin_id] = np.median(a_arr[ids])
      angle_data[r_bin_id][1,a_bin_id] = np.std(a_arr[ids])
  return cden_data, angle_data

def fplot_angle(ion):
  plt.title('%s' % ion)
  plt.xlabel('Azimuthal Angle [degrees]')
  plt.xlim((0,90))
  plt.legend(title='Radius')
  print('%s_angle.png' % ion)
  plt.savefig('plots/%s_angle.png' % ion)
  plt.clf()

def fplot_radius(ion):
  plt.title('%s' % ion)
  plt.xlabel('Impact Parameter [kpc]')
  plt.xlim((0,50))
  plt.legend(title='Azimuthal Angle')
  print('%s_radial.png' % ion)
  plt.savefig('plots/%s_radial.png' % ion)
  plt.clf()

def fplot_big_angle(ion):
  plt.title('%s' % ion)
  plt.xlabel('Azimuthal Angle [Degrees]')
  plt.xlim((0,90))
  plt.legend(title='Impact Parameter')
  print('%s_big_angle.png' % ion)
  plt.savefig('plots/%s_big_angle.png' % ion)
  plt.clf()

def plot_big_angle(angle_data, cden_data, label, color, marker):
  '''
  Plots the azimuthal angle vs Cden for 3 angular and 3 radial bins along with
  a error bar of 1 std for each axis at each point
  '''
  # makes sure the error bars never go below 0
  ylower = np.maximum(1e-29, cden_data[0,:] - cden_data[1,:])
  yerr_lower = cden_data[0,:] - ylower
  
  # plots points
  plt.errorbar(angle_data[0,:], cden_data[0,:], xerr=angle_data[1,:], yerr=yerr_lower, marker=marker, label=label)
  plt.semilogy()

def radial_profile_big(a_arr, r_arr, cdens_arr, a_bins, r_bins, a_n_bins, r_n_bins):
  '''
  Makes 2 bins for angle and 4 for radius and plots median point for each with
  error bars of 1 standard deviation.
  '''
  r_bin_ids = np.digitize(r_arr, r_bins)
  a_bin_ids = np.digitize(a_arr, a_bins)
  cden_data = [np.zeros([2, r_n_bins]), np.zeros([2, r_n_bins])]
  radius_data = [np.zeros([2, r_n_bins]), np.zeros([2, r_n_bins])]
  for r_bin_id in range(r_n_bins):
    for a_bin_id in range(a_n_bins):
      ids = np.logical_and(r_bin_ids == r_bin_id, a_bin_ids == a_bin_id)
      sample = cdens_arr[ids]
      cden_data[a_bin_id][0,r_bin_id] = np.median(sample)
      cden_data[a_bin_id][1,r_bin_id] = np.std(sample)
      radius_data[a_bin_id][0,r_bin_id] = np.median(r_arr[ids])
      radius_data[a_bin_id][1,r_bin_id] = np.std(r_arr[ids])
  return cden_data, radius_data

def plot_big_radius(radius_data, cden_data, label, color, marker):
  '''
  Plots b vs Cden for 2 angular and 4 radial bins along with
  a error bar of 1 std for each axis at each point
  '''
  # makes sure the error bars never go below 0
  ylower = np.maximum(1e-29, cden_data[0,:] - cden_data[1,:])
  yerr_lower = cden_data[0,:] - ylower
  
  # plots points
  plt.errorbar(radius_data[0,:], cden_data[0,:], xerr=radius_data[1,:], yerr=yerr_lower, marker=marker, label=label)
  plt.semilogy()

def fplot_big_radius(ion):
  plt.title('%s' % ion)
  plt.xlabel('b [kpc]')
  plt.xlim((0,90))
  plt.legend(title='Azimuthal Angle')
  print('%s_big_radius.png' % ion)
  plt.savefig('plots/%s_big_radius.png' % ion)
  plt.clf()

if __name__ == '__main__':
  """
  Takes file generated by azimuthal_projections.py and creates column density
  plots.
  """

  threshold = {'H_number_density' : 10**16, 'O_p5_number_density':1e14, 'density':1e-4}

  # Get observational data from file
  COS_data = get_COS_data()

  # Color cycling
  colors = 3*['black', 'cyan', 'green', 'magenta', 'yellow', 'blue', 'red']
  markers = ['o', '^', 's']

  fn_head = sys.argv[1].split('.')[0]
  profiles_dict = read_parameter_file(sys.argv[1])

  # Get the list of ion_fields from the first file available
  fn = list(profiles_dict.values())[0][0]
  f = h5.File(fn, 'r')
  ion_fields = list(f.keys())
  ion_fields.remove('radius')
  ion_fields.remove('phi')
  f.close()

  # Step through each ion and make plots of azimuthal angle vs N
  for field in ion_fields:
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
      
      # create bins for data
      a_n_bins = 9
      r_n_bins = 30
      r_bins = np.linspace(150, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      profile_data = make_profiles2(a_arr, r_arr, cdens_arr, a_bins, r_bins)
      for i in range(3):
        plot_profile(np.linspace(0, 90, a_n_bins), profile_data[i], '%s < b < %s kpc' % \
                    (50*i, 50*i + 50), colors[3*i])
      ion = finish_plot(field, COS_data, fn_head)
      fplot_angle(ion)

      # redefine bins
      a_n_bins = 9
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      # Step through each ion and make plots of radius vs N for 3 radial bins
      radial_data = make_radial_profile(a_arr, r_arr, cdens_arr, a_bins, r_bins, a_n_bins, r_n_bins)
      ion = finish_plot(field, COS_data, fn_head)
      for i in range(3):
        plot_profile(r_bins, radial_data[i], '%s < Φ < %s degrees' % \
                    (30*i, 30*i + 30), colors[3*i])
      fplot_radius(ion)

      # Make plot similar to paper of phi vs N
      r_n_bins = 3
      r_bins = np.linspace(80, 20, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)

      cden_data, angle_data = angle_profile_big(a_arr, r_arr, cdens_arr, a_bins, r_bins, a_n_bins, r_n_bins)
      ion = finish_plot(field, COS_data, fn_head)
      plot_big_angle(angle_data[0], cden_data[0], 'b < 40 kpc', colors[0], markers[0])
      for i in range(1,3):
        plot_big_angle(angle_data[i], cden_data[i], '%s < b < %s kpc' % \
                    (i*20 + 20, i*20 + 40), colors[3*i], markers[i])
      fplot_big_angle(ion)

      # Makes plot similar to paper of b vs N
      r_n_bins = 4
      r_bins = np.linspace(80, 20, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)
      a_n_bins = 2
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      cden_data, radius_data = radial_profile_big(a_arr, r_arr, cdens_arr, a_bins, r_bins, a_n_bins, r_n_bins)
      ion = finish_plot(field, COS_data, fn_head)
      plot_big_radius(radius_data[0], cden_data[0], 'Φ < 45 degrees', colors[0], markers[0])
      plot_big_radius(radius_data[1], cden_data[1], 'Φ > 45 degrees', colors[3], markers[1])
      fplot_big_radius(ion)


