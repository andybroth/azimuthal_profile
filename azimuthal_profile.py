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
from radial_profile2 import read_parameter_file, plot_profile, finish_plot, limits_from_field

def make_profiles2(a_arr, a_bins, a_n_bins, cdens_arr, r_arr, r_bins, r_n_bins, normalize):
  '''
  Default is angle vs N plot w/ 3 radial bins in legend. normalize should be set
  to true for this to work. To make radius vs N plots, flip all angular and radius
  inputs around cdens_arr and set normalize to false.
  '''
  r_bin_ids = np.digitize(r_arr, r_bins)
  a_bin_ids = np.digitize(a_arr, a_bins)
  if normalize:
    radii = [0, 10, 20]
  else:
    radii = range(r_n_bins)

  profile_data = [np.zeros([3, a_n_bins]) for _ in range(r_n_bins)]
  for a_bin_id in range(a_n_bins):
    for j, radius in enumerate(radii):
      if normalize:
        sample = normalize_by_radius(cdens_arr, r_bin_ids, radius, a_bin_ids, a_bin_id)
      else:
        ids = np.logical_and(r_bin_ids == j, a_bin_ids == a_bin_id)
        sample = cdens_arr[ids]
      profile_data[j][0,a_bin_id] = np.median(sample)
      profile_data[j][1,a_bin_id] = np.percentile(sample, 25)
      profile_data[j][2,a_bin_id] = np.percentile(sample, 75)
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

def big_profile(a_arr, a_bins, a_n_bins, cdens_arr, r_arr, r_bins, r_n_bins):
  '''
  Makes profile of data with error bars. When parameters are copied as above, it
  makes different angular bins which show up in legend. To make radial bins, just
  flip all the angular and radial inputs around the cdens_arr input.
  '''
  r_bin_ids = np.digitize(r_arr, r_bins)
  a_bin_ids = np.digitize(a_arr, a_bins)
  cden_data = [np.zeros([2, a_n_bins]) for _ in range(r_n_bins)]
  angle_data = [np.zeros([2, a_n_bins]) for _ in range(r_n_bins)]
  
  for r_bin_id in range(r_n_bins):
    for a_bin_id in range(a_n_bins):
      ids = np.logical_and(r_bin_ids == r_bin_id, a_bin_ids == a_bin_id)
      sample = cdens_arr[ids]
      cden_data[r_bin_id][0,a_bin_id] = np.median(sample)
      cden_data[r_bin_id][1,a_bin_id] = np.std(sample)
      angle_data[r_bin_id][0,a_bin_id] = np.median(a_arr[ids])
      angle_data[r_bin_id][1,a_bin_id] = np.std(a_arr[ids])
  return cden_data, angle_data

def fplot_angle(ion, description):
  plt.title('%s' % ion)
  plt.xlabel('Azimuthal Angle [degrees]')
  plt.xlim((0,90))
  # if limits_from_field(field):
  #   plt.ylim(limits_from_field(field))
  plt.legend(title='Radius')
  if len(description) > 0:
    print('%s_%s_angle.png' % (ion, description))
    plt.savefig('plots/%s_%s_angle.png' % (ion, description))
  else:
    print('%s_angle.png' % ion)
    plt.savefig('plots/%s_angle.png' % ion)
  plt.clf()

def fplot_radius(ion, description):
  plt.title('%s' % ion)
  plt.xlabel('Impact Parameter [kpc]')
  max = 150
  if description == 'test':
    max = 50
  elif description == 'big':
    max = 80
  plt.xlim((0,max))
  # if limits_from_field(field):
  #   plt.ylim(limits_from_field(field))
  plt.legend(title='Azimuthal Angle')
  if len(description) > 0:
    print('%s_%s_radius.png' % (ion, description))
    plt.savefig('plots/%s_%s_radius.png' % (ion, description))
  else:
    print('%s_radial.png' % ion)
    plt.savefig('plots/%s_radial.png' % ion)
  plt.clf()

def plot_big_angle(angle_data, cden_data, label, color, marker):
  '''
  Plots the azimuthal angle vs Cden for 3 angular and 3 radial bins along with
  a error bar of 1 std for each axis at each point
  '''
  # makes sure the error bars never go below 0
  ylower = np.maximum(1e-5, cden_data[0,:] - cden_data[1,:])
  yerr_lower = cden_data[0,:] - ylower
  
  # plots points
  # plt.errorbar(angle_data[0,:], cden_data[0,:], xerr=angle_data[1,:], yerr=[yerr_lower, cden_data[1,:]], marker=marker, label=label)
  plt.errorbar(angle_data[0,:], cden_data[0,:], xerr=0, yerr=0, marker=marker, label=label)
  plt.semilogy()

def plot_big_radius(radius_data, cden_data, label, color, marker):
  '''
  Plots b vs Cden for 2 angular and 4 radial bins along with
  a error bar of 1 std for each axis at each point
  '''
  # makes sure the error bars never go below 0
  ylower = np.maximum(1e-5, cden_data[0,:] - cden_data[1,:])
  yerr_lower = cden_data[0,:] - ylower
  
  # plots points
  # plt.errorbar(radius_data[0,:], cden_data[0,:], xerr=radius_data[1,:], yerr=[yerr_lower, cden_data[1,:]], marker=marker, label=label)
  plt.errorbar(radius_data[0,:], cden_data[0,:], xerr=0, yerr=0, marker=marker, label=label)
  plt.semilogy()

if __name__ == '__main__':
  """
  Takes file generated by azimuthal_projections.py and creates column density
  plots.
  """

  threshold = {'H_number_density' : 10**16, 'O_p5_number_density':1e14, 'density':1e-4}

  # # Get observational data from file
  # COS_data = get_COS_data()
  COS_data = ""

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
      if field == 'Mg_p1_number_density' or field == 'O_p5_number_density':
        cdens_arr *= 6.02 * 10**23

      # create bins for data
      a_n_bins = 9
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      r_n_bins = 30
      r_bins = np.linspace(150, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)

      profile_data = make_profiles2(a_arr, a_bins, a_n_bins, cdens_arr, r_arr, r_bins, r_n_bins, True)
      for i in range(3):
        plot_profile(np.linspace(0, 90, a_n_bins), profile_data[i], '%s < b < %s kpc' % \
                    (50*i, 50*i + 50), colors[3*i])
      ion = finish_plot(field, COS_data, fn_head)
      fplot_angle(ion, '')

      # redefine bins
      a_n_bins = 3
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      r_n_bins = 30
      r_bins = np.linspace(150, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)

      # Step through each ion and make plots of radius vs N for 3 radial bins
      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins, False)
      ion = finish_plot(field, COS_data, fn_head)
      angle = 90/a_n_bins
      for i in range(a_n_bins):
        color = 3*i
        while color >= len(colors):
          color -= len(colors)
        plot_profile(r_bins, radial_data[i], '%s < Φ < %s degrees' % (angle*i, angle*i + angle), colors[color])
      fplot_radius(ion, '')
      

      # Test plot, should show much higher density in 75-90 degree bins over
      # 0-15 degree bin
      a_n_bins = 9
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      r_n_bins = 15
      r_bins = np.linspace(150, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)
      r_bins_plot = np.linspace(0, 150, r_n_bins)

      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins, False)
      ion = finish_plot(field, COS_data, fn_head)
      angle = 90/a_n_bins
      i=0
      plot_profile(r_bins_plot, radial_data[i], '%s < Φ < %s degrees' % \
                    (angle*i, angle*i + angle), colors[0])
      i=8
      plot_profile(r_bins_plot, radial_data[i], '%s < Φ < %s degrees' % \
                    (angle*i, angle*i + angle), colors[3])
      fplot_radius(ion, 'test')


      # Make plot similar to paper of phi vs N
      a_n_bins = 3
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      r_n_bins = 3
      r_bins = np.linspace(80, 20, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)

      cden_data, angle_data = big_profile(a_arr, a_bins, a_n_bins, cdens_arr, r_arr, r_bins, r_n_bins)
      ion = finish_plot(field, COS_data, fn_head)
      plot_big_angle(angle_data[0], cden_data[0], 'b < 40 kpc', colors[0], markers[0])
      for i in range(1,3):
        plot_big_angle(angle_data[i], cden_data[i], '%s < b < %s kpc' % \
                    (i*20 + 20, i*20 + 40), colors[3*i], markers[i])
      fplot_angle(ion, 'big')

      # Makes plot similar to paper of b vs N
      a_n_bins = 3
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      r_n_bins = 4
      r_bins = np.linspace(80, 20, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)
      
      cden_data, radius_data = big_profile(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
      ion = finish_plot(field, COS_data, fn_head)
      plot_big_radius(radius_data[0], cden_data[0], '0 < Φ < 30 degrees', colors[0], markers[0])
      plot_big_radius(radius_data[1], cden_data[1], '30 < Φ < 60 degrees', colors[3], markers[1])
      plot_big_radius(radius_data[2], cden_data[2], '60 < Φ < 90 degrees', colors[6], markers[2])
      fplot_radius(ion, 'big')


