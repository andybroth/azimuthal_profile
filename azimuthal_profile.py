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
yt.enable_parallelism()
import numpy as np
import trident
import h5py as h5
import sys
import glob
import os.path
from yt.units.yt_array import \
    YTArray, \
    YTQuantity
import cmocean
from scipy.signal import filtfilt, gaussian
from scipy.ndimage import filters
import ytree

from matplotlib.colors import LogNorm
sys.path.insert(0, '/home/andyr/src/frb')
from get_COS_data import get_COS_data, plot_COS_data
from radial_profile2 import plot_profile, finish_plot
from grid_figure import GridFigure

def make_profiles2(a_arr, a_bins, a_n_bins, cdens_arr, r_arr, r_bins, r_n_bins):
  '''
  Default is angle vs N plot w/ 3 radial bins in legend. normalize should be set
  to true for this to work. To make radius vs N plots, flip all angular and radius
  inputs around cdens_arr and set normalize to false.
  '''
  r_bin_ids = np.digitize(r_arr, r_bins)
  a_bin_ids = np.digitize(a_arr, a_bins)

  profile_data = [np.zeros([3, a_n_bins]) for _ in range(r_n_bins)]
  for a_bin_id in range(a_n_bins):
    for r_bin_id in range(r_n_bins):
      ids = np.logical_and(r_bin_ids == r_bin_id, a_bin_ids == a_bin_id)
      sample = cdens_arr[ids]
      profile_data[r_bin_id][0,a_bin_id] = np.median(sample)
      profile_data[r_bin_id][1,a_bin_id] = np.percentile(sample, 25)
      profile_data[r_bin_id][2,a_bin_id] = np.percentile(sample, 75)
  return profile_data

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
      # cden_data[r_bin_id][1,a_bin_id] = np.std(sample)
      angle_data[r_bin_id][0,a_bin_id] = np.median(a_arr[ids])
      # angle_data[r_bin_id][1,a_bin_id] = np.std(a_arr[ids])
  return cden_data, angle_data

def fplot_angle(ion, description, fn, field, ax):
  ax.title('%s' % ion)
  ax.xlabel('Azimuthal Angle [degrees]')
  ax.xlim((0,90))
  if limits_from_field(field):
    ax.ylim(limits_from_field(field))
  ax.legend(title='b/rvir')
  # if len(description) > 0:
  #   print('%s_%s_angle.png' % (ion, description))
  #   # ax.savefig('%s/plots/%s_%s_angle.png' % (fn, ion, description))
  # else:
  #   print('%s_angle.png' % ion)
  #   # ax.savefig('%s/plots/%s_angle.png' % (fn, ion))
  # # plt.clf()

def fplot_radius(ion, description, fn, field, ax):
  ax.title('%s' % ion)
  ax.xlabel('b/rvir')
  ax.xlim((0,1))
  if limits_from_field(field):
    ax.ylim(limits_from_field(field))
  ax.legend(title='Azimuthal Angle [degrees]')
  # if len(description) > 0:
  #   print('%s_%s_radius.png' % (ion, description))
  #   ax.savefig('%s/plots/%s_%s_radius.png' % (fn, ion, description))
  # else:
  #   print('%s_radial.png' % ion)
  # #   ax.savefig('%s/plots/%s_radial.png' % (fn, ion))
  # # ax.clf()

def limits_from_field(field):
  '''Give limits for the plots depending on the ion field'''
  if field == 'H_number_density':
    return (1e13, 1e20)
  elif field == 'O_p5_number_density':
    return (1e15, 1e19)
  elif field == 'Mg_p1_number_density':
    return (1e6, 1e18)
  elif field == 'density':
    return (1e-5, 1e-1)
  elif field == 'temperature':
    return (1e4, 1e7)
  elif field == 'C_p1_number_density':
    return (1e-12, 1e-3)
  elif field == 'C_p2_number_density':
    return (1e-12, 1e-4) 
  elif field == 'Ne_p7_number_density':
    return (1e-10, 1e-6)
  elif field == 'Si_p3_number_density':
    return (1e-14, 1e-6)
  elif field == 'metal_density':
    return (1e-8, 1e-5)
  else:
    return None

def read_parameter_file(fn):
    """
    read the parameter file and return a dictionary with pairs of:

    label : [fn1, fn2, fn3]
    """
    f = open(fn, 'r')
    profiles = {}
    text = f.readlines()
    i = 0
    while i < len(text):
        profile_label = text[i].split()[0][:-1]
        i += 1
        profile_list = []
        while not text[i].split() == []: 
            file_name = text[i].split()[0]
            file = h5.File(file_name,'a')
            if file.attrs.get('Ang_Mom') > 3*10**29:
              if '_1_' in file_name:
                print(file_name)
                print(file.attrs.get('Ang_Mom'))
              profile_list.append(file_name)
            file.close()
            i += 1
        profiles[profile_label] = profile_list
        i += 1
    return profiles

''''
Things to fix to make correct images:
- plot_profile()
    Make it use ax instead of plt

- Fix titles/labels
    Won't be sure how it all works until I run it for the first time
'''

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

  fn_head = sys.argv[1].split('.')[0][:-11]
  profiles_dict = read_parameter_file(sys.argv[1])

  # Get the list of ion_fields from the first file available
  fn = list(profiles_dict.values())[0][0]
  f = h5.File(fn, 'r')
  ion_fields = list(f.keys())
  ion_fields.remove('radius')
  ion_fields.remove('phi')
  f.close()

  fig = GridFigure(2, 5)

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
        cdens_arr = np.concatenate((cdens_arr, f["%s/%s" % (field, 'edge')].value))
        new_r = f['radius'].value / f.attrs.get('rvir')
        r_arr = np.concatenate((r_arr, new_r))
      if field == 'Mg_p1_number_density' or field == 'O_p5_number_density':
        cdens_arr *= 6.02 * 10**23

      # create bins for data
      a_n_bins = 9
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      r_n_bins = 4
      r_bins = np.linspace(1, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)

      ax = fig[c]
      profile_data = make_profiles2(a_arr, a_bins, a_n_bins, cdens_arr, r_arr, r_bins, r_n_bins)
      for i in range(r_n_bins):
        plot_profile(np.linspace(0, 90, a_n_bins), profile_data[i], '%s < b/rvir < %s' % \
                    (.25*i, .25*i + .25), colors[i], ax)
      ion = finish_plot(field, COS_data, fn_head)
      fplot_angle(ion, '', fn_head, field, ax)

  plt.savefig('fig1.png')

  fig = GridFigure(2, 5)

  for field in ion_fields:
    for c, (k,v) in enumerate(profiles_dict.items()):
      n_files = len(v)
      cdens_arr = np.array([])
      a_arr = np.array([])
      r_arr = np.array([])
      for j in range(n_files):
        f = h5.File(v[j], 'r')
        a_arr = np.concatenate((a_arr, f['phi'].value))
        cdens_arr = np.concatenate((cdens_arr, f["%s/%s" % (field, 'edge')].value))
        new_r = f['radius'].value / f.attrs.get('rvir')
        r_arr = np.concatenate((r_arr, new_r))
      if field == 'Mg_p1_number_density' or field == 'O_p5_number_density':
        cdens_arr *= 6.02 * 10**23

      # redefine bins
      a_n_bins = 2
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      r_n_bins = 20
      r_bins = np.linspace(1, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)
      r_bins_plot = np.linspace(0, 1, r_n_bins)

      # Step through each ion and make plots of radius vs N for 3 radial bins
      ax = fig[c]
      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
      ion = finish_plot(field, COS_data, fn_head)
      plot_profile(r_bins_plot, radial_data[0], 'Φ < 45 degrees', colors[0], ax)
      plot_profile(r_bins_plot, radial_data[1], 'Φ > 45 degrees', colors[6], ax)
      fplot_radius(ion, 'short', fn_head, field, ax)

  plt.savefig('fig2.png')
  
  fig = GridFigure(2, 5)

  for field in ion_fields:
    for c, (k,v) in enumerate(profiles_dict.items()):
      n_files = len(v)
      cdens_arr = np.array([])
      a_arr = np.array([])
      r_arr = np.array([])
      for j in range(n_files):
        f = h5.File(v[j], 'r')
        a_arr = np.concatenate((a_arr, f['phi'].value))
        cdens_arr = np.concatenate((cdens_arr, f["%s/%s" % (field, 'edge')].value))
        new_r = f['radius'].value / f.attrs.get('rvir')
        r_arr = np.concatenate((r_arr, new_r))
      if field == 'Mg_p1_number_density' or field == 'O_p5_number_density':
        cdens_arr *= 6.02 * 10**23

      # makes other plot
      a_n_bins = 3
      a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
      a_bins = np.flip(a_bins, 0)

      r_n_bins = 20
      r_bins = np.linspace(1, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)
      r_bins_plot = np.linspace(0, 1, r_n_bins)

      # Step through each ion and make plots of radius vs N for 3 radial bins
      ax = fig[c]
      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
      ion = finish_plot(field, COS_data, fn_head)
      plot_profile(r_bins_plot, radial_data[0], '0 < Φ < 30 degrees', colors[0], ax)
      plot_profile(r_bins_plot, radial_data[1], '30 < Φ < 60 degrees', colors[1], ax)
      plot_profile(r_bins_plot, radial_data[2], '60 < Φ < 90 degrees', colors[2], ax)
      fplot_radius(ion, '2', fn_head, field, ax)

  plt.savefig('fig3.png')

  for field in ion_fields:
    for c, (k,v) in enumerate(profiles_dict.items()):
      n_files = len(v)
      cdens_arr = np.array([])
      a_arr = np.array([])
      r_arr = np.array([])
      for j in range(n_files):
        f = h5.File(v[j], 'r')
        a_arr = np.concatenate((a_arr, f['phi'].value))
        cdens_arr = np.concatenate((cdens_arr, f["%s/%s" % (field, 'edge')].value))
        new_r = f['radius'].value / f.attrs.get('rvir')
        r_arr = np.concatenate((r_arr, new_r))
      if field == 'Mg_p1_number_density' or field == 'O_p5_number_density':
        cdens_arr *= 6.02 * 10**23

      # Test plot, should show much higher density in 75-90 degree bins over
      # 0-15 degree bin
      if field == 'density':
        a_n_bins = 9
        a_bins = np.linspace(90, 0, a_n_bins, endpoint=False)
        a_bins = np.flip(a_bins, 0)

        r_n_bins = 20
        r_bins = np.linspace(1, 0, r_n_bins, endpoint=False)
        r_bins = np.flip(r_bins, 0)
        r_bins_plot = np.linspace(0, 1, r_n_bins)

        radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
        ion = finish_plot(field, COS_data, fn_head)
        angle = 90/a_n_bins
        i=0
        plot_profile(r_bins_plot, radial_data[i], '%s < Φ < %s degrees' % \
                      (angle*i, angle*i + angle), colors[0], plt)
        i=8
        plot_profile(r_bins_plot, radial_data[i], '%s < Φ < %s degrees' % \
                      (angle*i, angle*i + angle), colors[3], plt)
        fplot_radius(ion, 'test', fn_head, field, plt)
        plt.savefig('test_plot.png')
        plt.close()


