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
  # ax.title('%s' % ion)
  ax.set_xlabel('Azimuthal Angle [degrees]')
  ax.set_xlim((0,90))
  # ax.legend(title='b/rvir')
  
  # if len(description) > 0:
  #   print('%s_%s_angle.png' % (ion, description))
  #   # ax.savefig('%s/plots/%s_%s_angle.png' % (fn, ion, description))
  # else:
  #   print('%s_angle.png' % ion)
  #   # ax.savefig('%s/plots/%s_angle.png' % (fn, ion))
  # # plt.clf()

def fplot_radius(ion, description, fn, field, ax):
  # ax.title('%s' % ion)
  ax.set_xlabel('b/rvir')
  if description == 'long' or description == 'long2':
    ax.set_xlim((0,1.25))
  else:
    ax.set_xlim((0,1))
  # ax.legend(title='Azimuthal Angle [degrees]')
  
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

def plot_profile(r_bins, profile_data, label, color, ax, linestyle='dashed'):
  '''
  Takes data from profiles and creates plot that is a part of the larger
  plot image
  '''
  ax.semilogy(r_bins, profile_data[0,:], label=label, color=color, linestyle=linestyle, linewidth=4)  
  ax.fill_between(r_bins, profile_data[1,:], profile_data[2,:], facecolor=color, alpha=0.2)

def finish_plot(field, ax):
  if limits_from_field(field):
    ax.set_ylim(limits_from_field(field))
  ylabel = None
  if field == 'H_number_density':
    ion = 'H I'
  elif field == 'Mg_p1_number_density':
    ion = 'Mg II'
  elif field == 'Si_p1_number_density':
    ion = 'Si II'
  elif field == 'Si_p2_number_density':
    ion = 'Si III'
  elif field == 'Si_p3_number_density':
    ion = 'Si IV'
  elif field == 'N_p1_number_density':
    ion = 'N II'
  elif field == 'N_p2_number_density':
    ion = 'N III'
  elif field == 'N_p4_number_density':
    ion = 'N V'
  elif field == 'C_p1_number_density':
    ion = 'C II'
  elif field == 'C_p2_number_density':
    ion = 'C III'
  elif field == 'Ne_p7_number_density':
    ion = 'Ne VIII'
  elif field == 'O_p5_number_density':
    ion = 'O VI'
  elif field == 'density':
    ion = 'density'
    ylabel = "Projected Density"
  elif field == 'temperature':
    ion = 'temperature'
    ylabel = ion
  elif field == 'metal_density':
    ion = 'metal_density'
    ylabel = "Projected Metal Density"
  elif field == 'O_nuclei_density':
    ion = 'oxygen'
    ylabel = "Projected Oxygen Density"
  else:
    sys.exit('Unidentified Field.')
  if not ylabel:
    ylabel = "%s Column Density [cm$^{-2}$]" % ion
  ax.set_ylabel(ylabel)

''''
Things to fix to make correct images:
- Fix titles/labels
    Won't be sure how it all works until I run it for the first time
'''

if __name__ == '__main__':
  """
  Takes file generated by azimuthal_projections.py and creates column density
  plots.
  """

  threshold = {'H_number_density' : 10**16, 'O_p5_number_density':1e14, 'density':1e-4}

  # Color cycling
  colors = ['red', 'blue', 'cyan', 'green', 'yellow', 'black']
  linestyles = ['-', '--', ':', '-.']

  fn_head = sys.argv[1].split('.')[0][:-11]
  profiles_dict = read_parameter_file(sys.argv[1])

  # Get the list of ion_fields from the first file available
  fn = list(profiles_dict.values())[0][0]

  ion_fields = ['density', 'metal_density', 'temperature', 'H_number_density']

  fig = GridFigure(2, 2, top_buffer=0.01, bottom_buffer=0.08, left_buffer=0.12, 
    right_buffer=0.02, vertical_buffer=0.08, horizontal_buffer=0.08, figsize=(12,8))

  # Step through each ion and make plots of azimuthal angle vs N
  for i, field in enumerate(ion_fields):
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

      ax = fig[i]
      profile_data = make_profiles2(a_arr, a_bins, a_n_bins, cdens_arr, r_arr, r_bins, r_n_bins)
      for i in range(r_n_bins):
        plot_profile(np.linspace(0, 90, a_n_bins), profile_data[i], '%s < b/rvir < %s' % \
                    (.25*i, .25*i + .25), colors[i], ax, linestyles[i])
      ion = finish_plot(field, ax)
      fplot_angle(ion, '', fn_head, field, ax)

  plt.savefig('%s/fig1a.png' % fn_head)

  fig = GridFigure(2, 2, top_buffer=0.01, bottom_buffer=0.08, left_buffer=0.12, 
    right_buffer=0.02, vertical_buffer=0.08, horizontal_buffer=0.08, figsize=(12,8))

  for i, field in enumerate(ion_fields):
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
      ax = fig[i]
      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
      ion = finish_plot(field, ax)
      plot_profile(r_bins_plot, radial_data[0], 'Φ < 45 degrees', colors[0], ax, linestyles[0])
      plot_profile(r_bins_plot, radial_data[1], 'Φ > 45 degrees', colors[1], ax, linestyles[1])
      fplot_radius(ion, 'short', fn_head, field, ax)

  plt.savefig('%s/fig1b.png' % fn_head)
  
  fig = GridFigure(2, 2, top_buffer=0.01, bottom_buffer=0.08, left_buffer=0.12, 
    right_buffer=0.02, vertical_buffer=0.08, horizontal_buffer=0.08, figsize=(12,8))

  for i, field in enumerate(ion_fields):
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
      ax = fig[i]
      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
      ion = finish_plot(field, ax)
      plot_profile(r_bins_plot, radial_data[0], '0 < Φ < 30 degrees', colors[0], ax, linestyles[0])
      plot_profile(r_bins_plot, radial_data[1], '30 < Φ < 60 degrees', colors[1], ax, linestyles[1])
      plot_profile(r_bins_plot, radial_data[2], '60 < Φ < 90 degrees', colors[2], ax, linestyles[2])
      fplot_radius(ion, '2', fn_head, field, ax)

  plt.savefig('%s/fig1c.png' % fn_head)

  ion_fields = ['C_p1_number_density', 'C_p2_number_density', 'Ne_p7_number_density', 'O_p5_number_density', 'Si_p3_number_density']

  fig = GridFigure(1, 5, top_buffer=0.04, bottom_buffer=0.08, left_buffer=0.12, 
    right_buffer=0.02, vertical_buffer=0.04, horizontal_buffer=0.04, figsize=(20,5))

  # Step through each ion and make plots of azimuthal angle vs N
  for i, field in enumerate(ion_fields):
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

      ax = fig[i]
      profile_data = make_profiles2(a_arr, a_bins, a_n_bins, cdens_arr, r_arr, r_bins, r_n_bins)
      for i in range(r_n_bins):
        plot_profile(np.linspace(0, 90, a_n_bins), profile_data[i], '%s < b/rvir < %s' % \
                    (.25*i, .25*i + .25), colors[i], ax, linestyles[i])
      ion = finish_plot(field, ax)
      fplot_angle(ion, '', fn_head, field, ax)

  plt.savefig('%s/fig2a.png' % fn_head)

  fig = GridFigure(1, 5, top_buffer=0.04, bottom_buffer=0.12, left_buffer=0.12, 
    right_buffer=0.02, vertical_buffer=0.04, horizontal_buffer=0.04, figsize=(20,5))

  for i, field in enumerate(ion_fields):
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

      r_n_bins = 25
      r_bins = np.linspace(1.25, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)
      r_bins_plot = np.linspace(0, 1.25, r_n_bins)

      # Step through each ion and make plots of radius vs N for 3 radial bins
      ax = fig[i]
      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
      ion = finish_plot(field, ax)
      plot_profile(r_bins_plot, radial_data[0], 'Φ < 45 degrees', colors[0], ax, linestyles[0])
      plot_profile(r_bins_plot, radial_data[1], 'Φ > 45 degrees', colors[1], ax, linestyles[1])
      fplot_radius(ion, 'long', fn_head, field, ax)

  plt.savefig('%s/fig2b.png' % fn_head)
  
  fig = GridFigure(1, 5, top_buffer=0.04, bottom_buffer=0.12, left_buffer=0.12, 
    right_buffer=0.02, vertical_buffer=0.04, horizontal_buffer=0.04, figsize=(20,5))

  for i, field in enumerate(ion_fields):
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

      r_n_bins = 25
      r_bins = np.linspace(1.25, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)
      r_bins_plot = np.linspace(0, 1.25, r_n_bins)

      # Step through each ion and make plots of radius vs N for 3 radial bins
      ax = fig[i]
      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
      ion = finish_plot(field, ax)
      plot_profile(r_bins_plot, radial_data[0], '0 < Φ < 30 degrees', colors[0], ax, linestyles[0])
      plot_profile(r_bins_plot, radial_data[1], '30 < Φ < 60 degrees', colors[1], ax, linestyles[1])
      plot_profile(r_bins_plot, radial_data[2], '60 < Φ < 90 degrees', colors[2], ax, linestyles[2])
      fplot_radius(ion, 'long2', fn_head, field, ax)

  plt.savefig('%s/fig2c.png' % fn_head)

  ion_fields = ['Mg_p1_number_density']

  fig = GridFigure(1, 1, top_buffer=0.01, bottom_buffer=0.12, left_buffer=0.12, 
    right_buffer=0.02, vertical_buffer=0.04, horizontal_buffer=0.04, figsize=(6,6))

  # Step through each ion and make plots of azimuthal angle vs N
  for i, field in enumerate(ion_fields):
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

      ax = fig[i]
      profile_data = make_profiles2(a_arr, a_bins, a_n_bins, cdens_arr, r_arr, r_bins, r_n_bins)
      for i in range(r_n_bins):
        plot_profile(np.linspace(0, 90, a_n_bins), profile_data[i], '%s < b/rvir < %s' % \
                    (.25*i, .25*i + .25), colors[i], ax, linestyles[i])
      ion = finish_plot(field, ax)
      fplot_angle(ion, '', fn_head, field, ax)

  plt.savefig('%s/fig3a.png' % fn_head)

  fig = GridFigure(1, 1, top_buffer=0.01, bottom_buffer=0.08, left_buffer=0.12, 
    right_buffer=0.02, vertical_buffer=0.04, horizontal_buffer=0.04, figsize=(6,6))

  for i, field in enumerate(ion_fields):
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
      ax = fig[i]
      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
      ion = finish_plot(field, ax)
      plot_profile(r_bins_plot, radial_data[0], 'Φ < 45 degrees', colors[0], ax, linestyles[0])
      plot_profile(r_bins_plot, radial_data[1], 'Φ > 45 degrees', colors[1], ax, linestyles[1])
      fplot_radius(ion, 'short', fn_head, field, ax)

  plt.savefig('%s/fig3b.png' % fn_head)
  
  fig = GridFigure(1, 1, top_buffer=0.01, bottom_buffer=0.08, left_buffer=0.12, 
    right_buffer=0.02, vertical_buffer=0.04, horizontal_buffer=0.04, figsize=(6,6))

  for i, field in enumerate(ion_fields):
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

      r_n_bins = 25
      r_bins = np.linspace(1.25, 0, r_n_bins, endpoint=False)
      r_bins = np.flip(r_bins, 0)
      r_bins_plot = np.linspace(0, 1, r_n_bins)

      # Step through each ion and make plots of radius vs N for 3 radial bins
      ax = fig[i]
      radial_data = make_profiles2(r_arr, r_bins, r_n_bins, cdens_arr, a_arr, a_bins, a_n_bins)
      ion = finish_plot(field, ax)
      plot_profile(r_bins_plot, radial_data[0], '0 < Φ < 30 degrees', colors[0], ax, linestyles[0])
      plot_profile(r_bins_plot, radial_data[1], '30 < Φ < 60 degrees', colors[1], ax, linestyles[1])
      plot_profile(r_bins_plot, radial_data[2], '60 < Φ < 90 degrees', colors[2], ax, linestyles[2])
      fplot_radius(ion, 'long2', fn_head, field, ax)

  plt.savefig('%s/fig3c.png' % fn_head)


  '''
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
        ion = finish_plot(field, plt)
        angle = 90/a_n_bins
        i=0
        plot_profile(r_bins_plot, radial_data[i], '%s < Φ < %s degrees' % \
                      (angle*i, angle*i + angle), colors[0], plt)
        i=8
        plot_profile(r_bins_plot, radial_data[i], '%s < Φ < %s degrees' % \
                      (angle*i, angle*i + angle), colors[3], plt)
        fplot_radius(ion, 'test', fn_head, field, plt)
        plt.savefig('%s/test_plot.png' % fn_head)
        plt.close()
    '''

