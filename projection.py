import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import yt
import trident
from yt.units.yt_array import YTQuantity
from yt.units import cm
import numpy as np
from grid_figure import GridFigure
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from yt.utilities.math_utils import ortho_find

import sys
sys.path.insert(0, '/home/andyr/src/frb')
from radial_profile1 import get_amiga_data, smooth_amiga, GizmoDataset, read_amiga_center, read_amiga_rvir, log

if __name__ == '__main__':
	'''
	Makes projections from m12i_res7100_md for density, H I, Mg II, and O VI
	in a grid figure plot for the paper. 
	'''
	ions = ['H I', 'Mg II', 'O VI', 'Si IV']

	fn = 'm11g_res12000'

	amiga_data = get_amiga_data('/mnt/data1/GalaxiesOnFire/%s/halo/ahf/halo_00000_smooth.dat' % fn)
	amiga_data = smooth_amiga(amiga_data)

	fn = '/mnt/data1/GalaxiesOnFire/%s/output/snapdir_600/snapshot_600.0.hdf5' % fn

	log("Starting projections for %s" % fn)
	ds = GizmoDataset(fn)
	trident.add_ion_fields(ds, ions=ions, ftype='gas')

	radial_extent = ds.quan(250, 'kpc')
	width = 2*radial_extent

	log("Reading amiga center for halo in %s" % fn)
	c = read_amiga_center(amiga_data, fn, ds)
	rvir = read_amiga_rvir(amiga_data, fn, ds)

	log('Finding Angular Momentum of Galaxy')
	sp = ds.sphere(c, (15, 'kpc'))
	L = sp.quantities.angular_momentum_vector(use_gas=False, use_particles=True, particle_type='PartType0')
	L, E1, E2 = ortho_find(L)

	one = ds.arr([.5, .5, .5], 'Mpc')
	box = ds.box(c-one, c+one)

	log('Generating Plot 1')
	p1 = yt.OffAxisProjectionPlot(ds, E1, ('gas', 'H_number_density'), center=c, 
		width=width, data_source=box, north_vector=L, weight_field=None)
	log('Generating Plot 2')
	p2 = yt.OffAxisProjectionPlot(ds, E1, ('gas', 'Mg_p1_number_density'), center=c, 
		width=width, data_source=box, north_vector=L, weight_field=None)
	log('Generating Plot 3')
	p3 = yt.OffAxisProjectionPlot(ds, E1, ('gas', 'O_p5_number_density'), center=c, 
		width=width, data_source=box, north_vector=L, weight_field=None)
	log('Generating Plot 4')
	p4 = yt.OffAxisProjectionPlot(ds, E1, ('gas', 'Si_p3_number_density'), center=c,
		width=width, data_source=box, north_vector=L, weight_field=None)

	print("Stitching together")
	fig = GridFigure(2, 2, top_buffer=0.01, bottom_buffer=0.01, left_buffer=0.01, right_buffer=0.13, vertical_buffer=0.01, horizontal_buffer=0.01, figsize=(9,8))

	# Actually plot in the different axes
	plot1 = fig[0].imshow(p1.frb['H_number_density'], norm=LogNorm())
	# clim1 = plot1.get_clim()
	plot1.set_cmap('thermal')
	plot2 = fig[1].imshow(p2.frb['Mg_p1_number_density'], norm=LogNorm())
	# plot2.set_clim(clim)
	plot2.set_cmap('thermal')
	plot3 = fig[2].imshow(p3.frb['O_p5_number_density'], norm=LogNorm())
	# clim = plot3.get_clim()
	plot3.set_cmap('thermal')
	plot4 = fig[3].imshow(p4.frb['Si_p3_number_density'], norm=LogNorm())
	# plot4.set_clim(clim)
	plot4.set_cmap('thermal')

	for i in range(len(fig)):

		# Set all of the axis labels to be invisible
		ax = fig[i]
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)

		# Add virial radius circles to each plot
		cir = Circle((0.5, 0.5), rvir/width, fill=False, color='white', linestyle='dashed', linewidth=5, transform=ax.transAxes, alpha=0.5)
		ax.add_patch(cir)
	
	row0_cax = fig.add_cax(fig[0], 'right', buffer=0.01, length=1, width=0.05)
	row0_cbar = plt.colorbar(plot2, cax=row0_cax, orientation='vertical')
	row0_cbar.set_label('H I Column Density [cm$^{-2}$]', weight='bold')
	row0_cbar.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
	row0_cbar.ax.tick_params(labelsize=14)

	row1_cax = fig.add_cax(fig[1], 'right', buffer=0.01, length=1, width=0.05)
	row1_cbar = plt.colorbar(plot2, cax=row1_cax, orientation='vertical')
	row1_cbar.set_label('Mg II Column Density [cm$^{-2}$]', weight='bold')
	row1_cbar.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
	row1_cbar.ax.tick_params(labelsize=14)

	row3_cax = fig.add_cax(fig[2], 'right', buffer=0.01, length=1, width=0.05)
	row3_cbar = plt.colorbar(plot2, cax=row3_cax, orientation='vertical')
	row3_cbar.set_label('O VI Column Density [cm$^{-2}$]', weight='bold')
	row3_cbar.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
	row3_cbar.ax.tick_params(labelsize=14)

	row2_cax = fig.add_cax(fig[3], 'right', buffer=0.01, length=1, width=0.05)
	row2_cbar = plt.colorbar(plot4, cax=row2_cax, orientation='vertical')
	row2_cbar.set_label('Si II Column Density [cm$^{-2}$]', weight='bold')
	row2_cbar.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
	row2_cbar.ax.tick_params(labelsize=14)
	
	plt.savefig('projections.png')
