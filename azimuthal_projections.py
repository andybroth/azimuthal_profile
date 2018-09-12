"""
Creates Edge on Projections of FIRE Simulations with selected ion fields

Input:
python azimuthal_profile.py filelist.txt /halo/ahf/halo_00000_smooth.dat

filelist.txt has a list of hdf5 data files where each line is the path to a 
different file. 
"""
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
from yt.frontends.gizmo.api import GizmoDataset
import cmocean
from scipy.signal import filtfilt, gaussian
from scipy.ndimage import filters
import ytree
sys.path.insert(0, '/home/andyr/src/frb')
from yt.utilities.math_utils import ortho_find
from radial_profile1 import set_image_details, get_amiga_data, smooth_amiga, GizmoDataset, read_amiga_center, read_amiga_rvir, log

'''
def find_angular_momentum(sp, c):
    """
    Finds the average angular momentum of a sphere of particles around a 
    center point c
    """

    pt = 'PartType0'
    halo_vel = sp.quantities.bulk_velocity()

    r_x = sp[(pt, 'x')] - c[0]
    r_y = sp[(pt, 'y')] - c[1]
    r_z = sp[(pt, 'z')] - c[2]

    v_x = sp[(pt, 'velocity_x')] - halo_vel[0]
    v_y = sp[(pt, 'velocity_y')] - halo_vel[1]
    v_z = sp[(pt, 'velocity_z')] - halo_vel[2]

    # cross product to find angular momentum
    x_ang_mom = np.sum(sp[(pt, 'mass')]*(r_y*v_z - r_z*v_y))
    y_ang_mom = np.sum(sp[(pt, 'mass')]*(r_z*v_x - r_x*v_z))
    z_ang_mom = np.sum(sp[(pt, 'mass')]*(r_x*v_y - r_y*v_x))
    ang_mom = yt.YTArray([x_ang_mom, y_ang_mom, z_ang_mom])
    ang_mom, b1, b2 = ortho_find(ang_mom)
    return ang_mom, b1, b2
'''

def make_off_axis_projection(ds, vec, north_vec, ion_fields, center, width, data_source, radius, fn_data, weight_field=None, dir=None):
	"""
	Use OffAxisProjectionPlot to make projection (cannot specify resolution)
	"""
	p = yt.OffAxisProjectionPlot(ds, vec, ion_fields, center=center, width=width,
		data_source=data_source, north_vector=north_vec, weight_field=weight_field)
	p.hide_axes()
	p.annotate_scale()
	p.annotate_timestamp(redshift=True)
	r = radius.in_units('kpc')
	p.annotate_sphere(center, (r, 'kpc'), circle_args={'color':'white', 'alpha':0.5, 'linestyle':'dashed', 'linewidth':5})
	for field in ion_fields:
		p.set_cmap(field, 'dusk')
		set_image_details(p, field, True)
		p.set_background_color(field)
	if dir is None:
		dir = 'face/'
	p.save(os.path.join('%s/' % fn_data, dir))
	return p.frb

def rotate(theta, E1, E2):
	'''
	Rotates vector E1 an angle theta around E2
	'''
	theta *= (np.pi/180)
	x = E2[0]
	y = E2[1]
	z = E2[2]
	R = np.zeros((3,3))
	R[0][0] = np.cos(theta) + x**2 * (1-np.cos(theta))
	R[0][1] = x * y * (1 - np.cos(theta)) - z * np.sin(theta)
	R[0][2] = x * z * (1 - np.cos(theta)) + y * np.sin(theta)
	R[1][0] = y * x * (1 - np.cos(theta)) + z * np.sin(theta)
	R[1][1] = np.cos(theta) + y**2 * (1 - np.cos(theta))
	R[1][2] = y * z * (1 - np.cos(theta)) - x * np.sin(theta)
	R[2][0] = z * x * (1 - np.cos(theta)) - y * np.sin(theta)
	R[2][1] = z * y * (1 - np.cos(theta)) + x * np.sin(theta)
	R[2][2] = np.cos(theta) + z**2 * (1 - np.cos(theta))
	return (R*E1)[[0,1,2],[0,0,0]]

if __name__ == '__main__':
	"""
	Reads Amiga Data, picks desired ion fields, creates edge on projections, and
	outputs data in hdf5 files for later manipulation.
	"""

	# Variables to set for each run
	
	res = 800

	# Loading datasets
	fn_list = open(sys.argv[1], 'r')
	fns = fn_list.readlines()

	# Import FIRE Data
	# Read in halo information from amiga output if present
	amiga_data = get_amiga_data(sys.argv[2])
	# Smooth the data to remove jumps in centroid
	amiga_data = smooth_amiga(amiga_data)

	dir1 = '1/'
	dir2 = '2/'

	for fn in yt.parallel_objects(fns):
		fn = fn.strip() # Get rid of trailing \n
		fn_head = fn.split('/')[-1]
		fn_data = fn.split('/')[-4]
		cdens_fn_1 = "%s/%s_1_cdens.h5" % (fn_data, fn_head)
		cdens_fn_2 = "%s/%s_2_cdens.h5" % (fn_data, fn_head)

		# Define ions we care about
		ions = []
		ion_fields = []
		full_ion_fields = []
		ions.append('H I')
		ion_fields.append('H_number_density')
		full_ion_fields.append(('gas', 'H_number_density'))
		# ions.append('Mg II')
		# ion_fields.append('Mg_p1_number_density')
		# full_ion_fields.append(('gas', 'Mg_p1_number_density'))
		# ions.append('Si II')
		# ion_fields.append('Si_p1_number_density')
		# full_ion_fields.append(('gas', 'Si_p1_number_density'))
		# ions.append('Si III')
		# ion_fields.append('Si_p2_number_density')
		# full_ion_fields.append(('gas', 'Si_p2_number_density'))
		# ions.append('Si IV')
		# ion_fields.append('Si_p3_number_density')
		# full_ion_fields.append(('gas', 'Si_p3_number_density'))
		# ions.append('N II')
		# ion_fields.append('N_p1_number_density')
		# full_ion_fields.append(('gas', 'N_p1_number_density'))
		# ions.append('N III')
		# ion_fields.append('N_p2_number_density')
		# full_ion_fields.append(('gas', 'N_p2_number_density'))
		# ions.append('N V')
		# ion_fields.append('N_p4_number_density')
		# full_ion_fields.append(('gas', 'N_p4_number_density'))
		# ions.append('C II')
		# ion_fields.append('C_p1_number_density')
		# full_ion_fields.append(('gas', 'C_p1_number_density'))
		# ions.append('C III')
		# ion_fields.append('C_p2_number_density')
		# full_ion_fields.append(('gas', 'C_p2_number_density'))
		# ions.append('Ne VIII')
		# ion_fields.append('Ne_p7_number_density')
		# full_ion_fields.append(('gas', 'Ne_p7_number_density'))
		# ions.append('O VI')
		# ion_fields.append('O_p5_number_density')
		# full_ion_fields.append(('gas', 'O_p5_number_density'))

		others = []
		other_fields = []
		full_other_fields = []
		# others.append('Temperature')
		# other_fields.append('temperature')
		# full_other_fields.append(('gas', 'temperature'))

		log("Starting projections for %s" % fn)
		ds = GizmoDataset(fn)
		trident.add_ion_fields(ds, ions=ions, ftype='gas')

		radial_extent = ds.quan(250, 'kpc')
		width = 2*radial_extent

		# ions.append('O_nuclei_density')
		# ion_fields.append('O_nuclei_density')
		# full_ion_fields.append(('gas', 'O_nuclei_density'))
		ions.append('density')
		ion_fields.append('density')
		full_ion_fields.append(('gas', 'density'))
		# ions.append('metal_density')
		# ion_fields.append('metal_density')
		# full_ion_fields.append(('gas', 'metal_density'))

		# Figure out centroid and r_vir info
		log("Reading amiga center for halo in %s" % fn)
		c = read_amiga_center(amiga_data, fn, ds)
		rvir = read_amiga_rvir(amiga_data, fn, ds)

		# Create box around galaxy so we're only sampling galaxy out to 1 Mpc
		one = ds.arr([.5, .5, .5], 'Mpc')
		box = ds.box(c-one, c+one)

		log('Finding Angular Momentum of Galaxy')
		sp = ds.sphere(c, (15, 'kpc'))
		L = sp.quantities.angular_momentum_vector(use_gas=False, use_particles=True, particle_type='PartType0')
		L, v1, v2 = ortho_find(L)
		inclines = [-20, -10, 10, 20]
		dirs = ['20n/', '10n/', '10/', '20/']

		for i, angle in enumerate(inclines):
			cdens_fn_1 = "%s/%s/%s_1_cdens.h5" % (fn_data, dirs[i], fn_head)
			cdens_fn_2 = "%s/%s/%s_2_cdens.h5" % (fn_data, dirs[i], fn_head)
			cdens_file_1 = h5.File(cdens_fn_1, 'a')
			cdens_file_2 = h5.File(cdens_fn_2, 'a')

			dir1 = dirs[i] + 'images/1/'
			dir2 = dirs[i] + 'images/2/'

			E1 = rotate(angle, v1, v2)
			E2 = rotate(angle, v2, v1)

			# Identify the radius from the center of each pixel (in sim units)
			px, py = np.mgrid[-width/2:width/2:res*1j, -width/2:width/2:res*1j]	
			radius = (px**2.0 + py**2.0)**0.5
			if "radius" not in cdens_file_1.keys():
				cdens_file_1.create_dataset("radius", data=radius.ravel())

			# Finds azimuthal angle for each pixel
			phi = np.abs(np.arctan(py / px))

			phi *= 180 / np.pi
			if "phi" not in cdens_file_1.keys():
				cdens_file_1.create_dataset("phi", data=phi.ravel())


			log('Generating Edge on Projections with 1st vec')
			log('Ion Fields')
			frb = make_off_axis_projection(ds, E1, L, full_ion_fields, \
			                           c, width, box, rvir, fn_data, dir=dir1)
			
			for i, ion_field in enumerate(ion_fields):
				dset = "%s/%s" % (ion_field, 'edge')
				if dset not in cdens_file_1.keys():
				    cdens_file_1.create_dataset(dset, data=frb[full_ion_fields[i]].ravel())
				    cdens_file_1.flush()
			
			log('Other fields')

			frb = make_off_axis_projection(ds, E1, L, full_other_fields, \
				                           c, width, box, rvir, fn_data, weight_field=('gas', 'density'), dir=dir1)
			for i, other_field in enumerate(other_fields):
				dset = "%s/%s" % (other_field, 'edge')
				if dset not in cdens_file_1.keys():
					cdens_file_1.create_dataset(dset, data=frb[full_other_fields[i]].ravel())
					cdens_file_1.flush()
			cdens_file_1.close()

			
			# Identify the radius from the center of each pixel (in sim units)
			if "radius" not in cdens_file_2.keys():
				cdens_file_2.create_dataset("radius", data=radius.ravel())

			# Finds azimuthal angle for each pixel
			if "phi" not in cdens_file_2.keys():
				cdens_file_2.create_dataset("phi", data=phi.ravel())

			log('Generating Edge on Projections with 2nd vec')
			log('Ion Fields')
			frb = make_off_axis_projection(ds, E2, L, full_ion_fields, \
			                           c, width, box, rvir, fn_data, dir=dir2)
			
			for i, ion_field in enumerate(ion_fields):
				dset = "%s/%s" % (ion_field, 'edge')
				if dset not in cdens_file_2.keys():
				    cdens_file_2.create_dataset(dset, data=frb[full_ion_fields[i]].ravel())
				    cdens_file_2.flush()
			
			log('Other fields')

			frb = make_off_axis_projection(ds, E2, L, full_other_fields, \
				                           c, width, box, rvir, fn_data, weight_field=('gas', 'density'), dir=dir2)
			for i, other_field in enumerate(other_fields):
				dset = "%s/%s" % (other_field, 'edge')
				if dset not in cdens_file_2.keys():
					cdens_file_2.create_dataset(dset, data=frb[full_other_fields[i]].ravel())
					cdens_file_2.flush()
			cdens_file_2.close()
		
