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
import sys
sys.path.insert(0, '/home/andyr/src/frb')
from yt.utilities.math_utils import ortho_find
from radial_profile1 import *

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

if __name__ == '__main__':
	"""
	Reads Amiga Data, picks desired ion fields, creates edge on projections, and
	outputs data in hdf5 files for later manipulation.
	"""

	# Variables to set for each run
	radial_extent = YTQuantity(250, 'kpc')
	width = 2*radial_extent
	res = 800

	# Loading datasets
	fn_list = open(sys.argv[1], 'r')
	fns = fn_list.readlines()

	# Import FIRE Data
	# Read in halo information from amiga output if present
	amiga_data = get_amiga_data(sys.argv[2])
	# Smooth the data to remove jumps in centroid
	amiga_data = smooth_amiga(amiga_data)

	for fn in yt.parallel_objects(fns):
		fn = fn.strip() # Get rid of trailing \n
		fn_head = fn.split('/')[-1]
		cdens_fn = "%s_cdens.h5" % fn_head

		# Define ions we care about
		ions = []
		ion_fields = []
		full_ion_fields = []
		# ions.append('H I')
		# ion_fields.append('H_number_density')
		# full_ion_fields.append(('gas', 'H_number_density'))
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
		n_fields = len(ion_fields)

		others = []
		other_fields = []
		full_other_fields = []
		# others.append('Temperature')
		# other_fields.append('temperature')
		# full_other_fields.append(('gas', 'temperature'))

		log("Starting projections for %s" % fn)
		ds = GizmoDataset(fn)

		trident.add_ion_fields(ds, ions=ions, ftype='gas')

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

		cdens_file = h5.File(cdens_fn, 'a')

		# Create box around galaxy so we're only sampling galaxy out to 1 Mpc
		one = ds.arr([.5, .5, .5], 'Mpc')
		box = ds.box(c-one, c+one)

		# Identify the radius from the center of each pixel (in sim units)
		px, py = np.mgrid[-width/2:width/2:res*1j, -width/2:width/2:res*1j]
		radius = (px**2.0 + py**2.0)**0.5
		if "radius" not in cdens_file.keys():
			cdens_file.create_dataset("radius", data=radius.ravel())

		# Finds azimuthal angle for each pixel
		phi = np.abs((np.pi / 2) - np.arctan(np.abs(py / px)))
		phi *= 180 / np.pi
		if "phi" not in cdens_file.keys():
			cdens_file.create_dataset("phi", data=phi.ravel())

		log('Finding Angular Momentum of Galaxy')
		log(str(c))
		_, c = ds.find_max('density')
		log(str(c))
		sp = ds.sphere(c, (15, 'kpc'))
		L = sp.quantities.angular_momentum_vector(use_gas=False, use_particles=True, particle_type='PartType0')
		L, E1, E2 = ortho_find(L)
		log(L)
		log(E1)
		log(E2)

		log('Making basic projection')
		p = yt.OffAxisProjectionPlot(ds, E1, 'density', center=c, width=(100, 'kpc'), north_vector=L)
		p.save('off1.png')

		log('Generating Edge on Projections')
		frb = make_off_axis_projection(ds, E1, L, full_ion_fields, \
		                           c, width, box, rvir, dir='edge/')
		
		for i, ion_field in enumerate(ion_fields):
			dset = "%s/%s" % (ion_field, 'edge')
			if dset not in cdens_file.keys():
			    cdens_file.create_dataset(dset, data=frb[full_ion_fields[i]].ravel())
			    cdens_file.flush()
		frb = make_off_axis_projection(ds, E1, L, full_other_fields, \
			                           c, width, box, rvir, weight_field=('gas', 'density'), dir='edge/')
		for i, other_field in enumerate(other_fields):
			dset = "%s/%s" % (other_field, 'edge')
			if dset not in cdens_file.keys():
				cdens_file.create_dataset(dset, data=frb[full_other_fields[i]].ravel())
				cdens_file.flush()
		cdens_file.close()

