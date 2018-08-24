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
from radial_profile1 import get_amiga_data, smooth_amiga, log


if __name__ == '__main__':
	"""
	Reads Amiga Data, picks desired ion fields, creates edge on projections, and
	outputs data in hdf5 files for later manipulation.
	"""

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
		ions.append('Mg II')
		ion_fields.append('Mg_p1_number_density')
		full_ion_fields.append(('gas', 'Mg_p1_number_density'))
		# ions.append('Si II')
		# ion_fields.append('Si_p1_number_density')
		# full_ion_fields.append(('gas', 'Si_p1_number_density'))
		# ions.append('Si III')
		# ion_fields.append('Si_p2_number_density')
		# full_ion_fields.append(('gas', 'Si_p2_number_density'))
		ions.append('Si IV')
		ion_fields.append('Si_p3_number_density')
		full_ion_fields.append(('gas', 'Si_p3_number_density'))
		# ions.append('N II')
		# ion_fields.append('N_p1_number_density')
		# full_ion_fields.append(('gas', 'N_p1_number_density'))
		# ions.append('N III')
		# ion_fields.append('N_p2_number_density')
		# full_ion_fields.append(('gas', 'N_p2_number_density'))
		# ions.append('N V')
		# ion_fields.append('N_p4_number_density')
		# full_ion_fields.append(('gas', 'N_p4_number_density'))
		ions.append('C II')
		ion_fields.append('C_p1_number_density')
		full_ion_fields.append(('gas', 'C_p1_number_density'))
		ions.append('C III')
		ion_fields.append('C_p2_number_density')
		full_ion_fields.append(('gas', 'C_p2_number_density'))
		ions.append('Ne VIII')
		ion_fields.append('Ne_p7_number_density')
		full_ion_fields.append(('gas', 'Ne_p7_number_density'))
		ions.append('O VI')
		ion_fields.append('O_p5_number_density')
		full_ion_fields.append(('gas', 'O_p5_number_density'))

		others = []
		other_fields = []
		full_other_fields = []
		others.append('Temperature')
		other_fields.append('temperature')
		full_other_fields.append(('gas', 'temperature'))

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
		ions.append('metal_density')
		ion_fields.append('metal_density')
		full_ion_fields.append(('gas', 'metal_density'))

		# Figure out centroid and r_vir info
		log("Reading amiga center for halo in %s" % fn)
		c = read_amiga_center(amiga_data, fn, ds)

		cdens_file_1 = h5.File(cdens_fn_1, 'a')
		cdens_file_2 = h5.File(cdens_fn_2, 'a')

		log('Finding Angular Momentum of Galaxy')
		sp = ds.sphere(c, (15, 'kpc'))
		L = sp.quantities.angular_momentum_vector(use_gas=False, use_particles=True, particle_type='PartType0')

		log('Adding to file')

		cdens_file_1.attrs.create('Ang_Mom', L)
		cdens_file_2.attrs.create('Ang_Mom', L)

		L, E1, E2 = ortho_find(L)

		cdens_file_1.attrs.create('Ang_Mom_Norm', L)
		cdens_file_2.attrs.create('Ang_Mom_Norm', L)

