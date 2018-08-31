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
from radial_profile1 import get_amiga_data, smooth_amiga, log, read_amiga_center, read_amiga_rvir


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
		ions.append('H I')
		ions.append('Mg II')
		ions.append('Si IV')
		ions.append('C II')
		ions.append('C III')
		ions.append('Ne VIII')
		ions.append('O VI')

		log("Starting projections for %s" % fn)
		log(fn)
		ds = GizmoDataset(fn)
		trident.add_ion_fields(ds, ions=ions, ftype='gas')

		# Figure out centroid and r_vir info
		log("Reading amiga center for halo in %s" % fn)
		c = read_amiga_center(amiga_data, fn, ds)
		rvir = read_amiga_rvir(amiga_data, fn, ds)
		
		cdens_file_1 = h5.File(cdens_fn_1, 'a')
		cdens_file_2 = h5.File(cdens_fn_2, 'a')
		
		log('Finding Angular Momentum of Galaxy')
		sp = ds.sphere(c, (15, 'kpccm'))
		L = sp.quantities.angular_momentum_vector(use_gas=False, use_particles=True, particle_type='PartType0')
		mass = sp.quantities.total_mass()

		log('Adding to file')
		mag = np.linalg.norm(L)
		L, E1, E2 = ortho_find(L)
		
		cdens_file_1.attrs.create('Ang_Mom', mag)
		cdens_file_2.attrs.create('Ang_Mom', mag)
		cdens_file_1.attrs.create('Ang_Mom_Norm', L)
		cdens_file_2.attrs.create('Ang_Mom_Norm', L)
		cdens_file_1.attrs.create('mass', mass)
		cdens_file_2.attrs.create('mass', mass)
		cdens_file_1.attrs.create('val', mag/mass[0])
		cdens_file_2.attrs.create('val', mag/mass[0])
		cdens_file_1.attrs.create('rvir', rvir)
		cdens_file_2.attrs.create('rvir', rvir)
		cdens_file_1.close()
		cdens_file_2.close()