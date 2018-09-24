Files:

azimuthal_profile.py - Makes profiles of plots at inclination angles
How to use: 
python azimuthal_projections.py filelist.txt /mnt/data1/GalaxiesOnFire/metaldiff/m12i_res7100_md/halo/ahf/halo_00000_smooth.dat

filelist.txt contains m12i snapshot (hdf5) files
Change the arrays inclines and dirs to change the inclination angles for the projections. It is set to do 0, +-10, +-20 and puts the data in 0/, 10/, 10n/, 20/, 20n/ directories. (n = negative)
Change the ions being projected by uncommenting them


azimuthal_projections.py - makes projections from FIRE 2 data
How to use:
python azimuthal_profile filelist.txt

filelist.txt formatted as:

dir_name:
snapshot_600_cdens.h5


Where dir_name is the directory the profiles are saved to. Needs empty line at the end to work. 

Creates 2 images, one with all ions except Mg II and O VI, the other with just Mg II and O VI. (FINISH WRITING)



projection.py - Creates GridFigure of projections used in paper
How to use:

python projection.py

Just run and it creates the image of the projections used in the paper. 