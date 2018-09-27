##############################################################
##############################################################

#basic imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
#if os.uname()[1] == 'amalthea':import matplotlib.pyplot as plt

import scipy
import scipy.optimize
import scipy.integrate
from scipy.integrate import quad, dblquad, tplquad, nquad
from scipy.stats import norm, uniform
import random
import subprocess
import string

import astropy
import astropy.coordinates as coord
from astropy.coordinates import ICRS, Galactic, SkyCoord, Distance
from astropy import units as u
import transforms3d
#import mcint
#import cld_fnc as cf 
#import gloess.gloess_fits as gloess
#reload(gloess)
#import gala.coordinates as gc
#import timeit
#import statsmodels

sys.path.append("/Users/hendel/modules/helit/ms")
import ms

##############################################################
##############################################################

#general variables -- these were used to run the SCF simulations
jstokmkpcpersec = 3.241e-23
kmperkpc = 3.08567758*10.**16.
mperkpc  = 3.08567758*10.**19.
sminkg = 1.988435*10.**30.
secperyr = 3.15569e7
gee = 6.674*10**-11. #SI
gphys2 = gee/mperkpc*sminkg # in m^2 kpc / (M_sun s^2)
gphys3 = gee/(mperkpc**3.)*sminkg # in kpc^3/(M_sun s^2)

#halo parameters
mvir=1.77e12 #msun
rvir=389. #kpc
rhalo=24.6 #kpc
cc=rvir/rhalo
G=1
#rhalo = a
mhalo=mvir/(np.log(cc+1.)-cc/(cc+1.))

##############################################################
##############################################################
#functions to read snap/com files and prep aux info

def read_snap(dir, snap):
	#if 
	#read snap files and put them into physical units
	pars = np.loadtxt(dir + 'SCFPAR', dtype='string', usecols=(0,1))
	ru = float(pars[16,0]) #kpc
	mu = float(pars[17,0]) #msun

	gphys1 = 4.49*10**(-6) # in kpc^3/(M_sun Gyr^2)
	tu = np.sqrt(ru**3./(gphys1 * mu)) # in Gyr
	vu = np.sqrt(mu*gphys1/ru) * 3.086*10.**16. * 10.**(-9.) * 3.171*10**-8# in (kpc/Gyr ->) km/s
	eu = vu**2. * 10.**6. # in J per unit mass

	if snap < 10: snapf = 'SNAP00' + str(snap)
	if snap >= 10 and snap < 100: snapf = 'SNAP0' + str(snap)
	if snap >= 100: snapf = 'SNAP' + str(snap)

	tsnap = float(np.loadtxt(dir+snapf, usecols=(0,1),dtype='string')[0,1])
	data = np.loadtxt(dir+snapf, skiprows = 1)
	# mass x y z vx vy vz ep potext tub

	if data.shape[1] == 10:
		#print 'Remnant still bound'
		m = data[:,0] * mu
		x = data[:,1] * ru
		y = data[:,2] * ru
		z = data[:,3] * ru
		vx = data[:,4] * vu
		vy = data[:,5] * vu
		vz = data[:,6] * vu
		ep = data[:,7] * eu
		potex = data[:,8] * eu
		tub = data[:,9] * tu
		return m, x, y, z, vx, vy, vz, ep, potex, tub, tsnap*tu

	if data.shape[1] == 9:
		#print 'Remnant disrupted'
		m = data[:,0] * mu
		x = data[:,1] * ru
		y = data[:,2] * ru
		z = data[:,3] * ru
		vx = data[:,4] * vu
		vy = data[:,5] * vu
		vz = data[:,6] * vu
		potex = data[:,7] * eu
		tub = data[:,8] * tu
		return m, x, y, z, vx, vy, vz, 0.*x, potex, tub, tsnap*tu#*np.ones(len(x))

	if data.shape[1] == 8:
		#print 'Remnant disrupted'
		m = data[:,0] * mu
		x = data[:,1] * ru
		y = data[:,2] * ru
		z = data[:,3] * ru
		vx = data[:,4] * vu
		vy = data[:,5] * vu
		vz = data[:,6] * vu
		potex = data[:,7] * eu
		return m, x, y, z, vx, vy, vz, 0.*x, potex, 0.*x, tsnap*tu#*np.ones(len(x))


def read_com(dir):
	#read COM file and give data physical units
	pars = np.loadtxt(dir + 'SCFPAR', dtype='string', usecols=(0,1))
	ru = float(pars[16,0]) #kpc
	mu = float(pars[17,0]) #msun

	gphys1 = 4.49*10**(-6) # in kpc^3/(M_sun Gyr^2)
	tu = np.sqrt(ru**3./(gphys1 * mu)) # in Gyr
	vu = np.sqrt(mu*gphys1/ru) * kmperkpc * 10.**(-9.) * 3.171*10**-8. # in kpc/Gyr -> km/s

	com = np.loadtxt(dir + 'SCFCEN')
	#time dt x y z vx vy vz

	t = com[:,0] * tu
	dt = com[:,1] * tu
	x_c = com[:,2] * ru
	y_c = com[:,3] * ru
	z_c = com[:,4] * ru
	vx_c = com[:,5] * vu
	vy_c = com[:,6] * vu
	vz_c = com[:,7] * vu

	return t, dt, x_c, y_c, z_c, vx_c, vy_c, vz_c

def radial(x,y,z,vx,vy,vz):
	r = np.sqrt(x**2. + y**2. + z**2.)
	vr = (x*vx + y*vy + z*vz)/abs(r)
	theta = np.arccos(z/r)
	phi = np.arctan2(y,x)
	b = 90. - theta*180./np.pi
	l =  phi*180./np.pi

	return r, theta, phi, vr, l, b

def radial_s(x,y,z,vx,vy,vz):
	#same as radial but w.r.t. sun (at 8.33 kpc)
	x = x+8.33
	r = np.sqrt(x**2. + y**2. + z**2.)
	vr = (x*vx + y*vy + z*vz)/abs(r)
	theta = np.arccos(z/r)
	phi = np.arctan2(y,x)
	b = 90. - theta*180./np.pi
	l =  phi*180./np.pi

	return r, theta, phi, vr, l, b

def get_Ls(x,y,z,vx,vy,vz):
	#x in kpc, vx in km/s, result is J*s per unit mass
	L_x = (y*vz - z*vy)*mperkpc*1000.#*3.086*10.**22.        #yz - zy
	L_y = (z*vx - x*vz)*mperkpc*1000.#*3.086*10.**22.        #zx - xz
	L_z = (x*vy - y*vx)*mperkpc*1000.#*3.086*10.**22.        #xy - yx
	L_mag = np.sqrt(L_x**2. + L_y**2. + L_z**2.)

	return L_x, L_y, L_z, L_mag


##############################################################
##############################################################
#Class objects to hold simulation data + data products

class particles:
	def init(self, dire, snap, do_coords=False, do_jr=False, rel=False):
		q = snap+1
		self.dir = dire
		self.snap = snap
		self.m, self.x, self.y, self.z, self.vx, self.vy, self.vz, self.ep, self.potex, self.tub, self.tsnap = read_snap(dire, snap)
		self.r, self.theta, self.phi, self.vr, self.l, self.b = radial(self.x, self.y, self.z, self.vx, self.vy, self.vz)
		self.r_s, self.theta_s, self.phi_s, self.vr_s, self.l_s, self.b_s = radial_s(self.x, self.y, self.z, self.vx, self.vy, self.vz)
		self.l_x, self.l_y, self.l_z, self.l_mag = get_Ls(self.x, self.y, self.z, self.vx, self.vy, self.vz)
		self.te  = 0.5*1000.*1000.*((self.vx)**2.+(self.vy)**2+(self.vz)**2.) + self.potex
		self.nbods = len(self.x)

class gen_ms():
	def init(self, x, y):
		self.x = x
		self.y = y

	def do_meanshift(self, kernel='uniform', spatial='kd_tree', scale=.5, n_points = 100, do_modes = False, boundmask=False, mask=False):
		if boundmask:
			#subsample sim to just unbound xy
			self.xydata = np.transpose(np.vstack((self.x[self.tub>0],self.y[self.tub>0])))
		else:
			self.xydata = np.transpose(np.vstack((self.x,self.y)))

		#initilize mean shift object
		self.meanshift = ms.MeanShift()
		self.meanshift.set_data(self.xydata,'df')
		self.meanshift.set_kernel(kernel)
		self.meanshift.set_spatial(spatial)
		#scale is 1/distance
		self.meanshift.set_scale(np.array((scale,scale)))

		#the startpoints are random samples of the data
		#they move up the Hessian using mean shift
		#manifolds() projects them onto a 1d manifold instead of the 0d KDE maximum
		self.startpoints = np.random.choice(np.arange(len(self.xydata)),size=n_points,replace=False)
		self.temp = self.meanshift.manifolds(self.xydata[self.startpoints],1)
		self.rpx = self.temp[:,0,0]
		self.rpy = self.temp[:,1,0]
		self.seigval = self.temp[:,0,1]
		self.leigval = self.temp[:,1,1]
		self.seigvec = self.temp[:,:,2]
		self.leigvec = self.temp[:,:,3]

		if do_modes: 
			self.modepoints = self.meanshift.modes(self.xydata[self.startpoints])

	def get_stats_fourier(self, max_term=5):

		n = len(self.rpx)
		indexes = []
		sigma_imag = []
		sigma_real = []
		sum_imag = []
		sum_real = []
		mag_imag = []
		mag_real = []
		mu_ratio = []
		mag_tot = []

		for i in np.arange(n):

			try: x, y = gen_test_set(self,i)
			except: continue

			hist = np.histogram(y, bins=np.linspace(-4,4,40), normed=False)
			fft = np.fft.rfft(hist[0])

			indexes.append(i)
			mag_real.append(np.sum(np.absolute(np.real(fft[1:max_term]))))
			mag_imag.append(np.sum(np.absolute(np.imag(fft[1:max_term]))))
			mag_tot.append( np.sum(np.absolute(       (fft[1:max_term]))))

		self.indexesf = np.array(indexes)
		self.mag_realf = np.array(mag_real)
		self.mag_imagf = np.array(mag_imag)
		self.mag_totf = np.array(mag_tot)

		self.mu_ratiof = self.mag_imagf/self.mag_totf
		self.mean_morphf = np.mean(self.mu_ratiof)

		#warning - not size-adaptive#
		grid_morph_n       = np.histogram2d(self.xydata[self.startpoints,0],self.xydata[self.startpoints,1],
			bins=100,range=[[-50,50],[-50,50]])[0]
		grid_morph_weights = np.histogram2d(self.xydata[self.startpoints,0],self.xydata[self.startpoints,1],
			weights = (self.mu_ratiof), bins=100,range=[[-50,50],[-50,50]])[0]

		self.grid_morphf = np.mean(grid_morph_weights[grid_morph_n>0]/grid_morph_n[grid_morph_n>0])


	def get_stats_orth(self, nterms=5):

		n = len(self.rpx)
		indexes = []
		mag_sym = []
		mag_asym = []
		mu_ratio = []
		mag_tot = []

		for i in np.arange(n):

			try: x, y = gen_test_set(self,i)
			except: continue

			rebase = y*(2*np.pi/8.)
			n = float(len(y))
			aj = np.zeros(nterms)
			bj = np.zeros(nterms)
			var_aj = np.zeros(nterms)
			var_bj = np.zeros(nterms)

			for j in np.arange(nterms):
			    aj[j] = np.sum(np.cos((j+1)*rebase))
			    bj[j] = np.sum(np.sin((j+1)*rebase))

			    var_aj[j] = np.sum(np.cos((j+1)*rebase)*np.cos((j+1)*rebase))-aj[j]
			    var_bj[j] = np.sum(np.sin((j+1)*rebase)*np.sin((j+1)*rebase))-bj[j]

			indexes.append(i)
			mag_sym.append(  np.sum(np.absolute(aj)))
			mag_asym.append( np.sum(np.absolute(bj)))
			mag_tot.append(  np.sum(np.absolute(aj))+np.sum(np.absolute(bj)))

		self.indexes = np.array(indexes)
		self.mag_sym = np.array(mag_sym)
		self.mag_asym = np.array(mag_asym)
		self.mag_tot = np.array(mag_tot)

		self.mu_ratio = self.mag_asym/self.mag_tot
		self.mean_morph = np.mean(self.mu_ratio)
		self.median_morph = np.median(self.mu_ratio)

		#if np.sum(self.tub==0)>0:
		#	satx = np.mean(self.x[self.tub==0])
		#	saty = np.mean(self.y[self.tub==0])
		#	satmask = np.sqrt((self.rpx-satx)**2+(self.rpy-saty)**2)>5
		#else: satmask = True
		#cenmask = np.sqrt((self.rpx)**2+(self.rpy)**2)>15.
		#self.mask_morph = np.mean(self.mu_ratio[cenmask&satmask])

		#warning - not size-adaptive#
		#grid_morph_n       = np.histogram2d(self.xydata[self.startpoints,0],self.xydata[self.startpoints,1],
		#	bins=100,range=[[-50,50],[-50,50]])[0]
		#grid_morph_weights = np.histogram2d(self.xydata[self.startpoints,0],self.xydata[self.startpoints,1],
		#	weights = (self.mu_ratio), bins=100,range=[[-50,50],[-50,50]])[0]
		#self.grid_morph = np.mean(grid_morph_weights[grid_morph_n>0]/grid_morph_n[grid_morph_n>0])


   	def plot_xy(self):
		plt.scatter(self.x, self.y, 
			c='k', alpha=0.25,edgecolor='none', s=4)
   		if (hasattr(self, 'rpx') & hasattr(self, 'mu_ratio')):
   			plt.scatter(self.rpx, self.rpy, 
				c=np.abs(self.mu_ratio), cmap='plasma', alpha=1,edgecolor='none', s=5)
   			plt.colorbar()
		elif (hasattr(self, 'rpx')):
			plt.scatter(self.rpx, self.rpy, 
				c='cyan', alpha=1,edgecolor='none', s=5)
			#plt.colorbar()



class sim_ms(particles):
	def rotate(self,inclination = 0.):
		#requires transforms3d
		#angle should be in radians
		rot = transforms3d.euler.euler2mat(0, inclination, 0)
		self.inclination = inclination
		self.x, self.y, self.z = np.dot(rot, [self.x,self.y,self.z])

	def do_meanshift(self, kernel='uniform', spatial='kd_tree', scale=.5, n_points = 100, do_modes = False, boundmask=True, mask=False):
		if boundmask:
			#subsample sim to just unbound xy
			self.xydata = np.transpose(np.vstack((self.x[self.tub>0],self.y[self.tub>0])))
		else:
			self.xydata = np.transpose(np.vstack((self.x,self.y)))

		#initilize mean shift object
		self.meanshift = ms.MeanShift()
		self.meanshift.set_data(self.xydata,'df')
		self.meanshift.set_kernel(kernel)
		self.meanshift.set_spatial(spatial)
		#scale is 1/distance
		self.meanshift.set_scale(np.array((scale,scale)))

		#the startpoints are random samples of the data
		#they move up the Hessian using mean shift
		#manifolds() projects them onto a 1d manifold instead of the 0d KDE maximum
		self.startpoints = np.random.choice(np.arange(len(self.xydata)),size=n_points,replace=False)
		self.temp = self.meanshift.manifolds(self.xydata[self.startpoints],1)
		self.rpx = self.temp[:,0,0]
		self.rpy = self.temp[:,1,0]
		self.seigval = self.temp[:,0,1]
		self.leigval = self.temp[:,1,1]
		self.seigvec = self.temp[:,:,2]
		self.leigvec = self.temp[:,:,3]

		if do_modes: 
			self.modepoints = self.meanshift.modes(self.xydata[self.startpoints])

	def get_stats_fourier(self, max_term=5):

		n = len(self.rpx)
		indexes = []
		sigma_imag = []
		sigma_real = []
		sum_imag = []
		sum_real = []
		mag_imag = []
		mag_real = []
		mu_ratio = []
		mag_tot = []

		for i in np.arange(n):

			try: x, y = gen_test_set(self,i)
			except: continue

			hist = np.histogram(y, bins=np.linspace(-4,4,40), normed=False)
			fft = np.fft.rfft(hist[0])

			indexes.append(i)
			mag_real.append(np.sum(np.absolute(np.real(fft[1:max_term]))))
			mag_imag.append(np.sum(np.absolute(np.imag(fft[1:max_term]))))
			mag_tot.append( np.sum(np.absolute(       (fft[1:max_term]))))

		self.indexesf = np.array(indexes)
		self.mag_realf = np.array(mag_real)
		self.mag_imagf = np.array(mag_imag)
		self.mag_totf = np.array(mag_tot)

		self.mu_ratiof = self.mag_imagf/self.mag_totf
		self.mean_morphf = np.mean(self.mu_ratiof)

		#warning - not size-adaptive#
		grid_morph_n       = np.histogram2d(self.xydata[self.startpoints,0],self.xydata[self.startpoints,1],
			bins=100,range=[[-50,50],[-50,50]])[0]
		grid_morph_weights = np.histogram2d(self.xydata[self.startpoints,0],self.xydata[self.startpoints,1],
			weights = (self.mu_ratiof), bins=100,range=[[-50,50],[-50,50]])[0]

		self.grid_morphf = np.mean(grid_morph_weights[grid_morph_n>0]/grid_morph_n[grid_morph_n>0])


	def get_stats_orth(self, nterms=5):

		n = len(self.rpx)
		indexes = []
		mag_sym = []
		mag_asym = []
		mu_ratio = []
		mag_tot = []

		for i in np.arange(n):

			try: x, y = gen_test_set(self,i)
			except: continue

			rebase = y*(2*np.pi/8.)
			n = float(len(y))
			aj = np.zeros(nterms)
			bj = np.zeros(nterms)
			var_aj = np.zeros(nterms)
			var_bj = np.zeros(nterms)

			for j in np.arange(nterms):
			    aj[j] = np.sum(np.cos((j+1)*rebase))
			    bj[j] = np.sum(np.sin((j+1)*rebase))

			    var_aj[j] = np.sum(np.cos((j+1)*rebase)*np.cos((j+1)*rebase))-aj[j]
			    var_bj[j] = np.sum(np.sin((j+1)*rebase)*np.sin((j+1)*rebase))-bj[j]

			indexes.append(i)
			mag_sym.append(  np.sum(np.absolute(aj)))
			mag_asym.append( np.sum(np.absolute(bj)))
			mag_tot.append(  np.sum(np.absolute(aj))+np.sum(np.absolute(bj)))

		self.indexes = np.array(indexes)
		self.mag_sym = np.array(mag_sym)
		self.mag_asym = np.array(mag_asym)
		self.mag_tot = np.array(mag_tot)

		self.mu_ratio = self.mag_asym/self.mag_tot
		self.mean_morph = np.mean(self.mu_ratio)

		if np.sum(self.tub==0)>0:
			satx = np.mean(self.x[self.tub==0])
			saty = np.mean(self.y[self.tub==0])
			satmask = np.sqrt((self.rpx-satx)**2+(self.rpy-saty)**2)>5
		else: satmask = True
		cenmask = np.sqrt((self.rpx)**2+(self.rpy)**2)>15.
		self.mask_morph = np.mean(self.mu_ratio[cenmask&satmask])

		#warning - not size-adaptive#
		#grid_morph_n       = np.histogram2d(self.xydata[self.startpoints,0],self.xydata[self.startpoints,1],
		#	bins=100,range=[[-50,50],[-50,50]])[0]
		#grid_morph_weights = np.histogram2d(self.xydata[self.startpoints,0],self.xydata[self.startpoints,1],
		#	weights = (self.mu_ratio), bins=100,range=[[-50,50],[-50,50]])[0]
		#self.grid_morph = np.mean(grid_morph_weights[grid_morph_n>0]/grid_morph_n[grid_morph_n>0])


   	def plot_xy(self):
		plt.scatter(self.x, self.y, 
			c='k', alpha=0.1,edgecolor='none', s=4)
   		if (hasattr(self, 'rpx') & hasattr(self, 'mu_ratio')):
   			plt.scatter(self.rpx, self.rpy, 
				c=np.abs(self.mu_ratio), cmap=cmap, alpha=1,edgecolor='none', s=5)
   			plt.colorbar()
		if (hasattr(self, 'rpx')):
			plt.scatter(self.rpx, self.rpy, 
				c='cyan', alpha=1,edgecolor='none', s=5)
			plt.colorbar()


def var_test(sim_ms,i,snr=1):
	x, y = mspy.gen_test_set(sim_ms,i)
	rebase = y*(2*np.pi/8.)
	n = float(len(y))
	aj = np.zeros(nterms)
	bj = np.zeros(nterms)
	var_aj = np.zeros(nterms)
	var_bj = np.zeros(nterms)
	for j in np.arange(nterms):
	    aj[j] = np.sum(np.cos((j+1)*rebase))
	    bj[j] = np.sum(np.sin((j+1)*rebase))
	    var_aj[j] = np.sum(np.cos((j+1)*rebase)*np.cos((j+1)*rebase))-aj[j]
	    var_bj[j] = np.sum(np.sin((j+1)*rebase)*np.sin((j+1)*rebase))-bj[j]

	plt.hist(y, np.linspace(-4,4,40), histtype='step')

	newx = np.linspace(-np.pi,np.pi,40)
	newy = np.zeros(len(newx))

	for j in np.arange(nterms):
		newy = newy + 2*(aj[j]*np.cos((j+1)*newx) + bj[j]*np.sin((j+1)*newx))

	newy = n + newy
	plt.plot(newx/(2*np.pi/8.),newy/40., label = 'all')


	sna = np.sqrt(aj**2/var_aj)
	print sna
	snb = np.sqrt(bj**2/var_bj)
	print snb
	newy = np.zeros(len(newx))

	for j in np.arange(nterms):
		newy = newy + 2*(aj[j]*np.cos((j+1)*newx)*(sna[j]>snr) + bj[j]*np.sin((j+1)*newx))*(snb[j]>snr)

	newy = n + newy
	plt.plot(newx/(2*np.pi/8.),newy/40., label = 'high snr')
	plt.legend()


def load_sim_ms(file):
	with open(file, "rb") as f:
		return pickle.load(f)

#generate a subset of the data in a box aligned with the smallest eigenvalue of the local Hessian
def gen_test_set(sim_ms, i, box_perp = 8., box_para = 4., rot_out = False):
	#negative to rotate back to horizontal
	theta = np.arctan2(sim_ms.seigvec[i,0],sim_ms.seigvec[i,1])
	rot_matrix = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta),np.cos(theta))))

	dx = sim_ms.x - sim_ms.rpx[i]
	dy = sim_ms.y - sim_ms.rpy[i]

	(newdx,newdy) = np.dot(rot_matrix,((dx,dy)))

	if ((box_para>0) & (box_perp>0)):
		sel = ((newdx > - box_para/2.) & (newdx < box_para/2.) & 
			(newdy > - box_perp/2.) & (newdy < box_perp/2.))

	if rot_out ==True:
		return newdx[sel], newdy[sel], rot_matrix
	else:return newdx[sel], newdy[sel]

#plot a subset, rotated into the local ridge frame in x,y and hist(y)
def plot_subset(sim_ms, k, box_perp = 20., box_para =20.):
	plt.clf()
	#plot data used for a given ridge point
	plt.subplot(1,2,1,aspect='equal')
	dx, dy = gen_test_set(sim_ms, k, box_perp = box_perp, box_para = box_para)
	plt.scatter(sim_ms.x,sim_ms.y,s=2,c='k',alpha=0.03)
	plt.scatter(dy,dx, c='k', s=3, alpha=0.5, edgecolor='none')
	plt.xlabel('y')
	plt.ylabel('x')
	plt.subplot(1,2,2)
	plt.hist(dy, bins=np.linspace(-4,4,40))
	#plt.title("p-value = %0.6f"%pvals[k])

def plot_subset_fourier(sim_ms, k):
	plt.clf()
	#plot data used for a given ridge point
	plt.subplot(2,2,1, aspect='equal')
	dx, dy, rot = gen_test_set(sim_ms, k,rot_out=True)
	plt.scatter(dy,dx, c='k', s=3, alpha=0.5, edgecolor='none')
	(ridgex, ridgey) = np.dot(rot, ((sim_ms.rpx, sim_ms.rpy)))
	plt.scatter(ridgey-ridgey[k],ridgex-ridgex[k],c='r', edgecolor='none')
	plt.xlim([-4,4])
	plt.ylim([-2,2])
	plt.xlabel('y')
	plt.ylabel('x')

	plt.subplot(2,2,2)
	plt.hist(dy, bins=np.linspace(-4,4,40),histtype='step', lw=3, normed=False)
	hist = np.histogram(dy, bins=np.linspace(-4,4,40), normed=False)
	plt.xlabel('y')
	plt.ylabel('normed counts')

	plt.subplot(2,2,3)
	fft = np.fft.rfft(hist[0])
	plt.scatter(np.real(fft[1:]), -np.imag(fft[1:]))
	plt.xlabel('Re(coeff) [cosine parts]')
	plt.ylabel('-i * Im(coeff) [sine parts]')

	plt.subplot(2,2,4)
	plt.hist(dy, bins=np.linspace(-4,4,40),histtype='step', lw=3, normed=False)
	plt.plot(np.linspace(-4,4,38),np.fft.irfft(fft), label = 'reconstruction', linestyle = '--', c='g')
	fft_nomean = fft
	fft_nomean[0]=0
	plt.plot(np.linspace(-4,4,38),np.fft.irfft(np.real(fft_nomean)), label = 'cosine parts', linestyle = '-', c='r')
	plt.plot(np.linspace(-4,4,38),np.fft.irfft(1j*np.imag(fft)), label = 'sine parts', linestyle = '-', c='b')
	plt.legend(fontsize='x-small')

def get_single_morph_fourier(sim_ms, k, box_perp = 8., box_para = 4., max_term=10, bins=40):
	x, y = gen_test_set(sim_ms, k)
	hist = np.histogram(y, bins=np.linspace(-box_perp/2, box_perp/2, bins), normed=False)
	fft = np.fft.rfft(hist[0])
	morph =     np.sum(np.absolute(np.real(fft[1:max_term])))/np.sum(np.absolute((fft[1:max_term])))
	#inv_morph = np.sum(np.absolute(np.imag(fft[1:max_term])))/np.sum(np.absolute((fft[1:max_term])))

	return morph

def get_single_morph_ortho(sim_ms, k, box_perp = 8., box_para = 4., max_term=100):
	x, y = gen_test_set(sim_ms, k, box_perp = box_perp, box_para = box_para)

	rebase = y*(2*np.pi/8.)
	n = float(len(y))
	aj = np.zeros(int(max_term))
	bj = np.zeros(int(max_term))

	for j in np.arange(int(max_term)):
		aj[j] = np.sum(np.cos((j+1)*rebase))
		bj[j] = np.sum(np.sin((j+1)*rebase))
	mag_sym= np.sum(np.absolute(aj))
	mag_asym=np.sum(np.absolute(bj))
	mu_ratio = mag_asym/(mag_asym+mag_sym)

	return mu_ratio

#get FFT statistics for the ridgepoint selections
def get_stats_fourier(sim_ms):

	n = len(sim_ms.rpx)
	indexes = []
	sigma_imag = []
	sigma_real = []
	sum_imag = []
	sum_real = []
	mag_imag = []
	mag_real = []
	mu1 = []

	for i in np.arange(n):

		try: x, y = gen_test_set(sim_ms,i)
		except: continue

		hist = np.histogram(y, bins=np.linspace(-4,4,40), normed=False)
		fft = np.fft.rfft(hist[0])

		indexes.append(i)
		sigma_real.append(np.std(np.real(fft[1:])))
		sigma_imag.append(np.std(np.imag(fft[1:])))
		sum_real.append(np.sum(np.real(fft[1:])))
		sum_imag.append(np.sum(np.imag(fft[1:])))
		mag_real.append(np.sum(np.sqrt(np.real(fft[1:])**2)))
		mag_imag.append(np.sum(np.sqrt(np.imag(fft[1:])**2)))
		mu1.append(np.real(np.sum((np.real(fft[1:])-np.imag(fft[1:]))/(np.sqrt(fft[1:]*np.conj(fft[1:]))))))

	return np.array(indexes), np.array(sigma_real), np.array(sigma_imag), np.array(sum_real), np.array(sum_imag), np.array(mag_real), np.array(mag_imag), np.array(mu1)

#plot subset and fourier transform - use GLOESS to smooth the histogram
def plot_subset_fourier_gloess(sim_ms, k):
	plt.clf()
	#plot data used for a given ridge point
	plt.subplot(2,2,1, aspect='equal')
	dx, dy, rot = gen_test_set(sim_ms, k,rot_out=True)
	plt.scatter(dy,dx, c='k', s=3, alpha=0.5, edgecolor='none')
	(ridgex, ridgey) = np.dot(rot, ((sim_ms.rpx, sim_ms.rpy)))
	plt.scatter(ridgey-ridgey[k],ridgex-ridgex[k],c='r', edgecolor='none')
	plt.xlim([-4,4])
	plt.ylim([-2,2])
	plt.xlabel('y')
	plt.ylabel('x')

	plt.subplot(2,2,2)
	plt.hist(dy, bins=np.linspace(-4,4,40),histtype='step', lw=3, normed=False)
	hist = np.histogram(dy, bins=np.linspace(-4,4,40), normed=False)
	data1, x, ys, yerrs, xs = gloess.fit_generic_jit_start(hist[1][1:],hist[0],1./np.sqrt(hist[0]), smooth = 1.)
	plt.plot(np.linspace(-4,4,100), data1[100:200])
	plt.xlabel('y')
	plt.ylabel('normed counts')

	plt.subplot(2,2,3)
	fft = np.fft.rfft(data1[100:200])
	plt.scatter(np.real(fft[1:]), -np.imag(fft[1:]))
	plt.xlabel('Re(coeff) [cosine parts]')
	plt.ylabel('-i * Im(coeff) [sine parts]')

	plt.subplot(2,2,4)
	plt.hist(dy, bins=np.linspace(-4,4,40),histtype='step', lw=3, normed=False)
	plt.plot(np.linspace(-4,4,100),np.fft.irfft(fft), label = 'reconstruction', linestyle = '--', c='g')
	fft_nomean = fft
	fft_nomean[0]=0
	plt.plot(np.linspace(-4,4,100),np.fft.irfft(np.real(fft_nomean)), label = 'cosine parts', linestyle = '-', c='r')
	plt.plot(np.linspace(-4,4,100),np.fft.irfft(1j*np.imag(fft)), label = 'sine parts', linestyle = '-', c='b')
	plt.legend(fontsize='x-small')

# --- depreciated - part of sim_ms ------
#perform Subspace-Constrained Mean Shift on a simulation. Scale is 1/bandwith. Only Gaussian kernals supported.
def sim_meanshift(sim, kernel='uniform', spatial='kd_tree', scale=.5, n_points = 3000, do_modes = False):
	#subsample sim to just unbound xy
	data = np.transpose(np.vstack((sim.x[sim.tub>0],sim.y[sim.tub>0])))
	#initilize mean shift object
	meanshift = ms.MeanShift()
	meanshift.set_data(data,'df')
	meanshift.set_kernel(kernel)
	meanshift.set_spatial(spatial)
	#scale is 1/distance
	meanshift.set_scale(np.array((scale,scale)))

	#the startpoints are random samples of the data
	#they move up the Hessian using mean shift
	#manifolds() projects them onto a 1d manifold instead of the 0d KDE maximum
	startpoints =np.random.choice(np.arange(len(data)),size=n_points,replace=False)
	ridgepoints = meanshift.manifolds(data[startpoints],1)

	if do_modes: 
		modepoints = meanshift.modes(data[startpoints])
		return meanshift, startpoints, ridgepoints, modepoints

	return meanshift, startpoints, ridgepoints

# --- depreciated - use gen_test_set (rotates with eigenvector) ------
#fit a line to ridgepoints in a small area; should be depreciated in favor of using \vec{V}
def local_line_fit(ridgepoints, i, verbose = True):
	#select all ridge points within 1 of ridgepoint i, fit line to them
	ridge_dx = ridgepoints[:,0] - ridgepoints[i,0]
	ridge_dy = ridgepoints[:,1] - ridgepoints[i,1]
	distance = np.sqrt(ridge_dx**2+ridge_dy**2)
	line_fit_sel = distance<1.

	if np.sum(line_fit_sel)<5:
		if verbose: print "not enough points to fit line"
		return -1

	line_fit = scipy.stats.linregress(ridgepoints[line_fit_sel,0],ridgepoints[line_fit_sel,1])

	if (line_fit[2]**2)<.7:
		if verbose: print "bad line fit, r^2 = %f0.2"%(line_fit[2]**2)
		return -2

	return line_fit

# --- depreciated - use gen_test_set (rotates with eigenvector) ------
#generate a subset of the data in a box aligned with the local
def gen_test_set_line(sim, ridgepoints, i ,box_perp = 8., box_para = 4., rot_out = False):
	#negative to rotate back to horizontal
	line_fit = local_line_fit(ridgepoints, i, verbose = True)
	theta = -np.arctan2(line_fit[0],1.)
	rot_matrix = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta),np.cos(theta))))

	dx = sim.x - ridgepoints[i,0]
	dy = sim.y - ridgepoints[i,1]

	(newdx,newdy) = np.dot(rot_matrix,((dx,dy)))

	if ((box_para>0) & (box_perp>0)):
		sel = ((newdx > - box_para/2.) & (newdx < box_para/2.) & 
			(newdy > - box_perp/2.) & (newdy < box_perp/2.))

	if rot_out ==True:
		return newdx[sel], newdy[sel], rot_matrix
	else:return newdx[sel], newdy[sel]

# --- depreciated --- #
#get distribution statistics for the ridgepoint selections
def get_stats_basic(sim,ridgepoints):
	n = len(ridgepoints)
	indexes = []
	means = []
	medians = []
	stdevs = []
	skews = []
	kurtosises = []
	#sigmaderiv = []

	for i in np.arange(n):

		try: x, y = gen_test_set(sim,ridgepoints,i)
		except: continue

		indexes.append(i)
		means.append(np.mean(y))
		medians.append(np.median(y))
		stdevs.append(np.std(y))
		skews.append(scipy.stats.skew(y))
		kurtosises.append(scipy.stats.kurtosis(y))

		#hist = np.histogram(y, bins=np.linspace(-4,4,40))
		#deriv = (np.gradient(hist[0]))
		#sigmaderiv.append(np.max(np.abs(deriv))/np.std(deriv))

	return np.array(indexes), np.array(means), np.array(medians), np.array(stdevs), np.array(skews), np.array(kurtosises)

# --- depreciated --- #
#compute Wilcoxon signed-rank test p-value for each ridgepoint
def get_stats_wilcoxon(sim, ridgepoints, box_perp = 8., box_para = 4., verbose=True):
	#calculate symmetry test for every ridge point
	#use box aligned with local ridge
	pvals = np.zeros(len(ridgepoints))
	for i in np.arange(len(ridgepoints)):

		line_fit = local_line_fit(ridgepoints, i)

		if line_fit == -1: 
			pvals[i] = -1
			continue
		if line_fit == -2: 
			pvals[i] = -2
			continue

		newdx, newdy = gen_test_set(sim, ridgepoints, i, box_perp = box_perp, box_para = box_para)

		#select data and run symmetry test
		if len(newdx)<20:
			if verbose: print "not enough data to test symmetry"
			pvals[i] = -3
			continue

		sel_median = np.median(newdy)
		pvals[i] = scipy.stats.wilcoxon(x=(newdy-sel_median))[1]

	return pvals

# --- depreciated --- #
#get FFT statistics for the ridgepoint selections
def get_stats_fft(sim,ridgepoints):

	n = len(ridgepoints)
	indexes = []
	sigma_imag = []
	sigma_real = []
	sum_imag = []
	sum_real = []
	mag_imag = []
	mag_real = []
	mu1 = []

	for i in np.arange(n):

		try: x, y = gen_test_set(sim,ridgepoints,i)
		except: continue

		hist = np.histogram(y, bins=np.linspace(-4,4,40), normed=False)
		y_in = np.array(hist[0])+1.
		x_in = np.linspace(-4,4,40)
		data1, x, ys, yerrs, xs = gloess.fit_generic_jit_start(x_in,y_in,1./np.sqrt(y_in), smooth = 1.)
		data1 = data1-1.
		fft = np.fft.rfft(data1[100:200])

		indexes.append(i)
		sigma_real.append(np.std(np.real(fft[1:])))
		sigma_imag.append(np.std(np.imag(fft[1:])))
		sum_real.append(np.sum(np.real(fft[1:])))
		sum_imag.append(np.sum(np.imag(fft[1:])))
		mag_real.append(np.sum(np.sqrt(np.real(fft[1:])**2)))
		mag_imag.append(np.sum(np.sqrt(np.imag(fft[1:])**2)))
		mu1.append(np.real(np.sum((np.real(fft[1:])-np.imag(fft[1:]))/(np.sqrt(fft[1:]*np.conj(fft[1:]))))))

	return np.array(indexes), np.array(sigma_real), np.array(sigma_imag), np.array(sum_real), np.array(sum_imag), np.array(mag_real), np.array(mag_imag), np.array(mu1)

# --- depreciated --- #
#get statistics based on the Sobel filter for the ridgepoint selections
def get_stats_gloess(sim,ridgepoints):
	n = len(ridgepoints)
	indexes = []
	sobel_max = []
	sobel_ratio = []
	sobel_integral = []
	for i in np.arange(n):

		try: x, y = gen_test_set(sim,ridgepoints,i)
		except: continue

		indexes.append(i)
		hist = np.histogram(y, bins=np.linspace(-4,4,40), normed=False)
		data1, x, ys, yerrs, xs = gloess.fit_generic_jit_start(hist[1][1:],hist[0],1./np.sqrt(hist[0]), smooth = 1.)
		sobel = scipy.ndimage.sobel(data1[100:200])
		sobel_max.append(np.max(np.abs(sobel)))
		
		try:
			ratio = -np.max(sobel[sobel>0])/np.min(sobel[sobel<0])
			sobel_ratio.append(np.maximum(ratio,1./ratio))
		except: 
			sobel_ratio.append(0)
			#print 'error'
			continue

 	return np.array(indexes), np.array(sobel_max), np.array(sobel_ratio)

# --- depreciated --- #
#corner plot of some statistics - needs some inputs defined
def plot_stats_basic():

	fig = plt.subplots()
	ax1 = plt.subplot2grid((4,4), (0, 0))
	plt.scatter(sh_means, sh_medians, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_means, st_medians, c='r', edgecolor='none',alpha = 0.5)
	plt.ylabel('medians')
	ax2 = plt.subplot2grid((4,4), (1, 0))
	plt.scatter(sh_means, sh_stdevs, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_means, st_stdevs, c='r', edgecolor='none',alpha = 0.5)
	plt.ylabel('stdevs')
	ax3 = plt.subplot2grid((4,4), (2, 0))
	plt.scatter(sh_means, sh_skews, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_means, st_skews, c='r', edgecolor='none',alpha = 0.5)
	plt.ylabel('skews')
	ax4 = plt.subplot2grid((4,4), (3, 0))
	plt.scatter(sh_means, sh_kurtosises, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_means, st_kurtosises, c='r', edgecolor='none',alpha = 0.5)
	plt.ylabel('kurtosises')
	plt.xlabel('means')
	ax5 = plt.subplot2grid((4,4), (1, 1))
	plt.scatter(sh_medians, sh_stdevs, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_medians, st_stdevs, c='r', edgecolor='none',alpha = 0.5)
	ax6 = plt.subplot2grid((4,4), (2, 1))
	plt.scatter(sh_medians, sh_skews, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_medians, st_skews, c='r', edgecolor='none',alpha = 0.5)
	ax7 = plt.subplot2grid((4,4), (3, 1))
	plt.scatter(sh_medians, sh_kurtosises, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_medians, st_kurtosises, c='r', edgecolor='none',alpha = 0.5)
	plt.xlabel('medians')
	ax8 = plt.subplot2grid((4,4), (2, 2))
	plt.scatter(sh_stdevs, sh_skews, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_stdevs, st_skews, c='r', edgecolor='none',alpha = 0.5)
	ax9 = plt.subplot2grid((4,4), (3, 2))
	plt.scatter(sh_stdevs, sh_kurtosises, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_stdevs, st_kurtosises, c='r', edgecolor='none',alpha = 0.5)
	plt.xlabel('stdevs')
	ax10 = plt.subplot2grid((4,4), (3, 3))
	plt.scatter(sh_skews, sh_kurtosises, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_skews, st_kurtosises, c='r', edgecolor='none',alpha = 0.5)
	plt.xlabel('skews')

# --- depreciated --- #
def plot_stats_fourier():

	fig = plt.subplots()
	ax1 = plt.subplot2grid((4,4), (0, 0))
	plt.scatter(sh_means, sh_medians, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_means, st_medians, c='r', edgecolor='none',alpha = 0.5)
	plt.ylabel('imag_mag')
	ax2 = plt.subplot2grid((4,4), (1, 0))
	plt.scatter(sh_means, sh_stdevs, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_means, st_stdevs, c='r', edgecolor='none',alpha = 0.5)
	plt.ylabel('imag_')
	ax3 = plt.subplot2grid((4,4), (2, 0))
	plt.scatter(sh_means, sh_skews, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_means, st_skews, c='r', edgecolor='none',alpha = 0.5)
	plt.ylabel('skews')
	ax4 = plt.subplot2grid((4,4), (3, 0))
	plt.scatter(sh_means, sh_kurtosises, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_means, st_kurtosises, c='r', edgecolor='none',alpha = 0.5)
	plt.ylabel('kurtosises')
	plt.xlabel('means')

	ax5 = plt.subplot2grid((4,4), (1, 1))
	plt.scatter(sh_medians, sh_stdevs, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_medians, st_stdevs, c='r', edgecolor='none',alpha = 0.5)
	ax6 = plt.subplot2grid((4,4), (2, 1))
	plt.scatter(sh_medians, sh_skews, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_medians, st_skews, c='r', edgecolor='none',alpha = 0.5)
	ax7 = plt.subplot2grid((4,4), (3, 1))
	plt.scatter(sh_medians, sh_kurtosises, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_medians, st_kurtosises, c='r', edgecolor='none',alpha = 0.5)
	plt.xlabel('medians')
	ax8 = plt.subplot2grid((4,4), (2, 2))
	plt.scatter(sh_stdevs, sh_skews, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_stdevs, st_skews, c='r', edgecolor='none',alpha = 0.5)
	ax9 = plt.subplot2grid((4,4), (3, 2))
	plt.scatter(sh_stdevs, sh_kurtosises, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_stdevs, st_kurtosises, c='r', edgecolor='none',alpha = 0.5)
	plt.xlabel('stdevs')
	ax10 = plt.subplot2grid((4,4), (3, 3))
	plt.scatter(sh_skews, sh_kurtosises, c='k', edgecolor='none',alpha = 0.5)
	plt.scatter(st_skews, st_kurtosises, c='r', edgecolor='none',alpha = 0.5)
	plt.xlabel('skews')

# --- depreciated --- #
def plot_subset_spline(sim, ridgepoints, k):
	plt.clf()
	#plot data used for a given ridge point
	plt.subplot(2,2,1,aspect='equal')
	dx, dy = gen_test_set(sim,ridgepoints,k)
	plt.scatter(dy,dx, c='k', s=3, alpha=0.25, edgecolor='none')
	plt.xlabel('y')
	plt.ylabel('x')
	plt.subplot(2,2,2)
	plt.hist(dy, bins='40')
	plt.subplot(2,2,3)
	datahist = np.histogram(dy, bins='auto', range=(-box_perp/2.,box_perp/2.))
	spline = scipy.interpolate.UnivariateSpline(datahist[1][1:],datahist[0], ext=1)	

# --- depreciated --- #
def plot_subset_fft(sim, ridgepoints,k):
	plt.clf()
	#plot data used for a given ridge point
	plt.subplot(2,2,1,aspect='equal')
	dx, dy, rot = gen_test_set(sim,ridgepoints,k,rot_out=True)
	plt.scatter(dy,dx, c='k', s=3, alpha=0.5, edgecolor='none')
	(ridgex, ridgey) = np.dot(rot, ((ridgepoints[:,0],ridgepoints[:,1])))
	plt.scatter(ridgey-ridgey[k],ridgex-ridgex[k],c='r', edgecolor='none')
	plt.xlim([-4,4])
	plt.ylim([-2,2])
	plt.xlabel('y')
	plt.ylabel('x')

	plt.subplot(2,2,2)
	plt.hist(dy, bins=np.linspace(-4,4,40),histtype='step', lw=3, normed=False)
	hist = np.histogram(dy, bins=np.linspace(-4,4,40), normed=False)
	data1, x, ys, yerrs, xs = gloess.fit_generic_jit_start(hist[1][1:],hist[0],1./np.sqrt(hist[0]), smooth = 1.)
	plt.plot(np.linspace(-4,4,100), data1[100:200])
	plt.xlabel('y')
	plt.ylabel('normed counts')

	plt.subplot(2,2,3)
	fft = np.fft.rfft(data1[100:200])
	plt.scatter(np.real(fft[1:]), -np.imag(fft[1:]))
	plt.xlabel('Re(coeff) [cosine parts]')
	plt.ylabel('-i * Im(coeff) [sine parts]')

	plt.subplot(2,2,4)
	plt.hist(dy, bins=np.linspace(-4,4,40),histtype='step', lw=3, normed=False)
	plt.plot(np.linspace(-4,4,100),np.fft.irfft(fft), label = 'reconstruction', linestyle = '--', c='g')
	fft_nomean = fft
	fft_nomean[0]=0
	plt.plot(np.linspace(-4,4,100),np.fft.irfft(np.real(fft_nomean)), label = 'cosine parts', linestyle = '-', c='r')
	plt.plot(np.linspace(-4,4,100),np.fft.irfft(1j*np.imag(fft)), label = 'sine parts', linestyle = '-', c='b')
	plt.legend(fontsize='x-small')

# --- depreciated --- #
def plot_subset_gloess(sim, ridgepoints, k):
	plt.clf()
	#plot data used for a given ridge point
	plt.subplot(2,2,1,aspect='equal')
	dx, dy, rot = gen_test_set(sim,ridgepoints,k,rot_out=True)
	plt.scatter(dy,dx, c='k', s=3, alpha=0.5, edgecolor='none')
	(ridgex, ridgey) = np.dot(rot, ((ridgepoints[:,0],ridgepoints[:,1])))
	plt.scatter(ridgey-ridgey[k],ridgex-ridgex[k],c='r', edgecolor='none')
	plt.xlim([-4,4])
	plt.ylim([-2,2])
	plt.xlabel('y')
	plt.ylabel('x')

	plt.subplot(2,2,2)
	plt.hist(dy, bins=np.linspace(-4,4,40))
	hist = np.histogram(dy, bins=np.linspace(-4,4,40), normed=False)

	plt.subplot(2,2,3)
	plt.hist(dy, bins=np.linspace(-4,4,40),histtype='step', lw=3, normed=False)
	hist = np.histogram(dy, bins=np.linspace(-4,4,40), normed=False)
	data1, x, ys, yerrs, xs = gloess.fit_generic_jit_start(hist[1][1:],hist[0],1./np.sqrt(hist[0]), smooth = 1.)
	plt.plot(np.linspace(-4,4,100), data1[100:200])

	plt.subplot(2,2,4)
	#plt.plot(hist[1][1:-1],np.diff(hist[0]), label = 'hist diff')
	sobel = scipy.ndimage.sobel(data1[100:200])
	plt.plot(np.linspace(-4,4,100),sobel, label ='sobel')
	#plt.plot(np.linspace(-4,4,100),np.gradient(data1[100:200]), label = 'gradient')
	#plt.plot(np.linspace(-4,4,100),scipy.ndimage.laplace(data1[100:200]), label = 'laplace')
	ax =plt.gca()
	plt.text(0.05,0.17,'Sobel max   = %0.4f'%(np.max(abs(sobel))), transform=ax.transAxes,fontsize='x-small')
	ratio = -np.max(sobel[sobel>0])/np.min(sobel[sobel<0])
	plt.text(0.05,0.1,'Sobel ratio = %0.4f'%(np.maximum(ratio,1./ratio)), transform=ax.transAxes,fontsize='x-small')
	plt.legend(fontsize='x-small')
	
# --- depreciated --- #
def plot_subset_deriv(sim, ridgepoints,k):
	plt.clf()
	#plot data used for a given ridge point
	plt.subplot(2,2,1,aspect='equal')
	dx, dy, rot = gen_test_set(sim,ridgepoints,k,rot_out=True)
	plt.scatter(dy,dx, c='k', s=3, alpha=0.5, edgecolor='none')
	(ridgex, ridgey) = np.dot(rot, ((ridgepoints[:,0],ridgepoints[:,1])))
	plt.scatter(ridgey-ridgey[k],ridgex-ridgex[k],c='r', edgecolor='none')
	plt.xlim([-4,4])
	plt.ylim([-2,2])
	plt.xlabel('y')
	plt.ylabel('x')

	plt.subplot(2,2,2)
	plt.hist(dy, bins=np.linspace(-4,4,40))
	hist = np.histogram(dy, bins=np.linspace(-4,4,40), normed=False)

	plt.subplot(2,2,3)
	spline = scipy.interpolate.UnivariateSpline(hist[1][1:],hist[0], ext=3, k=3, w = (np.minimum(np.ones(len(hist[0])),1./(np.sqrt(hist[0])))))
	splinederiv= spline.derivative()
	plt.plot(np.linspace(-4,4,40)[1:],hist[0], label='density')
	plt.plot(np.linspace(-4,4,40)[1:],np.gradient(hist[0]), label = 'density deriv')
	plt.plot(np.linspace(-4,4,400)[1:], spline(np.linspace(-4,4,400)[1:]), label = 'spline')
	plt.plot(np.linspace(-4,4,400)[1:], splinederiv(np.linspace(-4,4,400)[1:]), label = 'spline deriv')	
	#plt.legend(fontsize='x-small')

	plt.subplot(2,2,4)
	plt.hist(abs(splinederiv(hist[1][1:])), bins=30)
	stdev = np.std(splinederiv(hist[1][1:]))
	#plt.plot(np.abs(np.diff(hist[0])))
	plt.plot([stdev*3,stdev*3],[0,10])

	#plt.title("p-value = %0.6f"%pvals[k])

# --- depreciated --- #
def plot_subset_sym(sim, ridgepoints, k):
	plt.clf()
	#plot data used for a given ridge point
	plt.subplot(1,2,1,aspect='equal')
	dx, dy = gen_test_set(sim,ridgepoints,k)
	plt.scatter(dx,dy, c='k', s=3, alpha=0.25, edgecolor='none')
	plt.subplot(1,2,2)
	plt.hist(dy, bins=30)
	ax = plt.gca()
	plt.title("p-value = %0.6f"%pvals[k])

# --- depreciated --- #
def plot_subset_mixmdl(k):
	plt.clf()
	#plot data used for a given ridge point
	plt.subplot(2,2,1,aspect='equal')
	dx, dy = gen_test_set(sim,ridgepoints,k)
	plt.scatter(dy,dx, c='k', s=3, alpha=0.25, edgecolor='none')
	plt.xlabel('y')
	plt.ylabel('x')
	plt.subplot(2,2,2)
	plt.hist(dy, bins=30)
	plt.subplot(2,2,3)
	distance = EstMixMdlUniform(dy)
	plt.plot(np.linspace(0,1,200),distance,c='k', label='distance')
	second_der = np.diff(np.diff(distance))
	ratio = np.max(distance)/np.max(second_der)
	plt.plot(np.linspace(0,1,198),second_der*ratio*0.8,c='r', label='2nd derivative')
	alpha_bound = np.sum(distance>(0.6792/np.sqrt(len(dy))))/np.float(200)
	print alpha_bound
	plt.title('95pct bound on alpha: %1.4f'%alpha_bound)

# --- depreciated --- #
def plot_fourier():
	plt.subplot(3,3,1)
	plt.scatter(st_sum_real_abs, st_sum_imag_abs, c='r', edgecolor='none', s=10, alpha = 0.25)
	plt.scatter(sh_sum_real_abs, sh_sum_imag_abs, c='k', edgecolor='none', s=10, alpha = 0.25)	
	plt.ylabel('sum imag abs')
	plt.subplot(3,3,4)
	plt.scatter(st_sum_real_abs, st_sigma_real, c='r', edgecolor='none', s=10, alpha = 0.25)
	plt.scatter(sh_sum_real_abs, sh_sigma_real, c='k', edgecolor='none', s=10, alpha = 0.25)
	plt.ylabel('sigma real')
	plt.subplot(3,3,7)
	plt.scatter(st_sum_real_abs, st_sigma_imag, c='r', edgecolor='none', s=10, alpha = 0.25)
	plt.scatter(sh_sum_real_abs, sh_sigma_imag, c='k', edgecolor='none', s=10, alpha = 0.25)
	plt.xlabel('sum real abs')
	plt.ylabel('sigma imag')

	plt.subplot(3,3,5)
	plt.scatter(st_sum_imag_abs, st_sigma_real, c='r', edgecolor='none', s=10, alpha = 0.25)
	plt.scatter(sh_sum_imag_abs, sh_sigma_real, c='k', edgecolor='none', s=10, alpha = 0.25)
	plt.subplot(3,3,8)
	plt.scatter(st_sum_imag_abs, st_sigma_imag, c='r', edgecolor='none', s=10, alpha = 0.25)
	plt.scatter(sh_sum_imag_abs, sh_sigma_imag, c='k', edgecolor='none', s=10, alpha = 0.25)
	plt.xlabel('sum imag abs')

	plt.subplot(3,3,9)
	plt.scatter(st_sigma_real, st_sigma_imag, c='r', edgecolor='none', s=10, alpha = 0.25)
	plt.scatter(sh_sigma_real, sh_sigma_imag, c='k', edgecolor='none', s=10, alpha = 0.25)
	plt.xlabel('sigma real')


#np.savez(datadir + "snap%i"%snapn + "_ms.npz", startpoints= startpoints, ridgepoints=ridgepoints, pvals=pvals)
#endtime= timeit.time.time()
#print endtime-starttime












