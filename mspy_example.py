import mspy

###########################################
###########################################
#Generic usage

#create some xy data
rs = np.random.normal(loc=5, size=5000)
phis = np.random.uniform(size=5000)*2*np.pi-np.pi
xs = rs*np.cos(phis)
ys = rs*np.sin(phis)

#create generic meanshift instance
a = mspy.gen_ms()

#initialize data
a.init(x=xs,y=ys)

plt.figure()
plt.subplot(111,aspect='equal')
plt.scatter(a.x,a.y,s=1,alpha=0.2,c='k')

#run meanshift
a.do_meanshift(n_points=1000)

plt.scatter(a.rpx,a.rpy,s=1,alpha=0.5,c='r')

#compute asymmetry
a.get_stats_orth()

#plot results
a.plot_xy()



###########################################
###########################################
#usage with SCF simulations

#initialize an instance of the sim_ms class
data = mspy.sim_ms()
#load simulation snapshot - basically just read_snap()
#data.init('/Users/hendel/Desktop/clouds_archive/rc25/M2.5e+07/L0.20/',60)
data.init('/Users/hendel/Desktop/clouds_archive/rc25/M2.5e+07/L0.90/',60)
#run the SCMS - scale is 1/smoothing length, do_modes will do regular mean shift as well
data.do_meanshift(scale=0.5, n_points=5000, do_modes=False)
#compute the morphology at each point and a global (mean) morphology
data.get_stats_orth()

#typing data. and tabbing [or dir(data)] at the terminal shows the available info:

#In [14]: data.
#data.b                  data.mag_asym           data.seigvec
#data.b_s                data.mag_sym            data.snap
#data.dir                data.mag_tot            data.startpoints
#data.do_meanshift       data.mask_morph         data.te
#data.ep                 data.mean_morph         data.temp
#data.get_stats_fourier  data.meanshift          data.theta
#data.get_stats_orth     data.mu_ratio           data.theta_s
#data.indexes            data.nbods              data.tsnap
#data.init               data.phi                data.tub
#data.l                  data.phi_s              data.vr
#data.l_mag              data.plot_xy            data.vr_s
#data.l_s                data.potex              data.vx
#data.l_x                data.r                  data.vy
#data.l_y                data.r_s                data.vz
#data.l_z                data.rotate             data.x
#data.leigval            data.rpx                data.xydata
#data.leigvec            data.rpy                data.y
#data.m                  data.seigval            data.z

#in particular self.mu_ratio = self.mag_asym/self.mag_tot = mu_i
#and self.mean_morph = np.mean(self.mu_ratio) = mu_S in the paper

#plot the data and ridge points, colored by mu_i
plt.figure()
plt.scatter(data.x, data.y, c='k', alpha=0.1,edgecolor='none', s=2)
plt.scatter(data.rpx, data.rpy, c=data.mu_ratio, alpha=.7, edgecolor='none', s=5, vmin=0, vmax=0.5)
plt.colorbar()
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.xlim([-50,50])
plt.ylim([-50,50])

#save data
np.savez('mspy_stream_example.npz', x=data.x, y=data.y, rpx = data.rpx, rpy = data.rpy, mu_ratio = data.mu_ratio,
	leigvec = data.leigvec, startx = data.xydata[data.startpoints][:,0], starty = data.xydata[data.startpoints][:,1])

#load data
d = np.load('mspy_shell_example.npz')
#they are stored as a dict, you can see what's in it with
d.keys()
#pull out e.g. the x values for convenience
x = d['x']

