import cld_fnc as cf
import os
import gzip
import shutil
from astropy.table import Table
import mspy

dire = '/Users/hendel/Desktop/clouds_archive/'

es = ['rc25', 'rc45', 'rc100']
ms = ['M2.5e+06', 'M2.5e+07', 'M2.5e+08', 'M2.5e+09']
ls = ['L0.05', 'L0.10', 'L0.20', 'L0.30', 'L0.40', 'L0.50', 'L0.60', 
		'L0.70', 'L0.80', 'L0.90', 'L0.95']
snaps = ['SNAP010', 'SNAP020', 'SNAP030', 'SNAP040', 'SNAP050', 
		'SNAP060', 'SNAP070', 'SNAP080', 'SNAP090', 'SNAP100', ]
snapns = [10,20,30,40,50,60,70,80,90,100]

# touch all archive files and unzip them
if 0:
	for a in es:
		for b in ms:
			for c in ls:
				parfile = dire + '/' + a  + '/' + b  + '/' + c  + '/' + 'SCFPAR'
				with gzip.open(parfile+'.z', 'rb') as f_in, open(parfile, 'wb') as f_out:
					shutil.copyfileobj(f_in, f_out)
				for d in snaps:
					snapfile = dire + '/' + a  + '/' + b  + '/' + c  + '/' + d
					print snapfile
					with gzip.open(snapfile+'.z', 'rb') as f_in, open(snapfile, 'wb') as f_out:
						shutil.copyfileobj(f_in, f_out)

scale = 0.5
n_points = 500
sim = mspy.sim_ms()
table = Table(names = ('r_circ','mass','j/j_circ','snap','omega_e','omega_e1','omega_l','alpha','mu',
	'mean_morph', 'mean_morphf', 'mask_morph'), 
	dtype = ('S8','S8','S8','S8','f','f','f','f','f','f','f','f'))

n = float(len(es[0:2])*len(ms[0:2])*len(ls)*len(np.arange(6)+2))
thisn = 0.
for a in es[0:2]:
	for b in ms[0:2]:
		for c in ls:
			for d in np.arange(6)+2:
				snapdir = dire + a  + '/' + b  + '/' + c  + '/'
				snapfile = dire + a  + '/' + b  + '/' + c  + '/' + snaps[d]
				print(snapfile)
				print("%1.2f percent"%((thisn)/n*100.))
				sim.init(snapdir,snapns[d])
				sim.calc_mu(snapdir,snapns[d])
				sim.do_meanshift(scale=scale, n_points=n_points, do_modes=False)
				sim.get_stats_fourier()
				sim.get_stats_orth()
				table.add_row([a,b,c,snaps[d],sim.omegae, sim.omegae1, sim.omegal, sim.alpha, sim.mu,
					sim.mean_morph, sim.mean_morphf, sim.mask_morph])
				thisn=thisn+1



table.write('/Users/hendel/projects/stats/archive_morphs.txt', format= 'ascii')

t= astropy.table.Table()
t = t.read('/Users/hendel/projects/stats/archive_mask_morphs.txt', format= 'ascii')

import matplotlib
import string

##################################################
##################################################
##################################################
# morph comp

cmap = matplotlib.cm.get_cmap('plasma')
colors = np.arange(12)/12.


markersizes=iter([5,20,40])
color = iter(cmap(colors))

for a in es[0:1]:
	markers = iter(['o','s'])
	markersizes=iter([15,30,15])
	for b in ms[0:2]:
		mk = markers.next()
		mks = markersizes.next()
		color = iter(cmap(colors))
		for c in ls[:]:
			col = color.next()
			for d in np.arange(6)+2:
				sel =((t['r_circ']==a) & (t['mass']==b) & (t['j/j_circ']==c))
				if ((b==ms[1])&(d==4)&(a==es[0])):
					plt.scatter(t['mu'][sel], t['mask_morph'][sel], s=mks, marker=mk, color = col, label = 'circularity '+string.split(c,'L')[1])
				else:
					plt.scatter(t['mu'][sel], t['mask_morph'][sel], s=mks, marker=mk, color = col, label=None)



plt.scatter([-20],[-20], c='k', s = 15, label = r'$\rm M = 6.5e+07\ M_\odot$')
plt.scatter([-20],[-20], c='k', s = 30, marker = 's', label = r'$\rm M = 6.5e+08\ M_\odot$')

plt.xlim([0,10])
plt.ylim([0,.45])
plt.xlabel(r'${\rm Semi-analytic\ morphology}\ \mu$')
plt.ylabel(r'${\rm SCUDS\ morphology}\ \mu_S$')
plt.legend()

#plt.savefig('/Users/hendel/projects/stats/paper_figures/morph_comp.png',dpi=200)


##################################################
##################################################
##################################################
# time evolution
cmap = matplotlib.cm.get_cmap('plasma')
colors = np.arange(12)/12.


markersizes=iter([5,20,40])
color = iter(cmap(colors))

for a in es[0:1]:
	markers = iter(['o','s'])
	markersizes=iter([2,4,15])
	for b in ms[0:2]:
		mk = markers.next()
		mks = markersizes.next()
		color = iter(cmap(colors))
		for c in ls[:]:
			col = color.next()
			sel =((t['r_circ']==a) & (t['mass']==b) & (t['j/j_circ']==c))
			if ((b==ms[1])&(a==es[0])):
				plt.plot(0.08*(np.arange(6)*10+20), t['mask_morph'][sel], markersize=mks, marker=mk, color = col, label = 'circularity '+string.split(c,'L')[1])
			else:
				plt.plot(0.08*(np.arange(6)*10+20), t['mask_morph'][sel], markersize=mks, marker=mk, color = col, label='_nolegend_')



plt.scatter([-20],[-20], c='k', s = 15, label = r'${\rm M = 6.5e+07\ M_\odot}$')
plt.scatter([-20],[-20], c='k', s = 30, marker = 's', label = r'${\rm M = 6.5e+08\ M_\odot}$')

plt.xlim([1.5,5.75])
plt.ylim([0,.45])
plt.xlabel(r'${\rm Time\ [Gyr]}$')
plt.ylabel(r'${\rm SCUDS\ morphology}\ \mu_S$')
plt.legend()

#plt.savefig('/Users/hendel/projects/stats/paper_figures/time_evol.png',dpi=200)

plt.savefig('/Users/hendel/projects/stats/paper_figures/timeandmorph.png',dpi=200)









