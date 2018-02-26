from __future__ import division
import sys
import cPickle
import numpy as np
import matplotlib as mpl
#from matplotlib.backends.backend_pdf import PdfPages
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

def get_pickles(infile):
    try:
        while True:
            yield cPickle.load(infile)
    except EOFError:
        pass

filename=sys.argv[1]
batches=['A','B']
files={}
BUA_points={'A':[],'B':[]}
EX_points={'A':[],'B':[]}

## Read the data files
for batch in batches:
    files[batch]=open(filename+batch+'.dat','rb')
    for rt_point,lt_point,EX_point in get_pickles(files[batch]):
        BUA_points[batch].extend([rt_point,lt_point])
        EX_points[batch].append(EX_point)

## Prepare the axes
fig,ax=plt.subplots()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)
fig.suptitle('Duration of a BUA decision cycle as a function of BUA size',fontsize=24)
plt.xlabel('BUA size',fontsize=18)
plt.ylabel('log_10(decision time in milliseconds)',fontsize=18)
## Form the scatter plots

#batchA
xA=np.array([ptx for ptx,pty in BUA_points['A']])
yA=np.log10(1000.*np.array([pty for ptx,pty in BUA_points['A']]))
plt.plot(xA,yA,'.b',alpha=0.05)

fit=np.polyfit(xA,yA,1)
fit_fn=np.poly1d(fit)
plt.plot(xA,fit_fn(xA),'-b',linewidth=3)

xE=np.array([ptx for ptx,pty in EX_points['A']])
yE=np.log10(1000*np.array([pty for ptx,pty in EX_points['A']]))
plt.plot(xE,yE,'.c',alpha=0.05)

fit=np.polyfit(xE,yE,1)
fit_fn=np.poly1d(fit)
plt.plot(xE,fit_fn(xE),'-c')

#batchB
xA=np.array([ptx for ptx,pty in BUA_points['B']])
yA=np.log10(1000.*np.array([pty for ptx,pty in BUA_points['B']]))
plt.plot(xA,yA,'.r',alpha=0.05)

fit=np.polyfit(xA,yA,1)
fit_fn=np.poly1d(fit)
plt.plot(xA,fit_fn(xA),'-r',linewidth=3)

xE=np.array([ptx for ptx,pty in EX_points['B']])
yE=np.log10(1000*np.array([pty for ptx,pty in EX_points['B']]))
plt.plot(xE,yE,'.m',alpha=0.05)

fit=np.polyfit(xE,yE,1)
fit_fn=np.poly1d(fit)
plt.plot(xE,fit_fn(xE),'-m')



plt.show()