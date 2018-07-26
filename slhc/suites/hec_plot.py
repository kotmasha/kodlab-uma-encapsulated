############ HC-based Control (HEC)
############ Main simulation module
from __future__ import division
import sys
import cPickle
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

#import matplotlib.patches as mpatches
#import matplotlib.path as mpath

### step-by-step data to cumulative data:
### INPUT: 2D array of numbers
### OUTPUT: numpy array of same dimensions, where column $k$ is the sum of the first $k$ columns
###         of the input array

def cumulative(arr):
    L=len(arr[0])
    return np.array([[0.+sum(row[:ind+1]) for ind in xrange(L)] for row in arr],dtype=float)

### version of division with 0/0=1
my_divide=np.vectorize(lambda x,y: 1. if (x==0 and y==0) else x/y)

### Prepare color maps                                              
colormaps={}
colormaps['TPR']='Greens'
colormaps['cTPR']='Greens'
colormaps['PPV']='Reds'
colormaps['cPPV']='Reds'
colormaps['ACC']='Blues'
colormaps['cACC']='Blues'
colormaps['DHdiff']='Greys'
colormaps['NCdiff']='Purples'
colormaps['ClusterAdjacency']='Greens'
colormaps['WorstCut']='Purples'

def my_cmap(x,key):
    return cm.get_cmap(colormaps[key])(0.2+0.6*x)



### INPUT: $dir_name$

### ASSUMPTIONS: The directory ./$dir_name$/ contains:
###

def main():
    ## Obtain directory name and read preamble
    dir_name=sys.argv[1]
    if sys.platform=='linux2':
        path_name=sys.path[0]+'/'+dir_name+'/'
        path_name_active=path_name+'active/'
        path_name_passive=path_name+'passive/'
    else:
        path_name=sys.path[0]+'\\'+dir_name+'\\'
        path_name_active=path_name+'active\\'
        path_name_passive=path_name+'passive\\'

    PREf=open(path_name+dir_name+'.pre','rb')
    PREAMBLE=cPickle.load(PREf)
    PREf.close()
    #augment the preamble:
    PREAMBLE['active']=path_name_active
    PREAMBLE['passive']=path_name_passive
    #easier access to some constants:
    N_trunc_levels=len(PREAMBLE['TRUNC'])
    N_runs=PREAMBLE['RUNS']

    PLOTTED=PREAMBLE['PERFORMANCE']+PREAMBLE['MEASURES']
    PERFS={} #starting a dictionary for performance measures

    ## Prepare axes
    fig,ax=plt.subplots(len(PLOTTED),2,squeeze=False,sharex=True,sharey='row')
    plt.subplots_adjust(hspace=0.4,wspace=1)
    fig.suptitle('DB performance over time for varying truncation height ('+dir_name+'.dat)',fontsize=10)
    for ind_col,RUN_TYPE in enumerate(['passive','active']):
        for ind_row,meas in enumerate(PLOTTED):
            ax[ind_row][ind_col].set_xlabel('time elapsed')
            ax[ind_row][ind_col].set_ylabel(meas)
            if meas in PREAMBLE['PERFORMANCE']:
                ax[ind_row][ind_col].set_ylim(0,1)
            ax[ind_row][ind_col].tick_params(labelsize=6)
            ax[ind_row][ind_col].set_title('['+RUN_TYPE+' mode, '+meas+']',fontsize=8)

    ## Read the raw data:
    
    # prepare dictionaries
    acceptedQ={}
    poisonQ={}
    detectableQ={}
    samples={}
    measures={}
    detectable_raw={}
    pred_pos_raw={}
    pred_neg_raw={}
    cond_pos_raw={}
    cond_neg_raw={}

    # go through run types:
    for RUN_TYPE in ['passive','active']:
        # for each run type, go over all truncation values:
        for K in xrange(N_trunc_levels):
            # for each truncation value...
            # form the relevant value arrays
            acceptedQ[(RUN_TYPE,K)]=[]
            poisonQ[(RUN_TYPE,K)]=[]
            detectableQ[(RUN_TYPE,K)]=[]
            samples[(RUN_TYPE,K)]=[]
            for meas in PREAMBLE['MEASURES']:
                measures[(RUN_TYPE,meas,K)]=[]

            # then go over the runs to collect the data:
            for N in xrange(N_runs):
                # for each run, open its output file and load the pickled information
                TMPf=open(PREAMBLE[RUN_TYPE]+'_T'+str(K)+'_R'+str(N)+'.out','rb')
                tmp_dict=cPickle.load(TMPf)
                TMPf.close()
                # append information to data arrays
                acceptedQ[(RUN_TYPE,K)].append(tmp_dict['acceptedQ'])
                poisonQ[(RUN_TYPE,K)].append(tmp_dict['poisonQ'])
                detectableQ[(RUN_TYPE,K)].append(tmp_dict['detectableQ'])
                samples[(RUN_TYPE,K)].append(tmp_dict['samples_submitted'])
                for meas in PREAMBLE['MEASURES']:
                    measures[(RUN_TYPE,meas,K)].append(tmp_dict['measures'][meas])
            detectable_raw[(RUN_TYPE,K)]=np.array(
                detectableQ[(RUN_TYPE,K)],dtype=int)
            pred_neg_raw[(RUN_TYPE,K)]=np.array(
                acceptedQ[(RUN_TYPE,K)],dtype=int)
            pred_pos_raw[(RUN_TYPE,K)]=1-np.array(
                acceptedQ[(RUN_TYPE,K)],dtype=int)
            cond_pos_raw[(RUN_TYPE,K)]=np.array(
                poisonQ[(RUN_TYPE,K)],dtype=int)
            cond_neg_raw[(RUN_TYPE,K)]=1-np.array(
                poisonQ[(RUN_TYPE,K)],dtype=int)
            #print cumulative(pred_neg_raw[(RUN_TYPE,K)])
            #print cumulative(pred_pos_raw[(RUN_TYPE,K)]*cond_pos_raw[(RUN_TYPE,K)])
            #print cumulative(pred_pos_raw[(RUN_TYPE,K)])
            #print cumulative(cond_neg_raw[(RUN_TYPE,K)])


    ## Prepare quantities to be plotted:
    MMEAS={}

    # True-Positive Ratios, "Recall"
    # (evolving over time, averaged across runs)
    if 'TPR' in PLOTTED:
        for RUN_TYPE in ['passive','active']:
            for K in xrange(N_trunc_levels):
                MMEAS[(RUN_TYPE,'TPR',K)]=np.mean(my_divide(
                    #true positives (tagged poison pills)
                    cumulative(pred_pos_raw[(RUN_TYPE,K)]*cond_pos_raw[(RUN_TYPE,K)]),
                    #divided by condition positives (total poison pills)
                    cumulative(cond_pos_raw[(RUN_TYPE,K)])
                    ),axis=0)            

    # Positive-Predictive Value, "Precision"
    # (evolving over time, averaged across runs)
    if 'PPV' in PLOTTED:
        for RUN_TYPE in ['passive','active']:
            for K in xrange(N_trunc_levels):
                MMEAS[(RUN_TYPE,'PPV',K)]=np.mean(my_divide(
                    #true positives (tagged poison pills)
                    cumulative(pred_pos_raw[(RUN_TYPE,K)]*cond_pos_raw[(RUN_TYPE,K)]),
                    #divided by predicted positives (total tagged)
                    cumulative(pred_pos_raw[(RUN_TYPE,K)])
                    ),axis=0)

    # Accuracy
    # (evolving over time, averaged across runs)
    if 'ACC' in PLOTTED:
        for RUN_TYPE in ['passive','active']:
            for K in xrange(N_trunc_levels):
                MMEAS[(RUN_TYPE,'ACC',K)]=np.mean(my_divide(
                    # true positives plus true negatives 
                    cumulative(pred_pos_raw[(RUN_TYPE,K)]*cond_pos_raw[(RUN_TYPE,K)]+pred_neg_raw[(RUN_TYPE,K)]*cond_neg_raw[(RUN_TYPE,K)]),
                    # divided by total population
                    cumulative(np.ones_like(pred_pos_raw[(RUN_TYPE,K)]))
                    ),axis=0)

    # CORRECTED True-Positive Ratios, "Recall"
    # (evolving over time, averaged across runs)
    if 'cTPR' in PLOTTED:
        for RUN_TYPE in ['passive','active']:
            for K in xrange(N_trunc_levels):
                MMEAS[(RUN_TYPE,'cTPR',K)]=np.mean(my_divide(
                    cumulative(pred_pos_raw[(RUN_TYPE,K)]*cond_pos_raw[(RUN_TYPE,K)]*detectable_raw[(RUN_TYPE,K)]),
                    cumulative(cond_pos_raw[(RUN_TYPE,K)]*detectable_raw[(RUN_TYPE,K)])
                    ),axis=0)            

    # CORRECTED Positive-Predictive Value, "Precision"
    # (evolving over time, averaged across runs)
    if 'cPPV' in PLOTTED:
        for RUN_TYPE in ['passive','active']:
            for K in xrange(N_trunc_levels):
                MMEAS[(RUN_TYPE,'cPPV',K)]=np.mean(my_divide(
                    cumulative(pred_pos_raw[(RUN_TYPE,K)]*cond_pos_raw[(RUN_TYPE,K)]*detectable_raw[(RUN_TYPE,K)]),
                    cumulative(pred_pos_raw[(RUN_TYPE,K)]*detectable_raw[(RUN_TYPE,K)])
                    ),axis=0)

    # CORRECTED Accuracy
    # (evolving over time, averaged across runs)
    if 'cACC' in PLOTTED:
        for RUN_TYPE in ['passive','active']:
            for K in xrange(N_trunc_levels):
                MMEAS[(RUN_TYPE,'cACC',K)]=np.mean(my_divide(
                    cumulative((pred_pos_raw[(RUN_TYPE,K)]*cond_pos_raw[(RUN_TYPE,K)]+pred_neg_raw[(RUN_TYPE,K)]*cond_neg_raw[(RUN_TYPE,K)])*detectable_raw[(RUN_TYPE,K)]),
                    cumulative(detectable_raw[(RUN_TYPE,K)])
                    ),axis=0)


    # Recorded measurements (evolving over time, averaged across runs)
    for RUN_TYPE in ['passive','active']:
        for K in xrange(N_trunc_levels):
            for meas in PREAMBLE['MEASURES']:
                MMEAS[(RUN_TYPE,meas,K)]=np.mean(measures[(RUN_TYPE,meas,K)],axis=0)

    # Construct the plots
    mmeas_curve={}
    for ind_col,RUN_TYPE in enumerate(['passive','active']):
        for ind_row,meas in enumerate(PLOTTED):
            for K in xrange(N_trunc_levels):
                mycolor=my_cmap((K+0.)/(N_trunc_levels+0.),meas)
                mmeas_curve[(RUN_TYPE,meas,K)]=ax[ind_row][ind_col].plot(
                    MMEAS[(RUN_TYPE,meas,K)],
                    color=mycolor,
                    linewidth=4,
                    label=meas+', trunc='+str(PREAMBLE['TRUNC'][K])
                    )
            ax[ind_row][ind_col].legend(loc='lower right',fontsize=6)

    #show the figure
    plt.show()
    #save the figure
    #with PdfPages(path_name+dir_name+'.pdf') as pdf:
    #    pdf.savefig(fig)

if __name__=='__main__':
    main()
