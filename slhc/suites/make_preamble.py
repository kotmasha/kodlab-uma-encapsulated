import cPickle
import sys

def main():
    #prepare file and pickler
    fname=sys.argv[1]+'.pre'
    myf=open(fname,'wb')
    pickler=cPickle.Pickler(myf,protocol=cPickle.HIGHEST_PROTOCOL)

    #construct preamble
    PREAMBLE={}
    PREAMBLE['RUNS']=1
    PREAMBLE['SAMPLE_INITIAL']=0.01
    PREAMBLE['SAMPLE_FINAL']=0.02

    PREAMBLE['METRIC']='elltwo'
    PREAMBLE['MEASURES']=['DHdiff']
    PREAMBLE['PERFORMANCE']=['TPR','cTPR','PPV','cPPV','ACC','cACC']

    PREAMBLE['POISONm']='BridgeBest' #another option: 'BridgeRandom'
    PREAMBLE['POISONd']='ClusterAdjacency'
    PREAMBLE['POISONf']=10

    PREAMBLE['TRUNC']=[pow(0.8,k+1) for k in xrange(10)]
    PREAMBLE['DETECTION_THRESHOLD']=0

    #pickle the preamble
    pickler.dump(PREAMBLE)
    myf.close()

if __name__=="__main__":
    main()
