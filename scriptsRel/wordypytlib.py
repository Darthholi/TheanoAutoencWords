import numpy
from ctypes import * #############changed
import math #############changed
import theano
import theano.tensor as T


class TPreloader(object):
    def __init__(self,pbatch_size,pnum_train,pnum_valid,pnum_test,pnumOutDatas,porigfile,pnuminputwords,pwordmovespeed,pcsvtextcol,poutputcol,poutputismatrix=False):
        #settings:-----------------------------------------------------------
        #we know apriori that the number of data is  408693, lets make it 408690=239*90*19
        self.batch_size = pbatch_size
        self.num_train = pnum_train
        self.num_test = pnum_test
        self.num_valid = pnum_valid
        
        self.id_train=0#consts:
        self.id_valid=self.id_train+self.num_train
        self.id_test=self.id_valid+self.num_valid
        #params
        self.OrigFileIn = porigfile
        self.CSVTextCol = pcsvtextcol
        self.NumInputWords = pnuminputwords
        self.WordMoveSpeed = pwordmovespeed
        self.OutputCol=poutputcol
        self.NumOutputDatas=pnumOutDatas #1
        self.handlein = c_int(-1)#library will tell
        self.totalinputs = c_int(-1)#library will tell
        self.docast=True#True # do cast outputs to 'int32'?
        #consts (dependent on library)
        self.libraryname='wordylib64.dll'
        self.libraryname32='wordylib32.dll'
        self.const_charnumber=26
        self.const_onewordinplen=2*self.const_charnumber
        self.const_lineInSize=self.const_onewordinplen*self.NumInputWords
        self.outputismatrix=poutputismatrix
        
        self.arrsize=self.const_lineInSize*self.batch_size
        self.inarr=numpy.empty((self.batch_size,self.const_lineInSize), dtype=theano.config.floatX, order='C')#row major matrix of minibatch datas - each single data is a single row
        if self.NumOutputDatas>0:
            if (self.NumOutputDatas==1 and not poutputismatrix): #for 1output works use just vectors
                self.outarr=numpy.empty(self.batch_size, dtype=theano.config.floatX)#, order='C') 
            else:
                self.outarr=numpy.empty((self.batch_size,self.NumOutputDatas), dtype=theano.config.floatX, order='C') #and btw it is rows, columns....
        else:
            self.outarr = None#for autoencoder
        self.idarr=numpy.empty(self.batch_size, dtype=c_uint)
        #checking if we can use our library
        if self.inarr.ndim!=2:
            raise Exception('numpy dimension of input matrix is not two')
        if self.inarr.strides[1]!=4:
            print theano.config.floatX
            print self.inarr.strides[1] 
            raise Exception('numpy stride of input matrix in columns is not 4 for float32')
            
        if self.inarr.strides[0]!=4*self.const_lineInSize:
            raise Exception('numpy stride of input matrix in rows is not 4*inputlinesize for float32 (data cleverly aligned? or not rowwise?)')                        
            
        print "one preload batch would be ",self.arrsize, " floats..."
       
        try:
            self.inputlib = cdll.LoadLibrary(self.libraryname)
            print "loading 64bit version"
        except:
            self.inputlib = cdll.LoadLibrary(self.libraryname32)
            print "loading 32bit version"            
        
        self.handlein = self.inputlib.ExternInitNew(c_char_p(self.OrigFileIn),c_uint(self.CSVTextCol),c_uint(self.NumInputWords),c_uint(self.OutputCol),c_uint(self.NumOutputDatas),c_uint(self.WordMoveSpeed))
        self.inputlib.ExternCallback.argtypes = [c_int,c_void_p,c_void_p,c_void_p,c_int,c_int] #print inputlib.ExternCallback.argtypes
        self.totalinputs = self.inputlib.ExternCallback(self.handlein,c_void_p(None),c_void_p(None),c_void_p(None),c_int(0),c_int(0))
        print "total inputs: ", self.totalinputs
        
        self.shared_x=None
        self.shared_y=None #batch...
        self.cast_y=None
        
        self.borrow=True
        
        self.testmode=False
        
        self.careIDs=False
        
        #--------------------------------------------------------------------
    def SetTestMode(self):
        self.testmode=True
        
    def ExportIDNumbers(self):
        self.inputlib.ExternCallback(self.handlein,c_void_p(None),c_void_p(None),c_void_p(None),c_int(0),c_int(1))
        
    def MoveTo(self,nextpos):
        if (not self.testmode):
            self.inputlib.ExternMove(self.handlein,c_int(nextpos))
            
    def load_batch(self):
        #ok we do call this one each batch, but the point is, that if the data is not gonna fit in the memory, it doesnt matter if we load more than one minibatch at a time or not
        if (self.testmode):
            """
            ssum=0
            for i in xrange(self.batch_size):
                for j in xrange(self.const_lineInSize): 
                    self.inarr[j][i]=random.uniform(0.0, 1.0)
            self.inarr=numpy.empty((self.batch_size,self.const_lineInSize), dtype=theano.config.floatX, order='C')#row major matrix of minibatch datas - each single data is a single row
            if self.NumOutputDatas>0:
                if (self.NumOutputDatas==1 and not poutputismatrix): #for 1output works use just vectors
                    self.outarr=numpy.empty(self.batch_size, dtype=theano.config.floatX)#, order='C') 
                else:
                    self.outarr=numpy.empty((self.batch_size,self.NumOutputDatas)
            """
            return
                    
        if self.NumOutputDatas>0:
            outref=self.outarr.ctypes.data_as(c_void_p)
        else:
            outref=c_void_p(None)
        if (self.careIDs):
            idref=self.idarr.ctypes.data_as(c_void_p)
        else:
            idref=c_void_p(None)
        self.inputlib.ExternCallback(self.handlein,self.inarr.ctypes.data_as(c_void_p),outref,idref,c_int(self.batch_size),c_int(0))#byref inarr
            
    def free_me(self):
        self.inputlib.ExternFree(self.handlein)
        try:
            import _ctypes
            if hasattr(_ctypes, 'dlclose'):
                _ctypes.dlclose(self.inputlib._handle)
            else:
                _ctypes.FreeLibrary(self.inputlib._handle)
        except:
            print "cannot free library, but dont worry"        
        print "done"    
    
    #call only after load_batch...
    def getcurX(self,index):             
        return self.inarr
    
    def getcurIDs(self,index):             
        return self.idarr
    
    def getcurY(self,index): #add support for autoencoders
        #print "loading Y"
        #if self.docast:                #returns variable
        #    #dataset.cast_y = T.cast(dataset.shared_y,'int32') #special...
        #    #return dataset.cast_y#[0:dataset.batch_size]
        #    return T.cast(self.shared_y,'int32')
        #else:
        #    return self.shared_y
        
        return self.outarr #returns array

if __name__ == '__main__':
    print "testing wordy preloader"        
    dataset = TPreloader(pbatch_size=239,
      pnum_train=50*19*239,
      pnum_valid=20*19*239,
      pnum_test=20*19*239,
      pnumOutDatas=1,
      porigfile="..\\..\\Data\\doucka\\procords.csv",
      pnuminputwords=64,
      pcsvtextcol=1,
      poutputcol=1)
    
    dataset.MoveTo(400)
    loadbatchgetX(dataset)
    #for i in range(0,100):
    #    print dataset.inarr[0,i], " ",
    for i in range(0,100):
        print dataset.outarr[i], " ",
      
    dataset.free_me()