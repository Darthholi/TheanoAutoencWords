"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from wordylogreg import LogisticRegression
from fulllayer import *
from wordypytlib import *
from convolutional_mlpwordy import *
import gzip


# start-snippet-1
class NNmaster(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        n_ins,
        pretrain_number=0,
        L1reg=0.00,
        L2reg=0.0001
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        #definition
        self.L1_reg=L1reg
        self.L2_reg=L2reg
        self.n_layers=0
        self.n_ins=n_ins
        self.layerclasses = []
        self.hiddenunits = []
        self.lparams = []
        
        self.pretrain_number=pretrain_number
        
        #actual 'state' variables build at "makeNN"
        self.layers = []
        self.params = []
        #self.n_layers = len(hidden_layers_sizes)

        #assert self.n_layers > 0
    def addLayer(self,layerclass, numneurons, kwargs):             #todo in future add other parameters ...
        self.layerclasses.append(layerclass)
        self.hiddenunits.append(numneurons)
        self.lparams.append(kwargs)
        self.n_layers+=1

    def makeNN(self, numpy_rng, theano_rng=None, batch_size=1):                                     #todo also type of output doesnt need to be int and such...
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        #self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        #will be set by last layer
        #self.y = T.matrix('y')
                                         
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        self.layers = []
        layer_input = self.x
        
        self.L1 = 0
        self.L2_sqr = 0
        
        print "creating number of inputs - ",self.n_ins 
        input_size=self.n_ins
        
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            
            layer_size = self.hiddenunits[i] 

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer

            print "creating layer in:",input_size, "layer:",layer_size," ",self.lparams[i]
            newlayer = self.layerclasses[i](numpy_rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_layer=layer_size,
                                        batch_size=batch_size,
                                        **self.lparams[i]
                                        )
            # add the layer to our list of layers
            self.layers.append(newlayer)
            layer_input = newlayer.output#for next one
            input_size=newlayer.n_out #for next one ... the layer can processit anyhow it wishes
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(newlayer.params)
            self.L1 = self.L1 + newlayer.L1regnorm() #newlayer.W.sum()
            self.L2_sqr = self.L2_sqr + newlayer.L2regnorm() # (newlayer.W ** 2).sum()
        
        #measures from last layer - the errors and the cost
        self.y=self.layers[self.n_layers-1].classforY()
        
        self.errors=self.layers[self.n_layers-1].errors(self.y)    
        self.finetune_cost = (self.layers[self.n_layers-1].getFinetuneCost(self.y) #classifier.negative_log_likelihood(y)
                              + self.L1_reg * self.L1
                              + self.L2_reg * self.L2_sqr)

        """
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
        """

    def saveNN(self,file):                                                   
        #np.savez_compressed(file, *[param.get_value(borrow=True) for param in self.params])  #np.savez(tmp, *[getarray[i] for i in range(10)])
        savef=open(file,'wb')     
        
        numpy.save(savef,self.n_layers)
        numpy.save(savef,self.n_ins)
        for i in xrange (self.n_layers):
            numpy.save(savef,str(self.layerclasses[i]))
            numpy.save(savef,self.hiddenunits[i])
            numpy.save(savef,str(self.lparams[i]))      
           
        for param in self.params:             
            numpy.save(savef,param.get_value(borrow=True))
        savef.close()
                    
    def loadNN(self,file, numpy_rng, theano_rng=None, batch_size=1):                   #todo - add actually loading the network.
        
        savef=open(file,'rb')       
        
        do_loadArch = False     
        if do_loadArch:
            self.n_layers = numpy.load(savef)
            self.n_ins = numpy.load(savef)
            self.layerclasses = []#clear
            self.hiddenunits = []
            self.lparams = []
            self.layers = []
            self.params = []
            for i in xrange (self.n_layers):
                self.layerclasses.append(numpy.load(savef))
                self.hiddenunits.append(numpy.load(savef))
                self.lparams.append(numpy.load(savef))
                same=i
            self.makeNN(numpy_rng,theano_rng,batch_size)
        else:                                                   #todo - tohle je jenom takovej checking, realny nacitani by se muselo delat jinak
            print "checkloadnn"
            if self.n_layers != numpy.load(savef): 
                print "different number of layers"
            if self.n_ins != numpy.load(savef):
                print "different number of n_ins"
            same=self.n_layers
            for i in xrange (self.n_layers):
                if str(self.layerclasses[i]) != numpy.load(savef):
                    print "different layerclass"
                    same=i
                if str(self.hiddenunits[i]) != str(numpy.load(savef)):
                    print "different number of hiddenunits"
                    same=i
                if str(self.lparams[i]) != numpy.load(savef):
                    print "differentactivation finction"
                    same=i   
        
        #loadf.seek(0)
        #print numpy.load('pokus.bin.npy')      
         
        numdo=0
        for param in self.params:  
            try:
                loaded=numpy.load(savef)
                param.set_value(loaded,borrow=True)
                numdo+=1
                if numdo>=same:
                    break
            except:
                print "e"
        savef.close()
        print numdo, "layers loaded"
    
    def savePretrainState(self,file,idlayer):
        numpy.savez(file,self.layers[idlayer].get_pretrain_params().get_value(borrow=True))
        
    def loadPretrainState(self,file,idlayer):  #funguje nebo to neni pointer??
        self.layers[idlayer].get_pretrain_params().set_value(numpy.loadz(file),borrow=True)

    def transf_input(self,tolayer):           #function that just tells me the outputs for a layer given a batch
        fn = theano.function(
                inputs=[self.x],
                outputs=self.layers[tolayer].output
                #allow_input_downcast=True
            )
        return fn

    def pretraining_functions(self):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        #index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        contraction_level = T.scalar('contraction')
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        #batch_begin = index * batch_size
        # ending of a batch given `index`
        #batc/h_end = batch_begin + batch_size
        
        #prerain_fns[i] pretrains ONLY the layer [i] 

        pretrain_fns = []     
        for lri in xrange(len(self.layers)):            
            if self.pretrain_number<=lri:
                break
            layer = self.layers[lri]
            if not layer.canPretrain():
                self.pretrain_number=lri #if someone set greaters
                break
            # get the cost and the updates list
            cost, updates = layer.get_cost_updates_pretrain(learning_rate=learning_rate,
                                                            corruption_level=corruption_level,
                                                            contraction_level=contraction_level)
            if cost!=None and updates!=None:                           
                # compile the theano function
                fn = theano.function(
                    inputs=[self.x,
                        #index,
                        theano.Param(corruption_level, default=0.2),
                        theano.Param(learning_rate, default=0.1), #todo change defaults to zero?
                        theano.Param(contraction_level, default=0.0)
                    ],
                    outputs=cost,
                    updates=updates,
                    allow_input_downcast=True
                )
                # append `fn` to the list of functions
                pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, dataset, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        

        # compute number of minibatches for training, validation and testing
        #n_train_batches = dataset.num_train / dataset.batch_size
        n_valid_batches = dataset.num_valid / dataset.batch_size
        n_test_batches = dataset.num_test / dataset.batch_size
        

        #index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        #print "making trainfn"
        train_fn = theano.function(
            inputs=[self.x,self.y],#<- train set 
            outputs=self.finetune_cost,
            updates=updates,
            name='train',
            allow_input_downcast=True
        )

        #print "making test score"
        #print self.errors
        test_score_i = theano.function(
            [self.x,self.y],         #<- test set
            self.errors,
            name='test',
            allow_input_downcast=True
        )

        #print "making valid score"
        valid_score_i = theano.function(
            [self.x,self.y],#<- valid set
            self.errors,
            name='valid',
            allow_input_downcast=True
        )

        # Create a function that scans the entire validation set
        def valid_score(dataset):
            dataset.MoveTo(dataset.id_valid)#next getdata will go over training dataset
            validation_array = []
            for i in xrange(n_valid_batches):
                dataset.load_batch()#dataset.id_valid+i*n_valid_batches)
                validation_array.append(valid_score_i(dataset.getcurX(i),dataset.getcurY(i)))      
            return validation_array
              
        # Create a function that scans the entire test set
        def test_score(dataset):
            dataset.MoveTo(dataset.id_test)#next getdata will go over training dataset
            test_array = []
            for i in xrange(n_test_batches):
                dataset.load_batch()#dataset.id_valid+i*n_valid_batches)
                test_array.append(test_score_i(dataset.getcurX(i),dataset.getcurY(i)))   
            return test_array

        return train_fn, valid_score, test_score

"""
----------------------------------------------------------------------
----------------------------------------------------------------------
"""

def do_nn(
    name,
    finetune_lr,
    pretraining_epochs,
    corruption_levels,
    contraction_levels,
    pretrain_lr,
    training_epochs,
    batch_size, 
    pnum_train,
    pnum_valid,
    pnum_test,
    pnumOutDatas,
    porigfile,
    pnuminputwords,
    pwordmovespeed, #how fast we move the input window.
    pcsvtextcol,
    poutputcol,    
    poutputismatrix,#affected by last layer and .y settings
    ppretrain_number,#dont pretrain convolutions
    pL1reg,
    pL2reg,
    layerstooutput, #automatical, to empty set to = []
    architecture,
    writefullinput=False
    ):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    
    #total: dbpedia: 24828677 >= 100*6 *41381
    #lets take just a tiny bit of it, not everything
    #
    # 1: the sizes are for all the data and thats not the number of paragraphs
    # 2: text col is indexed by 1, outputcol is indexed by 0
    # 3: for classification it is not good to put there something without rounding (will predict average) -> so logregression OR rounding
    # 4: poutputismatrix says if the output data format IS matrix or vector 

    dataset = TPreloader(pbatch_size=batch_size,
      pnum_train=pnum_train,
      pnum_valid=pnum_valid,
      pnum_test=pnum_test,
      pnumOutDatas=pnumOutDatas,
      porigfile=porigfile,
      pnuminputwords=pnuminputwords,
      pwordmovespeed=pwordmovespeed,
      pcsvtextcol=pcsvtextcol,
      poutputcol=poutputcol,
      poutputismatrix=poutputismatrix)
    print dataset.id_train
    print dataset.id_valid
    print dataset.id_test
    #todo here and to the constructur lest put the decisions about the dataset size, num train,test,valid and batchsize. Now it is just in constructor
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = dataset.num_train / dataset.batch_size
    #n_valid_batches = dataset.num_valid / dataset.batch_size
    #n_test_batches = dataset.num_test / dataset.batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    net = NNmaster(
        n_ins=dataset.const_lineInSize,
        L1reg=pL1reg,
        L2reg=pL2reg,
        pretrain_number=ppretrain_number #how many layers will we pretrain?
    )
    
    for inda in architecture:
        net.addLayer(*inda)    
    
    net.makeNN(batch_size=dataset.batch_size,numpy_rng=numpy_rng)#batchsize needed to be given because jacobian needs to know somewhere for some layer   
    
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    
    layerstart=0
    epochstart=0
    finetuneepostart=0
    
    try:
        pretrainind=open('./pretrain.bin', 'r') #how far are we in pretraining
        layerstart, epochstart, finetuneepostart = [int(x) for x in pretrainind.readline().split()] 
        epochstart=epochstart+1 #because this is saved just where we last FINISHED, so now we need to continue on the next (and maybe that means that we will get to next layer, but that will decide the forcycle)
        pretrainind.close()
        
        net.loadNN(name+'nnsaved.bin',batch_size=dataset.batch_size,numpy_rng=numpy_rng)               #do we have neuralnetwork?                                                
        
        try:
            net.loadPretrainState('nnlastpretrain',layerstart)
        except:
            print "no last pretrain state"  
        
    except:
        e = sys.exc_info()
        print e
        layerstart=0
        epochstart=0
        finetuneepostart=0
        print "first time pretrainning"
    
    
    if net.pretrain_number>0:
    
        print '... getting the pretraining functions'
        pretraining_fns = net.pretraining_functions()

        print '... pre-training the model'
        print "number of batches in one epoch will be ",n_train_batches
        print "one batch is ",dataset.batch_size, " rows"
        print "each row is ",dataset.const_lineInSize," floats"
        print "pretraining epochs specified ",pretraining_epochs
        print "number of layers to pretrain ",net.pretrain_number
        print "total number of layers ",net.n_layers       
        
        print "starting from layer", layerstart
        print "starting from epoch", epochstart
        
        start_time = time.clock()
        ## Pre-train layer-wise
        
        for i in xrange(layerstart,net.pretrain_number):
            # go through pretraining epochs
            print "----------------------------------------------------------"
            print "pretraining layer", i          
            startthislayer=time.clock()
            if (i>layerstart):
                ceta=(startthislayer-start_time)/(i-layerstart) * (net.pretrain_number-layerstart)
                print " (pretrainETA ","{:10.2f}".format(ceta),") "
            
            if i>=len(pretraining_epochs):
                thispretrain=pretraining_epochs[-1]
            else:
                thispretrain=pretraining_epochs[i]
            
            for epoch in xrange(epochstart,thispretrain):
                # go through the training set
                
                print "-----"
                print "pretrain epoch ", epoch, " corruption ",corruption_levels[i], " contraction ",contraction_levels[i]
                startthisepoch=time.clock()
                if (epoch>epochstart):
                    ceta=(startthisepoch-startthislayer)/(epoch-epochstart) * (thispretrain-epochstart)
                    print " (epoETA ","{:10.2f}".format(ceta),") "
                
                c = []
                dataset.MoveTo(dataset.id_train)#next getdata will go over training dataset
                for batch_index in xrange(n_train_batches):   
                    thisbatchstart=time.clock()             
                    dataset.load_batch()
                    c.append(pretraining_fns[i](x=dataset.getcurX(batch_index),
                             corruption=corruption_levels[i],
                             #contraction=contraction_levels[i],
                             lr=pretrain_lr))
                    if batch_index<5 or (batch_index % (math.floor(n_train_batches/10)))==0 :
                        print "batch ", batch_index, " start ","{:10.2f}".format(thisbatchstart),
                        ceta=(time.clock()-startthisepoch)/(batch_index+1) * n_train_batches
                        print " (ETA ","{:5.2f}".format(ceta),") ",
                        print "cost was ",c[batch_index]         
                    
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
                
                print "saving NN"
                pretrainind=open('./pretrain.bin', 'w')
                pretrainind.write('%d %d %d' % (i,epoch,finetuneepostart))
                pretrainind.close()
                net.savePretrainState('nnlastpretrain',i)
                net.saveNN(name+'nnsaved.bin')
            epochstart=0 
    
        end_time = time.clock()
    
        print "end pretrain"
        print "--------------------------------------------------------"
    
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################
    
    done_finetune = finetuneepostart >= training_epochs #false to not skip finetune
    best_iter=0
    
    if not done_finetune:

        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = net.build_finetune_functions(
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=finetune_lr
        )
        
        #theano.printing.pydotprint(net.finetune_cost, outfile="theanograph.png", var_with_name_simple=True)
        #theano.printing.debugprint(net.finetune_cost) 
    
        print '... finetunning the model'
        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_validation_loss = numpy.inf
        beft_iter = 0
        test_score = 0.
        start_time = time.clock()
    
        
        epoch = finetuneepostart
        
        print "starting at epoch ",epoch
    
        while (epoch < training_epochs) and (not done_finetune):
            epoch = epoch + 1
            
            print "-------------------------------------------------------------"
            print "epoch ", epoch
            startthisepoch=time.clock()
            if (epoch>1):
                ceta=(startthisepoch-start_time)/(epoch-finetuneepostart) * (training_epochs-finetuneepostart)
                print " (epoETA ","{:10.2f}".format(ceta),") "
            
            trainavgs = []
            for minibatch_index in xrange(n_train_batches):
                thisbatchstart=time.clock()                
                dataset.MoveTo(dataset.id_train+minibatch_index*batch_size)                
                dataset.load_batch()#minibatch_index)
                #print dataset.getcurY(minibatch_index).shape[0], dataset.getcurY(minibatch_index).shape[1]
                minibatch_avg_cost = train_fn(dataset.getcurX(minibatch_index),dataset.getcurY(minibatch_index))
                trainavgs.append(minibatch_avg_cost)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                
                if minibatch_index<5 or (minibatch_index % (math.floor(n_train_batches/10)))==0 :
                    print "batch ", minibatch_index, " start ","{:10.2f}".format(thisbatchstart),
                    ceta=(time.clock()-startthisepoch)/(minibatch_index+1) * n_train_batches
                    print " (ETA ","{:5.2f}".format(ceta),") ",
                    print "cost was ",minibatch_avg_cost
    
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model(dataset)
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, train-avg-cost %f, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                          numpy.mean(trainavgs),
                          this_validation_loss * 100.))
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                        # test it on the test set
                        test_losses = test_model(dataset)
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
                    print "saving NN..."
                    pretrainind=open('./pretrain.bin', 'w')
                    pretrainind.write('%d %d %d' % (net.n_layers,0,epoch))
                    pretrainind.close()
                    net.saveNN(name+'nnsaved.bin')
              
                if patience <= iter:
                    epoch = training_epochs
                    done_finetune = True
                    #break
                
                if done_finetune:
                    break

        end_time = time.clock()
        print(
              (
                  'Optimization complete with best validation score of %f %%, '
                  'on iteration %i, '
                  'with test performance %f %%'
              )
              % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        print >> sys.stderr, ('The training code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    #---------------------------------------------------------
    #print the outputs for another processing:
    #format:
    #"idsetence","typ - 0 train 1 valid 2 test","right outputs...","computed outputs..."
    dataset.careIDs=True #dataset.ExportIDNumbers()
    if layerstooutput==None:
        layerstooutput= [net.n_layers-1] #,net.n_layers-2] #last layer and the one before
    for lnum in layerstooutput:
        final = net.transf_input(lnum)
        filenm=name+'out'+str(lnum)+'.csv.gz'
        
        #zf = zipfile.ZipFile('./out'+str(lnum)+'.csv.zip', 'w',                     compression=zipfile.ZIP_DEFLATED)

        finalout=gzip.open(filenm, 'wb')
        print "transforming input - writing out",lnum,".csv"    
  
        dataset.MoveTo(0)#next getdata will go over training dataset
        
        if writefullinput:
            xtotal=dataset.totalinputs/batch_size
        else:
            xtotal=pnum_train+pnum_valid+pnum_test     
        
        for i in xrange():
          dataset.load_batch()
          outbatch=final(dataset.getcurX(i))
          if dataset.NumOutputDatas>0:
              outreal=dataset.getcurY(i)
          idsbatch=dataset.getcurIDs(i)
          for j in xrange(batch_size):      #data in one batch     = outbatch.size
          #now just to tell which input are we transforming:
              dataid=i*batch_size+j
              if dataid<dataset.id_valid and dataid<dataset.id_test:
                  typeset=0 #0 train 1 valid 2 test
              else:
                  if dataset.id_valid>dataset.id_test:
                      if dataid>dataset.id_valid:
                          typeset=1
                      else:
                          typeset=2
                  else:
                      if dataid>dataset.id_test:
                          typeset=2
                      else:
                          typeset=1
              finalout.write(str(idsbatch[j]))
              finalout.write(", ")
              finalout.write(str(typeset))
              finalout.write(", ")
              
              #write what it should compute
              if dataset.NumOutputDatas>0:
                  if dataset.outputismatrix or dataset.NumOutputDatas>1:
                      for k in xrange(dataset.NumOutputDatas): 
                          finalout.write(str(outreal[j][k])) #todo right order??
                          finalout.write(", ")
                  else:
                      for k in xrange(dataset.NumOutputDatas): 
                          finalout.write(str(outreal[j]))
                          finalout.write(", ")                      
              
              #write what it computed
              layouts=net.layers[lnum].n_out# number of output numbers
              if layouts==1:
                  finalout.write(str(outbatch[j]))
              else:
                  for k in xrange(layouts):              #dimension of output
                      finalout.write(str(outbatch[j][k]))
                      if k!=layouts-1:
                          finalout.write(", ")
              finalout.write("\n")  
        finalout.close()        
        print "done writing"
              
    dataset.free_me()

if __name__ == '__main__':
    
    def do_dbpedia1250_cleverspeed_neworder():#
        pnuminputwords=52                     #shuffled and appended #560000sentences training #70000testing(&valid?)
        do_nn(                                #resulting number of nn-inputs (to 52size and speed 52-5): 916935  
        name='dbpedia1250all',                
        finetune_lr=0.1,
        pretraining_epochs=[20],
        corruption_levels = [.0, .0, .0,.0, .0, .0,.0, .0, .0],
        contraction_levels = [.00, .00, .00, .00, .00, .00, .00, .00, .00, .00],
        pretrain_lr=0.001,
        training_epochs=1000,
        
        #kratsi pro zkousku: 4*5*19*19 * 127   -> snizim na 126, protoze jsme to puvodne zvysili
        batch_size=19*19,
        pnum_train=90*4*5*19*19,
        pnum_valid=33*4*5*19*19,
        pnum_test=33*4*5*19*19,
        #megakratky
        #batch_size=363,
        #pnum_train=23*363,        
        #pnum_valid=2*363,         
        #pnum_test=2*363, 
        pnumOutDatas=1,
        porigfile="all1250.csv",
        pnuminputwords=pnuminputwords,
        pwordmovespeed=pnuminputwords-5, #->916935 inputs
        
        pcsvtextcol=2,
        poutputcol=0,
        poutputismatrix=False,#affected by last layer and .y settings
        ppretrain_number=0,
        pL1reg=0.00,
        pL2reg=0.0001,
        layerstooutput= None, #automatical, to empty set to = []
        architecture=[
        (ReshaperManip,(1,pnuminputwords,52), {"activation": None}),    #always (1,pnuminputwords,52)
        (LeNetConvLayer,(20,26,5),{"activation": None}), #20 kernels, filter 27x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 13x24
        (LeNetConvLayer,(30,6,5),{"activation": None}), #30 kernels, filter 6x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 4x10
        (LeNetConvLayer,(40,4,5),{"activation": None}), #20 kernels, filter 4x5
        (LeNetPoolLayer,(1,2),{"activation": T.tanh}), #downsample to half the size, now we are at 1x3
        (BaseLayer, 500, {"activation": T.nnet.sigmoid}),
        (LogisticRegression, 15, {"activation": None}), #1-14 ... let him ignore 0 ... predicting 0 is wrong ... always      
        ],
        writefullinput=False
        )
        
    def do_dbpedia1250_cleverspeed_big1():         #560000sentences training #70000testing(&valid?)
        pnuminputwords=52                     #shuffled and appended #resulting number of nn-inputs (to 52size and speed 52-5): 916935  ~ 916940 = 4*5*19*19 * 127
        do_nn(                                #resulting number of nn-inputs (to 52size and speed 1): 27991293 = (3 * 11 * 11 * 29 * 2659) = 2659*29*363  
        name='dbpedia1250allbig1',                
        finetune_lr=0.1,
        pretraining_epochs=[20],
        corruption_levels = [.0, .0, .0,.0, .0, .0,.0, .0, .0],
        contraction_levels = [.00, .00, .00, .00, .00, .00, .00, .00, .00, .00],
        pretrain_lr=0.001,
        training_epochs=1000,
        
        #kratsi pro zkousku: 4*5*19*19 * 127   -> snizim na 126, protoze jsme to puvodne zvysili
        batch_size=19*19,
        pnum_train=90*4*5*19*19,
        pnum_valid=33*4*5*19*19,
        pnum_test=33*4*5*19*19,
        #megakratky
        #batch_size=363,
        #pnum_train=23*363,        #celkem fajn, validacni a testovaci chyba klesa z 89% na 69% v prubehu trenovani
        #pnum_valid=2*363,         #finalne - valid 67.21% a test 65.28%  - without normalization of inputs
        #pnum_test=2*363,          #fail - 90%, 92%               - with normalization of inpus 
        pnumOutDatas=1,                         #zigzag nenormalizovanej je pak 100% chyba....
        porigfile="all1250.csv",
        pnuminputwords=pnuminputwords,
        pwordmovespeed=pnuminputwords-5, # speed 1 -> inputs 27991293 = (3 * 11 * 11 * 29 * 2659) = 2659*29*363
        
        pcsvtextcol=2,
        poutputcol=0,
        poutputismatrix=False,#affected by last layer and .y settings
        ppretrain_number=0,
        pL1reg=0.00,
        pL2reg=0.0001,
        layerstooutput= None, #automatical, to empty set to = []
        architecture=[
        (ReshaperManip,(1,pnuminputwords,52), {"activation": None}),    #always (1,pnuminputwords,52)
        (LeNetConvLayer,(30,5,5),{"activation": None}), #20 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 24x24
        (LeNetConvLayer,(40,5,5),{"activation": None}), #30 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 10x10
        (LeNetConvLayer,(50,5,5),{"activation": None}), #20 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 3x3
        (BaseLayer, 1000, {"activation": T.nnet.sigmoid}),
        (LogisticRegression, 15, {"activation": None}), #1-14 ... let him ignore 0 ... predicting 0 is wrong ... always      
        ],
        writefullinput=False
        )
    
    def do_dbpedia1250_cleverspeed():         #560000sentences training #70000testing(&valid?)
        pnuminputwords=52                     #shuffled and appended #resulting number of nn-inputs (to 52size and speed 52-5): 916935  ~ 916940 = 4*5*19*19 * 127
        do_nn(                                #resulting number of nn-inputs (to 52size and speed 1): 27991293 = (3 * 11 * 11 * 29 * 2659) = 2659*29*363  
        name='dbpedia1250all',                
        finetune_lr=0.1,
        pretraining_epochs=[20],
        corruption_levels = [.0, .0, .0,.0, .0, .0,.0, .0, .0],
        contraction_levels = [.00, .00, .00, .00, .00, .00, .00, .00, .00, .00],
        pretrain_lr=0.001,
        training_epochs=1000,
        
        #kratsi pro zkousku: 4*5*19*19 * 127   -> snizim na 126, protoze jsme to puvodne zvysili
        batch_size=19*19,
        pnum_train=90*4*5*19*19,
        pnum_valid=33*4*5*19*19,
        pnum_test=33*4*5*19*19,
        #megakratky
        #batch_size=363,
        #pnum_train=23*363,        #celkem fajn, validacni a testovaci chyba klesa z 89% na 69% v prubehu trenovani
        #pnum_valid=2*363,         #finalne - valid 67.21% a test 65.28%  - without normalization of inputs
        #pnum_test=2*363,          #fail - 90%, 92%               - with normalization of inpus 
        pnumOutDatas=1,                         #zigzag nenormalizovanej je pak 100% chyba....
        porigfile="all1250.csv",
        pnuminputwords=pnuminputwords,
        pwordmovespeed=pnuminputwords-5, # speed 1 -> inputs 27991293 = (3 * 11 * 11 * 29 * 2659) = 2659*29*363
        
        pcsvtextcol=2,
        poutputcol=0,
        poutputismatrix=False,#affected by last layer and .y settings
        ppretrain_number=0,
        pL1reg=0.00,
        pL2reg=0.0001,
        layerstooutput= None, #automatical, to empty set to = []
        architecture=[
        (ReshaperManip,(1,pnuminputwords,52), {"activation": None}),    #always (1,pnuminputwords,52)
        (LeNetConvLayer,(20,5,5),{"activation": None}), #20 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 24x24
        (LeNetConvLayer,(30,5,5),{"activation": None}), #30 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 10x10
        (LeNetConvLayer,(20,5,5),{"activation": None}), #20 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 3x3
        (BaseLayer, 500, {"activation": T.nnet.sigmoid}),
        (LogisticRegression, 15, {"activation": None}), #1-14 ... let him ignore 0 ... predicting 0 is wrong ... always      
        ],
        writefullinput=False
        )
    
    def do_doucka1250_try16autoenc():
    # protoze autoencodery davaji cost zapornou pouze pro prvni layer tak si to proste necham nakreslit v prvnim layeru pouze.
    # zkusime jak to vypada A jak to klasifikuje
    # aha takze output skoro vsude 1.0,1.0 a odhaduje nulou 
    #napady: normalizovat input
    #napady: zkontrolovat autoencoder podle ceho porovnava tu cenu?
    #vysledky s normaliz inputem a DRUHOU COLUMN a 16 jednotkama: valid 30.06% test 23.61%
        pnuminputwords=52                
        do_nn(                           
        name='doucka1250-autenc16',
        finetune_lr=0.1,
        pretraining_epochs=[20,20,40,60],
        corruption_levels = [.0, .0, .0,.0, .0, .0,.0, .0, .0],
        contraction_levels = [.01, .02, .03, .03, .03, .03, .03, .03, .03, .03],
        pretrain_lr=0.001,
        training_epochs=1000,
    
        batch_size=160, #pro speed 52-5 je celkem inputu pro nn 24320, takze 24320=160*19*8 
        pnum_train=10*8*160,
        pnum_valid=5*8*160,
        pnum_test=4*8*160,
        pnumOutDatas=1,
        porigfile="procords1250.csv",
        pnuminputwords=pnuminputwords,
        pwordmovespeed=pnuminputwords-5, #how fast we move the input window.
        pcsvtextcol=1,
        poutputcol=2,
        
        poutputismatrix=False,#affected by last layer and .y settings
        ppretrain_number=1,
        pL1reg=0.00,
        pL2reg=0.0001,
        layerstooutput= [0,1], 
        architecture=[
        (BaseLayer, 16, {"activation": T.nnet.sigmoid}),
        (LogisticRegression, 2, {"activation": None}),      
        ]
        )
    
    def do_doucka1250_try2autoenc():
    # protoze autoencodery davaji cost zapornou pouze pro prvni layer tak si to proste necham nakreslit v prvnim layeru pouze.
    # zkusime jak to vypada A jak to klasifikuje
    # aha takze output skoro vsude 1.0,1.0 a odhaduje nulou 
    #napady: normalizovat input
    #napady: zkontrolovat autoencoder podle ceho porovnava tu cenu?
        pnuminputwords=52                
        do_nn(                           
        name='doucka1250-autenc2',
        finetune_lr=0.1,
        pretraining_epochs=[20,20,40,60],
        corruption_levels = [.0, .0, .0,.0, .0, .0,.0, .0, .0],
        contraction_levels = [.01, .02, .03, .03, .03, .03, .03, .03, .03, .03],
        pretrain_lr=0.001,
        training_epochs=1000,
    
        batch_size=160, #pro speed 52-5 je celkem inputu pro nn 24320, takze 24320=160*19*8 
        pnum_train=10*8*160,
        pnum_valid=5*8*160,
        pnum_test=4*8*160,
        pnumOutDatas=1,
        porigfile="procords1250.csv",
        pnuminputwords=pnuminputwords,
        pwordmovespeed=pnuminputwords-5, #how fast we move the input window.
        pcsvtextcol=1,
        poutputcol=1,
        
        poutputismatrix=False,#affected by last layer and .y settings
        ppretrain_number=1,
        pL1reg=0.00,
        pL2reg=0.0001,
        layerstooutput= [0,1], 
        architecture=[
        (BaseLayer, 2, {"activation": T.nnet.sigmoid}),
        (LogisticRegression, 2, {"activation": None}),      
        ]
        )
    
    def do_doucka1250_cleverspeed():
        pnuminputwords=52                # -> if output samy nuly, tak :24.5% valid error, 11 epochs #28.43%test error
        do_nn(                           #74 min, valid 24.50/6% test 28.43% -> zase samy nuly!!!
        name='doucka1250-5x27-500',
        finetune_lr=0.1,
        pretraining_epochs=[12,20,40,60],
        corruption_levels = [.0, .0, .0,.0, .0, .0,.0, .0, .0],
        contraction_levels = [.01, .02, .03, .03, .03, .03, .03, .03, .03, .03],
        pretrain_lr=0.001,
        training_epochs=1000,
    
        batch_size=160, #pro speed 52-5 je celkem inputu pro nn 24320, takze 24320=160*19*8 
        pnum_train=10*8*160,
        pnum_valid=5*8*160,
        pnum_test=4*8*160,
        pnumOutDatas=1,
        porigfile="procords1250.csv",
        pnuminputwords=pnuminputwords,
        pwordmovespeed=pnuminputwords-5, #how fast we move the input window.
        pcsvtextcol=1,
        poutputcol=1,
        
        poutputismatrix=False,#affected by last layer and .y settings
        ppretrain_number=0,#dont pretrain convolutions
        pL1reg=0.00,
        pL2reg=0.0001,
        layerstooutput= None, #automatical, to empty set to = []
        architecture=[
        (ReshaperManip,(1,pnuminputwords,52), {"activation": None}),    #always (1,pnuminputwords,52)
        (LeNetConvLayer,(20,5,27),{"activation": None}), #20 kernels, filter 5x27
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 24x13
        (LeNetConvLayer,(30,5,6),{"activation": None}), #30 kernels, filter 5x6
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 10x4
        (LeNetConvLayer,(40,5,4),{"activation": None}), #20 kernels, filter 5x4
        (LeNetPoolLayer,(2,1),{"activation": T.tanh}), #downsample to half the size, now we are at 3x1  (*40 stacks)
        (BaseLayer, 500, {"activation": T.nnet.sigmoid}),
        (LogisticRegression, 2, {"activation": None}),      
        ]
        )
    
    def do_doucka_speed1():                         #1300minut, 90*19*239 dat,
        pnuminputwords=52                           #valid 26.26% 26.02% test -> a zase samy nuly .....
        do_nn(
        name='doucka-old',
        finetune_lr=0.1,
        pretraining_epochs=[12,20,40,60],
        corruption_levels = [.0, .0, .0,.0, .0, .0,.0, .0, .0],
        contraction_levels = [.01, .02, .03, .03, .03, .03, .03, .03, .03, .03],
        pretrain_lr=0.001,
        training_epochs=1000,
    
        batch_size=239,
        pnum_train=50*19*239,
        pnum_valid=20*19*239,
        pnum_test=20*19*239,
        pnumOutDatas=1,
        porigfile="..\\..\\Data\\doucka\\procords.csv",
        pnuminputwords=64,
        pwordmovespeed=1,
        pcsvtextcol=1,
        poutputcol=1,
        
        poutputismatrix=False,#affected by last layer and .y settings
        ppretrain_number=3,
        pL1reg=0.00,
        pL2reg=0.0001,
        layerstooutput= None,
        architecture=[
        (ReshaperManip,(1,pnuminputwords,52), {"activation": None}),    #always (1,pnuminputwords,52)
        (LeNetConvLayer,(20,5,5),{"activation": None}), #20 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 24x24
        (LeNetConvLayer,(30,5,5),{"activation": None}), #30 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 10x10
        (LeNetConvLayer,(20,5,5),{"activation": None}), #20 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 3x3
        (BaseLayer, 10, {"activation": T.nnet.sigmoid}),
        (LogisticRegression, 2, {"activation": None}),      
        ]
        )
    
    def do_obvious():
        pnuminputwords=52                        #-> uz outputuje i jiny veci nez jen nuly ... 
        do_nn(
        name='obvious',
        finetune_lr=0.1,
        pretraining_epochs=[12,20,40,60],
        corruption_levels = [.0, .0, .0,.0, .0, .0,.0, .0, .0],
        contraction_levels = [.01, .02, .03, .03, .03, .03, .03, .03, .03, .03],
        pretrain_lr=0.001,
        training_epochs=1000,
        batch_size=2, #pro speed 52-5 je celkem inputu pro nn 53, takze vezmeme 52 a batchsize 2 
        pnum_train=16*2,
        pnum_valid=5*2,
        pnum_test=5*2,
        pnumOutDatas=1,
        porigfile="obvious.csv",
        pnuminputwords=pnuminputwords,
        pwordmovespeed=pnuminputwords-5, #how fast we move the input window.
        pcsvtextcol=1,
        poutputcol=0,#obvious   
        poutputismatrix=False,#affected by last layer and .y settings
        ppretrain_number=0,#dont pretrain convolutions
        pL1reg=0.00,
        pL2reg=0.0001,
        layerstooutput= None, #automatical, to empty set to = []    
        architecture=[
        (ReshaperManip,(1,pnuminputwords,52), {"activation": None}),    #always (1,pnuminputwords,52)
        (LeNetConvLayer,(20,5,5),{"activation": None}), #20 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 24x24
        (LeNetConvLayer,(30,5,5),{"activation": None}), #30 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 10x10
        (LeNetConvLayer,(20,5,5),{"activation": None}), #20 kernels, filter 5x5
        (LeNetPoolLayer,(2,2),{"activation": T.tanh}), #downsample to half the size, now we are at 3x3
        (BaseLayer, 10, {"activation": T.nnet.sigmoid}),
        (LogisticRegression, 2, {"activation": None}),      
        ]
        )        
    #------------------------------------------------------------------------
    do_dbpedia1250_cleverspeed_big1()
    #do_dbpedia1250_cleverspeed()#_neworder()
    #do_doucka1250_try16autoenc()
    #do_doucka1250_try2autoenc()
    #do_dbpedia1250_cleverspeed()