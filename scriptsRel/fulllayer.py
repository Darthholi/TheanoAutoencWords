"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

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
from wordypytlib import *

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

#from utils import tile_raster_images

#try:
#    import PIL.Image as Image
#except ImportError:
#    import Image
    
    

class BaseLayer(object):
    """
    
    Layer class implementing classical nonlinear layer
    and functionality for autoencoders (denoising and/or) contracting
    the autoencoder functionality is initialized only if needed in 

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_in=0, #and also reconstructed in...
        n_layer=500,#and in this case even output size too
        W=None,
        b=None,
        activation=T.nnet.sigmoid,
        batch_size=1
    ):
        """
        Initialize the class by specifying the number of visible units (the
        dimension d of the input ), the number of output units ( the dimension
        d' of the latent or hidden space ). The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_in: int
        :param n_in: number of input (or visible units in autoencoder naming)

        :type n_out: int
        :param n_out:  number of output units (or hidden in called autoencoder hidden units)

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type b: theano.tensor.TensorType
        :param b: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong this layer and another
                     architecture; if should be standalone set this to None

        """
        
        
            
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
            self.n_in=n_in
        else:
            if (hasattr(n_in, "__len__") and len(n_in)>=2):       #not includeing batch_size
                self.x=input.flatten(2)
                self.n_in=numpy.prod(n_in)#self.x.shape[1] #shape 0 is batch_size
            else:
                self.x = input
                self.n_in=n_in
        
        self.n_layer = n_layer 
        self.n_out = n_layer       #n_hidden
        self.activation=activation
        self.n_batchsize = batch_size

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_in+n_out)) and
            # 4*sqrt(6./(n_out+n_in))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / numpy.float(self.n_out + self.n_in)),
                    high=4 * numpy.sqrt(6. / numpy.float(self.n_out + self.n_in)),
                    size=(self.n_in, self.n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not b:
            b = theano.shared(
                value=numpy.zeros(
                    self.n_out,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = b
        
        self.theano_rng = theano_rng
        
        self.output = self.get_output_tfunc(self.x)
        #print self.output.shape[0], self.output.shape[1]
        self.reconstructed_input = None #filled only for requesting autoenc functionality
        self.pretrain_output = None     #filled only for requesting autoenc functionality

        self.params = [self.W, self.b]
        self.params_pretrain = None #this will get filled only if requesting autoencoder functionality
    
    def get_output_tfunc(self,input):                                    #once for input and once for corrupted input, also used
        lin_output = T.dot(input, self.W) + self.b
        output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        return output
    def L1regnorm(self):
        return self.W.sum()
    def L2regnorm(self):
        return (self.W ** 2).sum() 
    
    def classforY(self):#... just for real valued regression ... use for classification only when activation set to sigmoid.
        return T.matrix('y')
        
    def getFinetuneCost(self,realy):                      #this layer can be output layer - must implement this method
        #print "realy"
        #print realy.shape[0], realy.shape[1]
        return T.sum((self.output-realy)**2)/self.n_batchsize        #specifically SummedsquaredError
    def errors(self,realy):
        return self.getFinetuneCost(realy)/self.n_out
 ######################functionality for pretraining:       
    def canPretrain(self):
        return True

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_pretrain_params(self):
        return self.b_prime

    def get_cost_updates_pretrain(self, learning_rate, corruption_level=0.0, contraction_level=0.0, bvis=None):       #call only when needed autoencoder functionality for pretraining
        """ This function computes the cost and the updates for one trainng
        step of the dA
        
        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
                     
                     normally bvis is just for pretraining and should be ignored. 
        
         """
        
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis          #for autoenc functionality
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T      #for autoenc functionality
        
        #for autoenc functionality:       
        if not self.b_prime:     #if was not initialized in constructor
            self.b_prime = theano.shared(
                value=numpy.zeros(
                    self.n_in,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        if not self.pretrain_output:                                            #output of just autoencoder for pretraining
            #theano ifelse
            tilde_x=ifelse(T.le(corruption_level, 0.0), self.x, self.get_corrupted_input(self.x, corruption_level)) 
            self.pretrain_output = self.get_output_tfunc(tilde_x)
        
        if not self.reconstructed_input:                                       #output of autoencoder mirrored - the reconstructed input for pretraining
            lin_reconstructed = T.dot(self.pretrain_output, self.W_prime) + self.b_prime
            self.reconstructed_input = (
                lin_reconstructed if self.activation is None
                else self.activation(lin_reconstructed)
            )
            
        z = self.reconstructed_input
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        self.L_rec = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        
        """
        #slower variant, that works for all activation functions:
        #def get_jacobian(self,hidden,W):
        #    return T.reshape(hidden * (1 - hidden),
        #                 (self.n_batchsize, 1, self.n_out)) * T.reshape(
        #                     W, (1, self.n_in, self.n_out))
        
        #J = self.get_jacobian(self.pretrain_output, self.W)
        #self.L_jacob = T.sum(J ** 2) / self.n_batchsize
        #faster
        ##sum(h_i * (1-h_i)) * sum(W_ij)
        ##results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences=X)compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=[results])        
                  
        cost = ifelse(T.le(contraction_level, 0.0),
                T.mean(self.L_rec),
                T.mean(self.L_rec) + contraction_level * T.mean(self.L_jacob)
                )         
        #classical autoencoder OR denoising autoencoder OR contracting autoencoder
        """
        self.L_jacob = T.sum(self.pretrain_output*(1-self.pretrain_output))/self.n_batchsize  *T.sum(self.W**2)#fast variant for sigmoids...#elementwise
        
        cost = ifelse(T.le(contraction_level, 0.0),
                T.mean(self.L_rec),
                T.mean(self.L_rec) + contraction_level * T.mean(self.L_jacob)
                )  
        
        #cost = T.mean(self.L_rec)

        #and beware!! for pretraining there are different parameters!
        self.params_pretrain = [self.W, self.b, self.b_prime]
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params_pretrain)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params_pretrain, gparams)
        ]

        return (cost, updates)



def test_autoenc(learning_rate=0.1, training_epochs=15,
            batch_size=239, output_folder='out'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """

    dataset = TPreloader(pbatch_size=batch_size,
      pnum_train=90*19*239,
      pnum_valid=0,
      pnum_test=0,
      pnumOutDatas=0,
      porigfile="..\\..\\Data\\doucka\\procords.csv",
      pnuminputwords=64,
      pcsvtextcol=1,
      poutputcol=1)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = dataset.num_train / dataset.batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    #index = T.lscalar()    # index to a [mini]batch ........ dont need  and theano will notice unused variable
    x = T.matrix('x')  # matrix because we put it there as minibatches
    # end-snippet-2

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    autoenc = BaseLayer(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_in=dataset.const_lineInSize,
        n_out=500
    )

    cost, updates = autoenc.get_cost_updates_pretrain(     #this says that we will be doing autoencoding pretraining
        learning_rate=learning_rate,
        corruption_level=0.,
        contraction_level=0.
    )

    train_layer = theano.function(
        inputs=[x],
        outputs=cost,
        updates=updates
    )

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    print "-------------------------------------------------------------"
    import datetime
    f1=open('./logfile.txt', 'w+')
    f1.write("wordy autoenc started at \n")
    f1.write(str(datetime.datetime.now()))
    f1.close()
    
    print "training. number of batches will be ",n_train_batches
    print "one batch is ",dataset.batch_size, " rows"
    print "each row is ",dataset.const_lineInSize," floats"
    print "training epochs specified ",training_epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        print "-------------------------------------------------------------"
        print "epoch ", epoch
        startthisepoch=time.clock()
        if (epoch>0):
           ceta=(startthisepoch-start_time)/(epoch) * training_epochs
           print " (epoETA ","{:10.2f}".format(ceta),") " 
        
        
        dataset.MoveTo(0)#reset, because the next function just fetches each time next input
        for batch_index in xrange(n_train_batches):
            if batch_index<10 or (batch_index % (math.floor(n_train_batches/10)))==0 :
                print "batch ", batch_index, " start ","{:10.2f}".format(time.clock()),
                if (batch_index>0):
                    ceta=(time.clock()-startthisepoch)/(batch_index) * n_train_batches
                    print " (ETA ","{:5.2f}".format(ceta),") "
                else:
                    print " "
                    
            thiscost = train_layer(dataset.getcurX(batch_index))
            c.append(thiscost)
            
            if batch_index<10 or (batch_index % (math.floor(n_train_batches/10))) == 0 :
                print "cost was ",thiscost

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
        f1=open('./logfile.txt', 'w+')
        f1.write("epoch ")
        f1.write(str(epoch))
        f1.write(" cost ")
        f1.write(str(numpy.mean(c)))
        f1.close()
        numpy.savez_compressed('autoencwordy.npz', a=autoenc.W.get_value(borrow=True))

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    #image = Image.fromarray(
    #    tile_raster_images(X=autoenc.W.get_value(borrow=True).T,
    #                       img_shape=(28, 28), tile_shape=(10, 10),
    #                       tile_spacing=(1, 1)))
    #image.save('filters_corruption_0.png')

"""
    # start-snippet-3
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_in=2,
        n_out=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 30% corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % (training_time / 60.))
    # end-snippet-3

    # start-snippet-4
    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')
    # end-snippet-4

    os.chdir('../')
"""


if __name__ == '__main__':
    test_autoenc()
