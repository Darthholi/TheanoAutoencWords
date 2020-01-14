"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from wordylogreg import LogisticRegression
from wordymlp import HiddenLayer
from wordypytlib import *


class ReshaperManip(object):    #not a true layer - no params
    def __init__(self,
        numpy_rng,
        input=None,
        n_in=0, #number told to us from previous layer
        n_layer=(1,28,28), #reshape to this image shape... (not including batch_size)
        batch_size=1,
        activation=None):
        
        self.x = input
        self.n_layer = n_layer
        if (len(n_layer)<3):
            inshaper=(batch_size,1,n_layer[0],n_layer[1])
        else:
            inshaper=(batch_size,n_layer[0],n_layer[1],n_layer[1])
        self.output = self.x.reshape(inshaper)    
        
        self.n_out = (inshaper[1],inshaper[2],inshaper[3])
        
        self.params = []
        
    #def classforY(self): throw error           
    # def getFinetuneCost(self,realy):  throw error
    #def errors(self,realy): throw error
    def L1regnorm(self):
        return 0
    def L2regnorm(self):
        return 0      
    def canPretrain(self):
        return True #but will be totally ignored :)
    def get_pretrain_params(self):
        return None
    def get_cost_updates_pretrain(self, learning_rate, corruption_level=0.0, contraction_level=0.0, bvis=None):
        return None, None

class LeNetConvLayer(object):
    def __init__(self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_in=(1,28,28), #number told to us from previous layer
        n_layer=(1,24,24), #number of kernels, filterX, filterY     ... note that the resulting n_out is different!
        W=None,                              #also the nubers will affect filter shapes!!!!
        b=None,
        batch_size=1,
        activation=None
    ):
        
        self.x = input
        self.n_in=n_in
        self.n_layer = n_layer
        
        #n_layer#without the batch size dimension (kernels,picX,picY)
         
        assert len(n_in) == len(n_layer)                
        #if len(n_in.shape) != len(n_out.shape):      #typically first convolution layer....
        #    except "insert reshaper before!"
            #inshaper=(batch_size,1,n_layer[1],n_layer[2])
            #rightinp=self.input.reshape(inshaper)
            #filter_shape=(n_layer[0],1,,)#(nkerns[1], nkerns[0], 5, 5),
        
        #inshape=image_shape=(batch_size,n_in[0],n_in[1],n_in[2]) #n_in[0] is the number of kernels from before
        
        filter_shape=(n_layer[0],n_in[0],n_layer[1],n_layer[2])#to new number of kernels, from old number of kernels, use filter thisbig
        
        self.n_out = (n_layer[0],n_in[1]-n_layer[1]+1,n_in[2]-n_layer[2]+1)
                                  
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) )#/ numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                numpy_rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # convolve input feature maps with filters
        self.output = conv.conv2d(
            input=self.x,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=(batch_size,n_in[0],n_in[1],n_in[2])
        )   
        
        self.params = [self.W]
        
    #def classforY(self): throw error           
    # def getFinetuneCost(self,realy):  throw error
    #def errors(self,realy): throw error
    def L1regnorm(self):
        return self.W.sum()
    def L2regnorm(self):
        return (self.W ** 2).sum()       
    def canPretrain(self):
        return False
    def get_pretrain_params(self):
        return None
        
class LeNetPoolLayer(object):
    def __init__(self,
        input,
        n_in,         #stack of images, + their X,Y dims.
        n_layer, #divide dimensions by this number
        batch_size,
        numpy_rng=None,
        activation=T.tanh,
        mode='max'
        ):        
        
        self.input=input
        self.n_in=n_in
        self.n_layer=n_layer
        self.n_out=(self.n_in[0] ,self.n_in[1] / n_layer[0],self.n_in[2] / n_layer[1]) #thats what pooling layer does
        self.activation=activation
        self.n_batchsize = batch_size
        self.mode=mode
        
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((n_in[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        self.params=[self.b]
        
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=self.input,
            ds=n_layer,
            ignore_border=True,
            mode=self.mode
        )
              
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        
    #def classforY(self): throw error           
    # def getFinetuneCost(self,realy):  throw error
    #def errors(self,realy): throw error
    def L1regnorm(self):
        return 0
    def L2regnorm(self):
        return 0      
    def canPretrain(self):
        return False  
    def get_pretrain_params(self):
        return None     
        
        
#----------------------------------- orig obj

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    nkerns=[20, 30, 40], batch_size=239):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    cNumInputWords=52

    dataset = TPreloader(pbatch_size=batch_size,
      pnum_train=50*19*239,
      pnum_valid=20*19*239,
      pnum_test=20*19*239,
      pnumOutDatas=1,
      porigfile="..\\..\\Data\\doucka\\procords.csv",
      pnuminputwords=cNumInputWords,
      pcsvtextcol=1,
      poutputcol=1)
    print dataset.id_train
    print dataset.id_valid
    print dataset.id_test
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = dataset.num_train / dataset.batch_size
    n_valid_batches = dataset.num_valid / dataset.batch_size
    n_test_batches = dataset.num_test / dataset.batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    #y = T.vector('y')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, cNumInputWords,52))        

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (52-5+1 , 52-5+1) = (48, 48)
    # maxpooling reduces this further to (48/2, 48/2) = (24, 24)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, cNumInputWords,52),  #input
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (24-5+1, 24-5+1) = (20, 20)
    # maxpooling reduces this further to (20/2, 20/2) = (10, 10)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 10, 10)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 24, 24),#input
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (10-5+1, 10-5+1) = (6, 6)
    # maxpooling reduces this further to (6/2, 6/2) = (3, 3)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 3, 3)
    layer1b = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 10, 10),
        filter_shape=(nkerns[2], nkerns[1], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 3 * 3),
    # or (500, 50 * 3 * 3) = (500, 450) with the default values.
    layer2_input = layer1b.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[2] * 3 * 3,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [x,y],
        layer3.errors(y),
        allow_input_downcast=True
    )

    validate_model = theano.function(
        [x,y],
        layer3.errors(y),
        allow_input_downcast=True
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params +layer1b.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x,y],
        cost,
        updates=updates,
        allow_input_downcast=True
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    finetuneepostart = start_time

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        
        print "-------------------------------------------------------------"
        print "epoch ", epoch
        startthisepoch=time.clock()
        if (epoch>1):
            ceta=(startthisepoch-start_time)/(epoch-finetuneepostart) * (n_epochs-finetuneepostart)
            print " (epoETA ","{:10.2f}".format(ceta),") "
        
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            thisbatchstart=time.clock()
            
            dataset.MoveTo(dataset.id_train+minibatch_index*batch_size)#next getdata will go over training dataset on this index. We set it each time cos training and validation could get somewhere else...
            dataset.load_batch()
            cost_ij = train_model(dataset.getcurX(minibatch_index),dataset.getcurY(minibatch_index))    
            
            if minibatch_index<5 or (minibatch_index % (math.floor(n_train_batches/10)))==0 :
                print "batch ", minibatch_index, " start ","{:10.2f}".format(thisbatchstart),
                ceta=(time.clock()-startthisepoch)/(minibatch_index+1) * n_train_batches
                print " (ETA ","{:5.2f}".format(ceta),") ",
                print "cost was ",cost_ij

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = []
                for i in xrange(n_valid_batches):
                    dataset.load_batch()#dataset.id_valid+i*n_valid_batches)
                    validation_losses.append(validate_model(dataset.getcurX(i),dataset.getcurY(i)))                #important to start at index zero for the dataset getting functions
                this_validation_loss = numpy.mean(validation_losses)
                
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = []
                    for i in xrange(n_test_batches):
                        dataset.load_batch()#dataset.id_test+i*n_test_batches)
                        test_losses.append(test_model(dataset.getcurX(i),dataset.getcurY(i)))#important to start at index zero for the dataset getting functions
                    test_score = numpy.mean(test_losses)
                    
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    dataset.ExportIDNumbers()
    
    
    final = theano.function(
                inputs=[x],
                outputs=layer3.output
                #allow_input_downcast=True
            )
        
    numberoutputnumbers = 1
    layertooutput=0

    finalout=open('./out'+str(layertooutput)+'.csv', 'w')
    print "transforming input - writing out.csv"    
  
    dataset.MoveTo(0)#next getdata will go over training dataset
    for i in xrange(dataset.totalinputs/batch_size):
      dataset.load_batch()
      outbatch=final(dataset.getcurX(i))
      for j in xrange(batch_size):      #data in one batch     = outbatch.size
          if numberoutputnumbers==1:
              finalout.write(str(outbatch[j]))
          else:
              for k in xrange(numberoutputnumbers):              #dimension of output
                  finalout.write(str(outbatch[j][k]))
                  if k!=numberoutputnumbers-1:
                      finalout.write(", ")
          finalout.write("\n")  
    finalout.close()
    print "done writing out.csv"
    

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
