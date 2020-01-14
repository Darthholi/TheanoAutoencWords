import ffnnwordy
from convolutional_mlpwordy import *
from fulllayer import *
#560000sentences training #70000testing(&valid?)
pnuminputwords=52                     #shuffled and appended #resulting number of nn-inputs (to 52size and speed 52-5): 916935  ~ 916940 = 4*5*19*19 * 127

ffnnwordy.do_nn(                                #resulting number of nn-inputs (to 52size and speed 1): 27991293 = (3 * 11 * 11 * 29 * 2659) = 2659*29*363  
name='dbpedia1250allnormalized',                
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
pwordmovespeed=pnuminputwords-5, # speed 1 -> inputs 27991293 = (3 * 11 * 11 * 29 * 2659) = 2659*29*363
pnormalize=True,

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