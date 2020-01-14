//---------------------------------------------------------------------------

#ifndef fannbigtrainH
#define fannbigtrainH

#include "FANN-2.2.0-Source\src\include\doublefann.h"
#include "FANN-2.2.0-Source\src\include\fann_cpp.h"

class neural_net_public : public FANN::neural_net
{
  public:
    struct fann* get_struct_ptr(){return ann;};
};

typedef int (*data_train) (struct fann *ann,void *pUser,fann_type **pInput,fann_type **pOutput);
//first calling of such function - reset yourself to first input and load yourself
//otherwise - can be reseted to begin at another input!
//NULL,NULL -> give me the number of training data!
//returns pInput and pOutput, pointers.
//the callbacks need to assert, that the pointers are valid til calling the next function data_train
//user should free all the data after using.

float fann_bigtrain_epoch_quickprop(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser);
float fann_bigtrain_epoch_irpropm(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser);
float fann_bigtrain_epoch_sarprop(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser);
float fann_bigtrain_epoch_batch(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser);
float fann_bigtrain_epoch_incremental(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser);
FANN_EXTERNAL float FANN_API fann_bigtrain_epoch(struct fann *ann, unsigned __int64 pNumDat, data_train pTrainCallback, void *pUser);
FANN_EXTERNAL void FANN_API fann_bigtrain_on_callback(struct fann *ann, data_train pTrainCallback, void *pUser,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error);
FANN_EXTERNAL void FANN_API fann_biginit_weights(struct fann *ann,
fann_type smallest_inp,fann_type largest_inp,
 data_train pTrainCallback, void *pUser);

//---------------------------------------------------------------------------
#endif
