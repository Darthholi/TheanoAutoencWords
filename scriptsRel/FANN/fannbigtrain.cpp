//---------------------------------------------------------------------------

#pragma hdrstop

#include "fannbigtrain.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)

#ifndef FIXEDFANN

/*
 * Internal train function
 */
float fann_bigtrain_epoch_quickprop(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser)
{
	unsigned __int64 i;

	if(ann->prev_train_slopes == NULL)
	{
		fann_clear_train_arrays(ann);
	}

	fann_reset_MSE(ann);

	for(i = 0; i < pNumDat; i++)
	{
    fann_type *pIn;
    fann_type *pOut;
    pTrainCallback(ann,pUser,&pIn,&pOut);
		fann_run(ann, pIn);
		fann_compute_MSE(ann, pOut);
		fann_backpropagate_MSE(ann);
		fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
	}
	fann_update_weights_quickprop(ann, pNumDat, 0, ann->total_connections);

	return fann_get_MSE(ann);
}

/*
 * Internal train function
 */
float fann_bigtrain_epoch_irpropm(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser)
{
	unsigned __int64 i;

	if(ann->prev_train_slopes == NULL)
	{
		fann_clear_train_arrays(ann);
	}

	fann_reset_MSE(ann);

	for(i = 0; i < pNumDat; i++)
	{
    fann_type *pIn;
    fann_type *pOut;
    pTrainCallback(ann,pUser,&pIn,&pOut);
		fann_run(ann, pIn);
		fann_compute_MSE(ann, pOut);
		fann_backpropagate_MSE(ann);
		fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
	}

	fann_update_weights_irpropm(ann, 0, ann->total_connections);

	return fann_get_MSE(ann);
}

/*
 * Internal train function
 */
float fann_bigtrain_epoch_sarprop(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser)
{
	unsigned __int64 i;

	if(ann->prev_train_slopes == NULL)
	{
		fann_clear_train_arrays(ann);
	}

	fann_reset_MSE(ann);

	for(i = 0; i < pNumDat; i++)
	{
    fann_type *pIn;
    fann_type *pOut;
    pTrainCallback(ann,pUser,&pIn,&pOut);
		fann_run(ann, pIn);
		fann_compute_MSE(ann, pOut);
		fann_backpropagate_MSE(ann);
		fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
	}

	fann_update_weights_sarprop(ann, ann->sarprop_epoch, 0, ann->total_connections);

	++(ann->sarprop_epoch);

	return fann_get_MSE(ann);
}

/*
 * Internal train function
 */
float fann_bigtrain_epoch_batch(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser)
{
	unsigned __int64 i;

	fann_reset_MSE(ann);

	for(i = 0; i < pNumDat; i++)
	{
    fann_type *pIn;
    fann_type *pOut;
    pTrainCallback(ann,pUser,&pIn,&pOut);
		fann_run(ann, pIn);
		fann_compute_MSE(ann, pOut);
		fann_backpropagate_MSE(ann);
		fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
	}

	fann_update_weights_batch(ann, pNumDat, 0, ann->total_connections);

	return fann_get_MSE(ann);
}

/*
 * Internal train function
 */
float fann_bigtrain_epoch_incremental(struct fann *ann,  unsigned __int64 pNumDat,data_train pTrainCallback, void *pUser)
{
	unsigned __int64 i;

	fann_reset_MSE(ann);

	for(i = 0; i < pNumDat; i++)
	{
    fann_type *pIn;
    fann_type *pOut;
    pTrainCallback(ann,pUser,&pIn,&pOut);
		fann_train(ann, pIn, pOut);
	}

	return fann_get_MSE(ann);
}

/*
 * Train for one epoch with the selected training algorithm
 */
FANN_EXTERNAL float FANN_API fann_bigtrain_epoch(struct fann *ann, unsigned __int64 pNumDat, data_train pTrainCallback, void *pUser)
{
	switch (ann->training_algorithm)
	{
	case FANN_TRAIN_QUICKPROP:
		return fann_bigtrain_epoch_quickprop(ann, pNumDat, pTrainCallback,pUser);
	case FANN_TRAIN_RPROP:
		return fann_bigtrain_epoch_irpropm(ann, pNumDat,pTrainCallback,pUser);
	case FANN_TRAIN_SARPROP:
		return fann_bigtrain_epoch_sarprop(ann, pNumDat,pTrainCallback,pUser);
	case FANN_TRAIN_BATCH:
		return fann_bigtrain_epoch_batch(ann, pNumDat,pTrainCallback,pUser);
	case FANN_TRAIN_INCREMENTAL:
		return fann_bigtrain_epoch_incremental(ann, pNumDat,pTrainCallback,pUser);
	}
	return 0;
}

FANN_EXTERNAL void FANN_API fann_bigtrain_on_callback(struct fann *ann, data_train pTrainCallback, void *pUser,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error)
{
	float error;
	unsigned int i;
	int desired_error_reached;

#ifdef DEBUG
	printf("Training with %s\n", FANN_TRAIN_NAMES[ann->training_algorithm]);
#endif

	if(epochs_between_reports && ann->callback == NULL)
	{
		printf("Max epochs %8d. Desired error: %.10f.\n", max_epochs, desired_error);
	}

  unsigned __int64 xNumDat=pTrainCallback(ann,pUser,NULL,NULL);   //init callback
	if(xNumDat <= 0)//nope
		return;

	for(i = 1; i <= max_epochs; i++)
	{
		/*
		 * train
		 */
		error = fann_bigtrain_epoch(ann, xNumDat, pTrainCallback,pUser);
		desired_error_reached = fann_desired_error_reached(ann, desired_error);

		/*
		 * print current output
		 */
		if(epochs_between_reports &&
		   (i % epochs_between_reports == 0 || i == max_epochs || i == 1 ||
			desired_error_reached == 0))
		{
			if(ann->callback == NULL)
			{
				printf("Epochs     %8d. Current error: %.10f. Bit fail %d.\n", i, error,
					   ann->num_bit_fail);
			}
			else if(((*ann->callback)(ann, NULL, max_epochs, epochs_between_reports,
									  desired_error, i)) == -1)
			{
				/*
				 * you can break the training by returning -1
				 */
				break;
			}
		}

		if(desired_error_reached == 0)
			break;
	}

}

#endif



FANN_EXTERNAL void FANN_API fann_biginit_weights(struct fann *ann,
fann_type smallest_inp, fann_type largest_inp,
 data_train pTrainCallback, void *pUser)
{
	unsigned int dat = 0, elem, num_connect, num_hidden_neurons;
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it, *last_neuron, *bias_neuron;

#ifdef FIXEDFANN
	unsigned int multiplier = ann->multiplier;
#endif
	float scale_factor;

  if (smallest_inp==largest_inp)   //we need to compute it
  {
    unsigned __int64 xNumDat=pTrainCallback(ann,pUser,NULL,NULL);   //init callback
    if(xNumDat <= 0)//nope
      return;
    fann_type *pIn;
    fann_type *pOut;
    pTrainCallback(ann,pUser,&pIn,&pOut);
    for(smallest_inp = largest_inp = pIn[0]; dat < xNumDat; dat++)
    {
      for(elem = 0; elem < ann->num_input; elem++)
      {
        if(pIn[elem] < smallest_inp)
          smallest_inp = pIn[elem];
        if(pIn[elem] > largest_inp)
          largest_inp = pIn[elem];
      }
      pTrainCallback(ann,pUser,&pIn,&pOut);
    }
  }

	num_hidden_neurons =
		ann->total_neurons - (ann->num_input + ann->num_output +
							  (ann->last_layer - ann->first_layer));
	scale_factor =
		(float) (pow
				 ((double) (0.7f * (double) num_hidden_neurons),
				  (double) (1.0f / (double) ann->num_input)) / (double) (largest_inp -
																		 smallest_inp));

#ifdef DEBUG
	printf("Initializing weights with scale factor %f\n", scale_factor);
#endif
	bias_neuron = ann->first_layer->last_neuron - 1;
	for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;

		if(ann->network_type == FANN_NETTYPE_LAYER)
		{
			bias_neuron = (layer_it - 1)->last_neuron - 1;
		}

		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			for(num_connect = neuron_it->first_con; num_connect < neuron_it->last_con;
				num_connect++)
			{
				if(bias_neuron == ann->connections[num_connect])
				{
#ifdef FIXEDFANN
					ann->weights[num_connect] =
						(fann_type) fann_rand(-scale_factor, scale_factor * multiplier);
#else
					ann->weights[num_connect] = (fann_type) fann_rand(-scale_factor, scale_factor);
#endif
				}
				else
				{
#ifdef FIXEDFANN
					ann->weights[num_connect] = (fann_type) fann_rand(0, scale_factor * multiplier);
#else
					ann->weights[num_connect] = (fann_type) fann_rand(0, scale_factor);
#endif
				}
			}
		}
	}

#ifndef FIXEDFANN
	if(ann->prev_train_slopes != NULL)
	{
		fann_clear_train_arrays(ann);
	}
#endif
}