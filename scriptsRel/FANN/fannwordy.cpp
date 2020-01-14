/*
 *
 *  Fast Artificial Neural Network (fann) for words autoencoder
 */



#include "uWordyReader.h"
#include "fannbigtrain.h"

unsigned int xTicksStarted;
unsigned int xTicksLastOut;
unsigned int xTicksLastArchived;
bool xActivityLearing;
extern TWordyReader *fannReaderObj=NULL;

// Callback function that simply prints the information to cout
int print_callback(FANN::neural_net &net, FANN::training_data &train,
  unsigned int max_epochs, unsigned int epochs_between_reports,
  float desired_error, unsigned int epochs, void *user_data)
{
  cout << "Epochs     " << setw(8) << epochs << ". " << "Current Error: " <<
    left << net.get_MSE() << right << endl;
  return 0;
}


int callback_train(struct fann *ann,void *pUser,fann_type **pInput,fann_type **pOutput, int &pFlags)
{
  TWordyReader *xRe=(TWordyReader*)pUser;

  if (!xRe->fProc)//begin of usage
  {
    xTicksStarted=GetTickCount();
    xTicksLastArchived=xRe->xTicksStarted;
    xTicksLastOut=0;
  }
  else
  {
    //stats from last time: - at this point the NN just asks for another input, that means that the last one is finished.
    int uN=GetTickCount();
    if (uN-xTicksLastOut>60000)
    {
      int xDif=(uN-xRe->xTicksStarted)/1000;
      xTicksLastOut=uN;
      printf("paragraph %i, order %i, time %is, MSE %f \n",xRe->fLastStatementProc,xRe->fLastInOrder,xDif,fann_get_MSE(ann));
    }
    if (xActivityLearning && uN-xTicksLastArchived>300000)
    {
      xTicksLastArchived=uN;
      string xSavenetfile="wordy_float.net";
      string xSavetrainlast="wordy_float_state.bin";
      fann_save(ann,xSavenetfile.c_str());
      xRe->SaveTrainState(xSavetrainlast.c_str());
      printf("...saved\n");
    }
  }

  return pUser->ReadData(pInput,pOutput,pFlags);
}




// fann C++ wrapper on words representation
void runWordy()
{
  /*string xSavenetfile="wordy_float.net";
  string xSavetrainlast="wordy_float_state.bin";
  string xSourceData="dbpedia\\train.csv";
  bool xLearn=true;
  int xCSVStringCol=2;/**/
  /**/
  string xSavenetfile="wordy_float.net";
  string xSavetrainlast="wordy_float_state.bin";
  string xSourceData="doucka\\procords.csv";
  bool xLearn=true;
  int xCSVStringCol=1;
  //"dbpedia\\train.csv",2
  /**/

  const unsigned int xWordsActiv=16;//number of words we want as input;
  const unsigned int xNeuronsPerWord=2*strlen(xAllowed);
  unsigned int xArchitecture[] =  //originally i wanted here like 5000
    {xWordsActiv*xNeuronsPerWord, 1000, 800,100,50,10, 2,10,50,100, 800, 1000, xWordsActiv*xNeuronsPerWord}; //autoencoder


  cout << endl << "wordy autoenc started." << endl;

  const float learning_rate = 0.7f;

  const float desired_error = 0.001f;
  const unsigned int max_iterations = 300000;
  const unsigned int iterations_between_reports = 1;

  /*default training is FANN_TRAIN_RPROP, that needs some memory to store gradients in addition to weights
  -can be saved to disk
  -for just weights, get FANN_TRAIN_INCREMENT
  */

  cout << endl << "Creating network." << endl;

  neural_net_public net;
  bool xLoaded=false;

  if (!net.create_from_file(xSavenetfile.c_str()))    // Initialize and train the network with the data
  {
    xLearn=true;//nothing else possible
    net.create_standard_array(sizeof(xArchitecture) / sizeof(unsigned int),xArchitecture);
    // net.create_standard(num_layers, num_input, num_hidden, num_output);

    net.set_learning_rate(learning_rate);

    net.set_activation_steepness_hidden(1.0);
    net.set_activation_steepness_output(1.0);

    net.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
    net.set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

    // Set additional properties such as the training algorithm
    net.set_training_algorithm(FANN_TRAIN_INCREMENTAL);//dod ostatni je potreba zmenit protoze pouze tohle updatuje po kazdy precteny vaze!
    // net.set_training_algorithm(FANN::TRAIN_QUICKPROP);

    // Output network type and parameters
  }
  else
  {
    xLoaded=true;
  }
  TWordyReader *xDataz=new TWordyReader(xSourceData.c_str(),xCSVStringCol,xWordsActiv,xLearn);

  cout << endl << "Network Type                         :  ";
  switch (net.get_network_type())
  {
  case FANN::LAYER:
    cout << "LAYER" << endl;
    break;
  case FANN::SHORTCUT:
    cout << "SHORTCUT" << endl;
    break;
  default:
    cout << "UNKNOWN" << endl;
    break;
  }
  net.print_parameters();

  if (xLearn)
    cout << endl << "Training network." << endl;
  else
    cout << endl << "Computing outputs." << endl;

  //PCa component analysis - eigenvectors of covariance matrix
  //spocti covariance matrix pro ty slova.
  //maximum variance unfolding - unfolds a manifold
  //uloz jako csv a delej clustering a zobrazovani v knimu.
  if (xDataz)
  {
    cout << endl << "first load." << endl;
    unsigned __int64 xNumDat= callback_train(net.get_struct_ptr(),xDataz,NULL,NULL);   //init callback

    if (!xLoaded)
    {
      cout << endl << "init weights." << endl;
      fann_biginit_weights(net.get_struct_ptr(),xDataz->fSmallestInp,xDataz->fLargestInp,callback_train, xDataz);
      cout << endl << "Save inited" << endl;
      net.save(xSavenetfile.c_str());
      xDataz->SaveTrainState(xSavetrainlast);
    }
    else
    {
      if (xLearn)    //coontinue learning from last
        xDataz->LoadTrainState(xSavetrainlast);
    }

    if (xLearn)
    {
      cout << "Max Epochs " << setw(8)
        << max_iterations << ". " << "Desired Error: " << left << desired_error <<
        right << endl;
      net.set_callback(print_callback, NULL);
      // net.train_on_data(data, max_iterations, iterations_between_reports, desired_error);

      fann_bigtrain_on_callback(net.get_struct_ptr(), callback_train, xDataz,
        max_iterations, iterations_between_reports, desired_error);

      cout << endl << "Saving network." << endl;

      // Save the network in floating point and fixed point
      net.save(xSavenetfile.c_str());
      //unsigned int decimal_point = net.save_to_fixed("wordy_fixed.net");
      //data.save_train_to_fixed("xor_fixed.data", decimal_point);
    }
    else
    {
      cout << endl << "Testing, eval, network." << endl;

      FILE *xDump=fopen(string(xDataz->fOrigFile+string("Eval.bin")).c_str(),"w");

      int xLayerFetch=4;
      struct fann *ann=net.get_struct_ptr();
      int xLastStatement=-1;

      for (unsigned int i = 0; i < xNumDat; ++i)
      {
        fann_type *pIn;
        fann_type *pOut;
        int xCurStatement=callback_train(ann,xDataz,&pIn,&pOut);

        //do we want distinct?
        if (xCurStatement==xLastStatement && i!=0) continue;
        xLastStatement=xCurStatement;

        // Run the network on the test data
        fann_type *calc_out = net.run(pIn);

        //save MSE for just this output
        /*cout << "XOR test (" << showpos << data.get_input()
          [i][0] << ", " << data.get_input()
          [i][1] << ") -> " << *calc_out << ", should be " << data.get_output()
          [i][0] << ", " << "difference = " << noshowpos << fann_abs
          (*calc_out - data.get_output()[i][0]) << endl;*/
        double xErr=0.0;
        for (int j=0;j<net.get_num_output();j++)
        {
          xErr+=(pOut[j]-calc_out[j])*(pOut[j]-calc_out[j]);
        }
        xErr=sqrt(xErr);
        xErr/=double(net.get_num_output());

        fprintf(xDump,"%i, %f",xCurStatement,xErr);
        //save layer fetch values
        struct fann_layer *layer_it;
        struct fann_neuron *neurons;//last_layer = ann->last_layer;
        layer_it = ann->first_layer + xLayerFetch;
        for (neurons = (layer_it - 1)->first_neuron;neurons!=(layer_it - 1)->last_neuron;neurons++)
        {
          double xC=neurons->value;
          fprintf(xDump,",%f",xC);
        }
        fputs("\n",xDump);

      }    /**/
      fclose(xDump);
    }

    cout << endl << "wordy completed." << endl;
    delete xDataz;
  }
}

/* Startup function. Syncronizes C and C++ output, calls the test function
 and reports any exceptions */
int main(int argc, char **argv)
{
  try
  {
    std::ios::sync_with_stdio(); // Syncronize cout and printf output
    runWordy();
  }
  catch (...)
  {
    cerr << endl << "Abnormal exception." << endl;
  }
  getch();
  return 0;
}
