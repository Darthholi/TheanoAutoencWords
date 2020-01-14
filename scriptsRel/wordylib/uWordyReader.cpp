//---------------------------------------------------------------------------

#pragma hdrstop

#include "uWordyReader.h"

#include <ios>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <conio.h>
#include <stdio.h>
#include <math.h>
using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::left;
using std::right;
using std::showpos;
using std::noshowpos;
//---------------------------------------------------------------------------
#pragma package(smart_init)

static vector<TWordyReader *> libReaderObjs;
//---------------------------------------------------------------------------


TWordyReader::TWordyReader(string pOrigFile, unsigned int pCSVTextCol, unsigned int pNumInputWords,
  unsigned int pOutputCol, unsigned int pNumOutputDatas, unsigned int pWindowSpeed, bool pDoNormalize)
  //pwindowspeed = 0 -> we take just the first pNumInputWords from each paragraph. num inputs to NN = num paragraphs
  //... >0 -> we move about this amount to get the next input. num inputs to NN > num paragraphs
{
  fOrigFile=pOrigFile;
  fColumnProc=pCSVTextCol;
  fOutputCol=pOutputCol;
  fWindowSpeed=pWindowSpeed;
  fNumOutputDatas=pNumOutputDatas;
  fNumInputWords=pNumInputWords;
  fProc=NULL;
  fProcOut=NULL;
  fAllData=0;
  fLastRowIn=NULL;
  fLastRowOut=NULL;
  fLastParagraphValid=false;
  fThisInpID=0;
  fDoNormalize=pDoNormalize;
}
TWordyReader::~TWordyReader()
{
  if (fLastRowIn)
    free(fLastRowIn);
  if (fLastRowOut)
    free(fLastRowOut);
  if (fProc)
    fclose(fProc);
  if (fProcOut)
    fclose(fProcOut);
}
void TWordyReader::SaveTrainState(string pFile)
{
  ofstream os (pFile.c_str(), ios::binary);
  os.write((const char*)&fLastInOrder, sizeof(fLastInOrder));
  os.write((const char*)&fLastStatementProc, sizeof(fLastStatementProc));
  os.write((const char*)&fThisInpID, sizeof(fThisInpID));
  os.close();
}
void TWordyReader::LoadTrainState(string pFile)
{
  FILE *xIn=fopen(pFile.c_str(),"r");
  if (xIn)
  {
    fread(&fLastInOrder, sizeof(fLastInOrder),1,xIn);
    fread(&fLastStatementProc, sizeof(fLastStatementProc),1,xIn);
    fread(&fThisInpID, sizeof(fThisInpID),1,xIn);
    fclose(xIn);
    fLastParagraphValid=false;
  }
}
//---------------------------------------------------------------------------
unsigned int TWordyReader::ReadDataPtrz(TYPERET **pInput,TYPERET **pOutput)//gives pointer to my internal memory
{
  if (!fLastRowIn)
  {
    fLastRowIn=(TYPERET*)malloc(sizeof(TYPERET)*fNumInputWords);
  }
  if (!fLastRowOut && fNumOutputDatas>0)
  {
    fLastRowOut=(TYPERET*)malloc(sizeof(TYPERET)*fNumOutputDatas);
  }
  int xRet=ReadData(fLastRowIn,fLastRowOut);
  *pInput=fLastRowIn;
  *pOutput=fLastRowOut;//autoencoder
  return xRet;
}
unsigned int TWordyReader::ReadData(TYPERET *pInput,TYPERET *pOutput)
{
  if (!fProc)//first call - beginning to use
  {
    fAllData=0;//i64
    fProc=fopen(string(fOrigFile+string("BinRep.bin")).c_str(),"r");
    if (!fProc)//file open failed - file does not exist!
    {
      FILE *xIn=fopen(fOrigFile.c_str(),"r");
      if (!xIn) return 0;
      fProc=fopen(string(fOrigFile+string("BinRep.bin")).c_str(),"w");
      if (!fProc) return 0;
      if (fNumOutputDatas>0)
        fProcOut=fopen(string(fOrigFile+string("BinRepOuts.bin")).c_str(),"w");


      fSmallestInp=0;//this we know;
      fLargestInp=0;//this we compute;
      int xValMost=0;//this is the integer that will recieve the maximal value in input...

      printf("preparing inputs, please w8 \n");

      unsigned int xWords=0;
      unsigned int xRejected=0;

      char xStor[4*4096];//long lines of text possible?
      char xWord[512];
      char xProcced[128];
      const int xCodwordlen=fAllowed.length()+2;//for bin repr.

      xStatementBegs.clear();
      xWordsNumber.clear();


      while (fgets(xStor,sizeof(xStor),xIn))
      {
        if (fNumOutputDatas>0 && fOutputCol<UINT_MAX)
        {
          TYPERET xDV=0;
          char *xBeg=FindCSVCol(xStor,sizeof(xStor),fOutputCol); //at this CSV col we start and then we just sscanf eat consequent numbers.
          for (int ot=0;ot<fNumOutputDatas;ot++)
          {
            int xReaded=0;
            //Adjacent ordinary string literal tokens are concatenated. Adjacent wide string literal tokens are concatenated:
            //g++ and borland c++ builder are different in one thing - sscanf with "%*[^-+0123456789]" can cosume 0 characters in C++ builder, but cannot in g++...

            const char *movetofirst="%*[^-+0123456789]%n";  //so here we eat inputs that are not numbers
            sscanf(xBeg,movetofirst,&xDV,&xReaded);
            xBeg+=xReaded;

            const char *cf=TYPESCAN"%n";                    //and here we eat the number
            sscanf(xBeg,cf,&xDV,&xReaded);
            xBeg+=xReaded;
            fwrite(&xDV,sizeof(TYPERET),1,fProcOut);
          }
        }

        char *xCur=xStor;
        xCur=FindTextCol(xCur,sizeof(xStor),fColumnProc);
        char *xEnd=FindWord(xCur);

        long xNow=ftell(fProc);//where are we now

        unsigned int xWordsNow=0;
        while (xEnd)//found a word.
        {
          if (ProcWordDict(xCur,xEnd-xCur,xWord,2,xValMost))//to binary    true == has at least one allowed character
          {
            fwrite(xWord,xCodwordlen,1,fProc);
            xWordsNow++;
          }
          else xRejected++;

          xCur=xEnd+1;
          xEnd=FindWord(xCur);
          xWords++;
        }

        //save index:
        xStatementBegs.push_back(xNow);
        xWordsNumber.push_back(xWordsNow);
        //printf("w: %u a: %u   ",xWordsNow,GetNumInpsFromStatment(xWordsNow));
        fAllData+=GetNumInpsFromStatment(xWordsNow);

      }
      fLargestInp=xValMost;//computed...
      //save to file:
      int size1 = xStatementBegs.size();
      ofstream os (string(fOrigFile+string("BinIndBeg.bin")).c_str(), ios::binary);
      os.write((const char*)&fSmallestInp, sizeof(fSmallestInp));
      os.write((const char*)&fLargestInp, sizeof(fLargestInp));
      os.write((const char*)&size1, sizeof(size1));
      os.write((const char*)&xStatementBegs[0], size1 * sizeof(long));
      os.close();
      size1 = xWordsNumber.size();
      ofstream os2 (string(fOrigFile+string("BinIndCounts.bin")).c_str(), ios::binary);
      os2.write((const char*)&size1, sizeof(size1));
      os2.write((const char*)&xWordsNumber[0], size1 * sizeof(unsigned int));
      os2.close();
      //fprintf(tind,"%li %u \n",xWordsNow,xNow);//at this position in file begins this much words, strlen(xAllocated)*2 is the length per word.

      //reopen files
      fclose(xIn);
      fclose(fProc); //to open again in r mode.
      fProc=fopen(string(fOrigFile+string("BinRep.bin")).c_str(),"r");
      if (fNumOutputDatas>0)
      {
        fclose(fProcOut);
        fProcOut=fopen(string(fOrigFile+string("BinRepOuts.bin")).c_str(),"r");
      }

      //fclose(tind);
      printf("done %i words, %i rejected \n",xWords,xRejected);
      printf("paragraphs: %i \n",xStatementBegs.size());
      printf("largest input: %i \n",xValMost);
    }
    else//files exist!
    {
      int size1;
      ifstream is(string(fOrigFile+string("BinIndBeg.bin")).c_str(), ios::binary);
      is.read((char*)&fSmallestInp, sizeof(fSmallestInp));
      is.read((char*)&fLargestInp, sizeof(fLargestInp));
      is.read((char*)&size1, sizeof(size1));
      xStatementBegs.resize(size1);
      is.read((char*)&xStatementBegs[0], size1 * sizeof(long));
      is.close();
      ifstream is2(string(fOrigFile+string("BinIndCounts.bin")).c_str(), ios::binary);
      is2.read((char*)&size1, sizeof(size1));
      xWordsNumber.resize(size1);
      is2.read((char*)&xWordsNumber[0], size1 * sizeof(unsigned int));
      is2.close();
      fAllData=0;//i64
      if (fWindowSpeed<=0)
        fAllData=xWordsNumber.size();//one nninput per one statement
      else
      {
        for (int i=0;i<xWordsNumber.size();i++)
          fAllData+=GetNumInpsFromStatment(xWordsNumber[i]);
      }

      if (fNumOutputDatas>0)
        fProcOut=fopen(string(fOrigFile+string("BinRepOuts.bin")).c_str(),"r");
    }
    //either way we have it now!

    printf("in total one epoch would be %I64d of things... \n",fAllData);

    //reset state:
    MoveTo(0);
  }
  if (!pInput && !pOutput)//flaggy - give me number of inputs.
  {
    return fAllData;
  }


  const unsigned int xNeuronsPerWord=2*fAllowed.length();
  const unsigned int xSrcPerWord=fAllowed.length()+2;
  //*fWordsActiv

  fLastInOrder++;
  fThisInpID++;
  
  //FILE *flog=fopen("flog.log","a");
      //fprintf(flog,"going to last %u \n",fLastStatementProc);  //debug
      //fprintf(flog,"outvalue is %f \n",((TYPERET*)fCurParagraphOrigOuts)[0]);//debug
  //fprintf(flog,"inorder from %u \n",fLastInOrder);  //debug
  //fprintf(flog,"inpid from %I64 \n",fThisInpID);  //debug

  int xAdded=GetNumInpsFromStatment(xWordsNumber[fLastStatementProc]);

  while (fLastInOrder>=xAdded)   //while because rows woth zero inputs will then not be processed
  {
    fLastInOrder=0;
    fLastStatementProc++;
    //fprintf(flog,"inc statement proc to %u \n",fLastStatementProc);
    if (fLastStatementProc>=xWordsNumber.size())
    {
      fLastStatementProc=0;
      fThisInpID=0;
      //fprintf(flog,"reset statement proc");
    }
    fLastParagraphValid=false;

    //to skip zero lengths lets check it again
    xAdded=GetNumInpsFromStatment(xWordsNumber[fLastStatementProc]);
  }  
  if (!fLastParagraphValid)
  {
    fseek(fProc,xStatementBegs[fLastStatementProc],SEEK_SET);
    fread(fCurParagraphOrigData,xWordsNumber[fLastStatementProc]*xSrcPerWord,1,fProc);

    if (fNumOutputDatas>0)
    {
      fseek(fProcOut,fLastStatementProc*sizeof(TYPERET)*fNumOutputDatas,SEEK_SET);
      fread(fCurParagraphOrigOuts,sizeof(TYPERET)*fNumOutputDatas,1,fProcOut);
    }
    fLastParagraphValid=true;
    
    //fprintf(flog,"going to last %u \n",fLastStatementProc);  //debug
    //fprintf(flog,"outvalue is %f \n",((TYPERET*)fCurParagraphOrigOuts)[0]);//debug
  }
  //fprintf(flog,"inorder to %u \n",fLastInOrder);  //debug
  //fprintf(flog,"inpid to %I64 \n",fThisInpID);  //debug
  //fclose(flog);/*debug*/

  //fill fLastRow
  //#define NOTNORMALIZE
  //#define ZIGZAG
  
  if (pInput)
  {
    for (int i=0;i<fNumInputWords;i++)
    {
      BYTE *xInWord=fCurParagraphOrigData+((i+fLastInOrder*fWindowSpeed)%xWordsNumber[fLastStatementProc]);
      if (!fDoNormalize)
      {
        #ifndef ZIGZAG
          for (int j=0;j<xNeuronsPerWord/2;j++)
            pInput[i*xNeuronsPerWord+j]=((TYPERET)(xInWord[j]));
          //copy the info from 26 chars... and expand the last two
          for (int j=xNeuronsPerWord/2;j<xNeuronsPerWord;j++)
            pInput[i*xNeuronsPerWord+j]=0;
          pInput[i*xNeuronsPerWord+xNeuronsPerWord/2+xInWord[xSrcPerWord-2]]+=1.0;
          pInput[i*xNeuronsPerWord+xNeuronsPerWord/2+xInWord[xSrcPerWord-1]]+=1.0;//here the maximum is 2
        #else  
          for (int j=0;j<xNeuronsPerWord/2;j++)
            pInput[i*xNeuronsPerWord+2*j]=((TYPERET)(xInWord[j]));
          //copy the info from 26 chars... and expand the last two
          for (int j=0;j<xNeuronsPerWord/2;j++)
            pInput[i*xNeuronsPerWord+2*j+1]=0;
          pInput[i*xNeuronsPerWord+2*xInWord[xSrcPerWord-2]+1]+=1.0;
          pInput[i*xNeuronsPerWord+2*xInWord[xSrcPerWord-1]+1]+=1.0;//here the maximum is 2
        #endif
      }
      else // normalize
      {
        #ifndef ZIGZAG
          for (int j=0;j<xNeuronsPerWord/2;j++)
            pInput[i*xNeuronsPerWord+j]=((TYPERET)(xInWord[j]))/((TYPERET)(fLargestInp));
          //copy the info from 26 chars... and expand the last two
          for (int j=xNeuronsPerWord/2;j<xNeuronsPerWord;j++)
            pInput[i*xNeuronsPerWord+j]=0;
          pInput[i*xNeuronsPerWord+xNeuronsPerWord/2+xInWord[xSrcPerWord-2]]+=1.0/2.0;
          pInput[i*xNeuronsPerWord+xNeuronsPerWord/2+xInWord[xSrcPerWord-1]]+=1.0/2.0;//here the maximum is 2
        #else                    //zigzag:
          for (int j=0;j<xNeuronsPerWord/2;j++)
            pInput[i*xNeuronsPerWord+2*j]=((TYPERET)(xInWord[j]))/((TYPERET)(fLargestInp));
          //copy the info from 26 chars... and expand the last two
          for (int j=0;j<xNeuronsPerWord/2;j++)
            pInput[i*xNeuronsPerWord+2*j+1]=0;
          pInput[i*xNeuronsPerWord+2*xInWord[xSrcPerWord-2]+1]+=1.0/2.0;
          pInput[i*xNeuronsPerWord+2*xInWord[xSrcPerWord-1]+1]+=1.0/2.0;//here the maximum is 2
        #endif
      }
    }
  }
  if (pOutput && fNumOutputDatas>0)
  {                                                                       //outputs dont change but still we need to copy them
                                                //do autoencoder without asking for output please.
    //printf("going to last %u \n",fLastStatementProc);  //debug
    //printf("outvalue is %f \n",((TYPERET*)fCurParagraphOrigOuts)[0]);//debug
    //memcpy(pOutput,fCurParagraphOrigOuts,sizeof(TYPERET)*fNumOutputDatas);
    for (int i=0;i<fNumOutputDatas;i++)
    {
      pOutput[i]=((TYPERET*)fCurParagraphOrigOuts)[i];
    }
  }

  return fLastStatementProc;//return the statement number
}

unsigned int TWordyReader::GetNumInpsFromStatment(unsigned int xWordsNow)
{
  if (fWindowSpeed<=0)
    return 1;//one nninput per one statement
  if (xWordsNow<=0)
    return 0;
  
  if (xWordsNow%fWindowSpeed==0)
    return xWordsNow/fWindowSpeed;
  else    
    return xWordsNow/float(fWindowSpeed)+1; //how many times do i need to move to fill it fully?
}

void TWordyReader::MoveTo(int pPos)
{                     //next load will load pPos
  __int64 xLastSaved=fThisInpID;
  if (pPos<=0)
  {
    //printf("move to 0");
    fLastStatementProc=xWordsNumber.size()-1;//==next will be reset.
    fLastInOrder=xWordsNumber[fLastStatementProc]-1;    //next will be reset in ANY case of fWindowSpeed
    fThisInpID=fAllData-1;
  }
  else
  {
    if (pPos<=fThisInpID)
    {
      fLastStatementProc=0;
      fLastInOrder=0;
      fThisInpID=0;
      //printf("move to less");
    }

    int xAdded=GetNumInpsFromStatment(xWordsNumber[fLastStatementProc]);

    while (fThisInpID+xAdded<pPos-1)
    {
      //printf("move to jump");
      fThisInpID+=xAdded;
      fLastStatementProc++;
      if (fLastStatementProc >= xWordsNumber.size()-1)
        break;

      xAdded=GetNumInpsFromStatment(xWordsNumber[fLastStatementProc]);
    }
    fLastInOrder=(pPos-1 - fThisInpID);
    if (fLastInOrder>xAdded-2)
      fLastInOrder=xAdded-2;//to move just before last record
    fThisInpID+=fLastInOrder;
  }
  //printf("moved from %I64 to %I64",xLastSaved,fThisInpID);
  fLastParagraphValid=fLastParagraphValid && fThisInpID!=xLastSaved;
}
void TWordyReader::MoveFind(int pPos)
{
  __int64 xLastSaved=fThisInpID;
  if (pPos<=0)
  {
    //printf("move to 0");
    fLastStatementProc=xWordsNumber.size()-1;//==next will be reset.
    fLastInOrder=xWordsNumber[fLastStatementProc]-1;    //next will be reset in ANY case of fWindowSpeed
    fThisInpID=fAllData-1;
  }
  else
  {
    fLastStatementProc=pPos-1;
    fLastInOrder=GetNumInpsFromStatment(xWordsNumber[fLastStatementProc-1])-1;
    fThisInpID=0;
    
    for (unsigned int i=0; i<fLastStatementProc;i++)
    {
      fThisInpID+=GetNumInpsFromStatment(xWordsNumber[i]);
    }
    fThisInpID+=fLastInOrder;
  }
  //printf("moved from %I64 to %I64",xLastSaved,fThisInpID);
  fLastParagraphValid=false;//fLastParagraphValid && fThisInpID!=xLastSaved;
}

extern "C" __declspec(dllexport) int __cdecl ExternMove(int pHandle,unsigned int pPosition)//so that pPosition is read next callback
{
  libReaderObjs[pHandle]->MoveTo(pPosition);
  return 0;
}

extern "C" __declspec(dllexport) int __cdecl ExternFind(int pHandle,unsigned int pParagraph)//so that pParagraph is read next callback
{
  libReaderObjs[pHandle]->MoveFind(pParagraph);
  return 0;
}                            

extern "C" __declspec(dllexport) int __cdecl ExternInitNew(char* pOrigFile, unsigned int pCSVTextCol, unsigned int pNumInputWords,
  unsigned int pOutputCol, unsigned int pNumOutputDatas,unsigned int pWindowSpeed,bool pDoNormalize)
{
  TWordyReader *xNewR=new TWordyReader(pOrigFile,pCSVTextCol,pNumInputWords,pOutputCol,pNumOutputDatas,pWindowSpeed,pDoNormalize);
  for (int i=0;i<libReaderObjs.size();i++)
  {
    if (libReaderObjs[i]==NULL)
    {
      libReaderObjs[i]=xNewR;
      return i;
    }
  }
  libReaderObjs.push_back(xNewR);
  return libReaderObjs.size()-1;
}
extern "C" __declspec(dllexport) int __cdecl ExternFree(int pHandle)
{
  delete libReaderObjs[pHandle];
  libReaderObjs[pHandle]=NULL;
  return 0;
}

extern "C" __declspec(dllexport) int __cdecl ExternCallback(int pHandle,TYPERET *pInput,TYPERET *pOutput, unsigned int *pSentencIds, int pBatchSize, int pFlags)
{
  //NULL,NULL or BatchSize<=0 mean just return the number of things
  //Flags == 1 means export paragraph ids

  unsigned int xR=0;
  const unsigned int xNeuronsPerWord=2*libReaderObjs[pHandle]->fAllowed.length();

  if ((!pInput && !pOutput) || pBatchSize<=0)
  {
    unsigned int xAllData= libReaderObjs[pHandle]->ReadData(pInput,pOutput);
    
    if (pFlags==1)
    {
    
      FILE *fpi=fopen(string(libReaderObjs[pHandle]->fOrigFile+string("stIdcs.bin")).c_str(),"w");
      unsigned int xIDN=0;
      for (vector<unsigned int>::iterator it=libReaderObjs[pHandle]->xWordsNumber.begin();it!=libReaderObjs[pHandle]->xWordsNumber.end();it++)
      {
        for (unsigned int i=0;i<(*it);i++)
        {
          fprintf(fpi,"%u \n",xIDN);
        }
        xIDN++;
      }
      fclose(fpi);
    
    
    }
    
    return xAllData;
  }

  //printf("%u \n",libReaderObjs[pHandle]->fLastInOrder);   //debug
  //printf("%u \n",libReaderObjs[pHandle]->fLastStatementProc);
  //printf("%I64d \n",libReaderObjs[pHandle]->fThisInpID);

  for (int i=0;i<pBatchSize;i++)      //can overflow to the beginning of the whole set..
  {
    xR=libReaderObjs[pHandle]->ReadData(pInput,pOutput);//todo can run at the end of input, beware
    pInput=(TYPERET*)((BYTE*)pInput+sizeof(TYPERET)*xNeuronsPerWord*libReaderObjs[pHandle]->fNumInputWords); //different interpretation of g++ and bcb
    if (pOutput)
    {
      pOutput=(TYPERET*)((BYTE*)pOutput+sizeof(TYPERET)*libReaderObjs[pHandle]->fNumOutputDatas);   //different interpretation of g++ and bcb
    }
    if (pSentencIds)
    {
      pSentencIds[0]=xR;
      pSentencIds=(unsigned int*)((BYTE*)pSentencIds+sizeof(unsigned int));
    }
  }
  return xR;//todo by flags
}
extern "C" __declspec(dllexport) int __cdecl ExternSaveState(int pHandle,char* pOrigFile)
{
  libReaderObjs[pHandle]->SaveTrainState(pOrigFile);
}
extern "C" __declspec(dllexport) int __cdecl ExternLoadState(int pHandle,char* pOrigFile)
{
  libReaderObjs[pHandle]->LoadTrainState(pOrigFile);
}