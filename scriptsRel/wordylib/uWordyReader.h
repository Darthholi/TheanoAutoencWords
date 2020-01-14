//---------------------------------------------------------------------------

#ifndef uWordyReaderH
#define uWordyReaderH

#include "uWordy.h"
//---------------------------------------------------------------------------
#ifndef TYPERET
#define TYPERET float
#define TYPESCAN "%f"
#endif
class TWordyReader : public TWordyTransformator         //state class
{
  public:
  TWordyReader(string pOrigFile, unsigned int pCSVTextCol, unsigned int pNumInputWords, unsigned int pOutputCol,
   unsigned int pNumOutputDatas, unsigned int pWindowSpeed, bool pDoNormalize);
  ~TWordyReader();
  string fOrigFile;                                            //param, IN
  unsigned int fColumnProc;//which text column in CSV file     //param, IN
  unsigned int fOutputCol;//OutputColumn. UINT_MAX if not used.//param, IN
  unsigned int fWindowSpeed;

  bool fDoNormalize;

  void SaveTrainState(string pFile);
  void LoadTrainState(string pFile);

  unsigned int ReadData(TYPERET *pInput,TYPERET *pOutput);//reads next data
  unsigned int ReadDataPtrz(TYPERET **pInput,TYPERET **pOutput);//reads next data
  //input and output must be allocated
  void MoveTo(int pPos);//just before to so to read to in next ReadData....
  void MoveFind(int pPos);

  unsigned int GetNumInpsFromStatment(unsigned int xStatID);
  
  unsigned int fNumInputWords;   //computed
  unsigned int fNumOutputDatas;
  FILE *fProc;
  FILE *fProcOut;
  vector<long> xStatementBegs;
  vector<unsigned int> xWordsNumber;
  unsigned __int64 fAllData;//including cyclic  //computed

  double fSmallestInp;            //computed, saved
  double fLargestInp;             //computed, saved

  unsigned int fLastInOrder;//0...number of words in this paragraph-1    //state variable, saved using SaveTrainState
  unsigned int fLastStatementProc; //index to xStatementBegs             //state variable, saved using SaveTrainState
  __int64 fThisInpID;                                                    //state variable, saved using SaveTrainState

  //buffers:
  bool fLastParagraphValid;//false = realod again from disk
  TYPERET *fLastRowIn;      //what we return only for *Ptrz
  TYPERET *fLastRowOut;      //what we return only for *Ptrz
  BYTE fCurParagraphOrigData[1024*64];         //what we read for current cycle through fLastStatementProc
  BYTE fCurParagraphOrigOuts[1024*64];
};

//functions to export, library style:

//TODO - separovat neuralni sit fann *ann od callbacku pro read a udelat tu jenom readovaci veci ktery pak bude callbacktrain sam volat

extern "C" __declspec(dllexport) int __cdecl ExternInitNew(char* pOrigFile, unsigned int pCSVTextCol, unsigned int pNumInputWords, unsigned int pOutputCol, unsigned int pNumOutputDatas, unsigned int pWindowSpeed,bool pDoNormalize);
  //todo - possibly add the dictionary possibilities like passing the fAllowed and such.....
extern "C" __declspec(dllexport) int __cdecl ExternFree(int pHandle);
extern "C" __declspec(dllexport) int __cdecl ExternCallback(int pHandle,TYPERET *pInput,TYPERET *pOutput,unsigned int *pSentencIds, int pBatchSize, int pFlags);
//NULL,NULL or BatchSize<=0 mean just return the number of things
//pFlags ==1 -> export statement ids in textfile
//pflags ==2
extern "C" __declspec(dllexport) int __cdecl ExternMove(int pHandle,unsigned int pPosition);//so that pPosition is read next callback
extern "C" __declspec(dllexport) int __cdecl ExternFind(int pHandle,unsigned int pParagrapg);//so that pParagraph is read next callback
extern "C" __declspec(dllexport) int __cdecl ExternSaveState(int pHandle,char* pOrigFile);
extern "C" __declspec(dllexport) int __cdecl ExternLoadState(int pHandle,char* pOrigFile);

//extern TWordyReader *libReaderObj;


/*
http://stackoverflow.com/questions/18276362/why-do-i-need-declspecdllexport-to-make-some-functions-accessible-from-ctype

Ccode and python:
https://cffi.readthedocs.org/en/latest/
http://www.dreamincode.net/forums/topic/252650-making-importing-and-using-a-c-library-in-python-linux-version/
https://docs.python.org/2/extending/extending.html

*/

//---------------------------------------------------------------------------
#endif

