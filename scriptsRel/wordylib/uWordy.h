//---------------------------------------------------------------------------

#ifndef uWordyH
#define uWordyH

#include <tchar.h>
#include <stdio.h>
#include <conio.h>
#include <windows.h>
#include <vector>
#include <algorithm>
#include <string>
using namespace std;

class TWordyTransformator
{
  public:
  TWordyTransformator();
  string fAllowed;                             //26
  string fTransfFrom;
  string fTransfTo;
  string fSeparators;//separators of words
  string fEndings;//things to be ignored at the end of words
  bool fBegEndSeparately;//true = beg and end words get their own datas. false - together (nothing happens).

  bool IsDiff(char *a,char *b);
  void ProcWord(char *xSrc,int xSMaxLen,char *xTrgt);  //to word without things, lowercase and such
  bool ProcWordDict(char *xSrc,int xSMaxLen,char *xTrgt, int xBinRepr, int &xLargestFreq);  //to readable representation (not binary)
  char *FindWord(char *xSrc);
  char *FindTextCol(char *xSrc,int xLen, int xIthTextCol);
  char *FindCSVCol(char *xSrc,int xLen, int xIthCol);
};

class STupl
{
  public:
  string xRep;
  string xOrig;
  STupl(string a,string b){xRep=a;xOrig=b;};
};
extern int Procstr(string &l, string &r);
extern int Proctup(STupl &l, STupl &r);


//---------------------------------------------------------------------------

#endif
