//---------------------------------------------------------------------------

#pragma hdrstop

#include "uWordy.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)
//---------------------------------------------------------------------------
                                   //now can affect only nonbinary repr, binary repr is false by default
/*
human readable version:
is it sufficient to store just one digit per character count in word?
*/
TWordyTransformator::TWordyTransformator()
{                     //defaults:
  fAllowed=string("abcdefghijklmnopqrstuvwxyz");                             //26
  fTransfFrom=string("ÏöË¯û˝·ÌÈ˙˘ÔÛùÚABCDEFGHIJKLMNOPQRSTUVWXYZÃä»ÿé›¡Õ…⁄Ÿœ”ç“");
  fTransfTo=string("escrzyaieuudotnabcdefghijklmnopqrstuvwxyzescrzyaieuudotn");
  fSeparators=string(" .,;\"\'\n(){}[]!?+*/\\%^");//separators of words
  fEndings=string("\n");//things to be ignored at the end of words
  fBegEndSeparately=false;//true = beg and end words get their own datas. false - together (nothing happens).
}

bool TWordyTransformator::IsDiff(char *a,char *b)
{
  if (strlen(a)!=strlen(b)) return true;
  for (int i=0;i<strlen(a) && i<strlen(b);i++)
  {
    char xNow1=a[i];
    const char *xDoTransf=strrchr(fTransfFrom.c_str(),xNow1);
    if (xDoTransf)
      xNow1=fTransfTo[xDoTransf-fTransfFrom.c_str()];
    char xNow2=b[i];
    const char *xDoTransf2=strrchr(fTransfFrom.c_str(),xNow2);
    if (xDoTransf2)
      xNow2=fTransfTo[xDoTransf2-fTransfFrom.c_str()];
    if (xNow1!=xNow2) return true;
  }
  return false;
}
void TWordyTransformator::ProcWord(char *xSrc,int xSMaxLen,char *xTrgt)  //to word without things, lowercase and such
{
  int xT=0;
  for (int i=0;i<xSMaxLen>0?xSMaxLen:strlen(xSrc);i++)
  {
    char xNow=xSrc[i];
    const char *xDoTransf=strrchr(fTransfFrom.c_str(),xNow);
    if (xDoTransf)
      xNow=fTransfTo[xDoTransf-fTransfFrom.c_str()];

    const char *xWhich=strrchr(fAllowed.c_str(),xNow);
    if (xWhich)
    {
      xTrgt[xT]=xNow;
      xT++;
    }
  }
  xTrgt[xT]=0;
}

bool TWordyTransformator::ProcWordDict(char *xSrc,int xSMaxLen,char *xTrgt, int pBinRepr, int &xLargestFreq)  //to readable representation (not binary)
{                                                    //0-readable,1-binary 2*26, 2-binary 26+2
  if (xSMaxLen<=0)
  {
    for (int i=0;i<strlen(xSrc);i++)
    {
      char xNow=xSrc[i];
      if (strrchr(fEndings.c_str(),xNow))
      {
        xSrc[i]=0;//end of word
        break;
      }
    }
  }

  BYTE xFill='0';
  if (pBinRepr>=1)
    xFill=0;

  int xReprLen;
  switch (pBinRepr)
  {
    case 1:  xReprLen=(fAllowed.length()*2);//todo could be compressed...
    break;
    case 2:
    case 0:  xReprLen=fAllowed.length()+2;
    break;
  }

  for (int i=0;i<xReprLen;i++)
    xTrgt[i]=xFill;

  xTrgt[xReprLen]=0;

  int xMP=xSMaxLen!=0?xSMaxLen:strlen(xSrc);

  bool xAtLeastOneAllowedChar=false;
  BYTE xFirstAllowed=0;
  BYTE xLastAllowed=0;

  for (int i=0;i<xMP;i++)
  {
    char xNow=xSrc[i];
    const char *xDoTransf=strrchr(fTransfFrom.c_str(),xNow);
    if (xDoTransf)
      xNow=fTransfTo[xDoTransf-fTransfFrom.c_str()];

    const char *xWhich=strrchr(fAllowed.c_str(),xNow);
    if (xWhich)
    {
      xTrgt[xWhich-fAllowed.c_str()]+=1;

      if (xFirstAllowed==0)
        xFirstAllowed=xWhich-fAllowed.c_str();

      xLastAllowed=xWhich-fAllowed.c_str();

      xAtLeastOneAllowedChar=true;
    }
    /*else
    {      //ignored character
      char xP[2];
      xP[0]=xNow;
      xP[1]=0;
      printf(xP);
      xSrc[i]=0;
    }   */
  }
  for (int i=0;i<fAllowed.length();i++)
  {
    if (xTrgt[i]-xFill > xLargestFreq)
      xLargestFreq=xTrgt[i]-xFill;
  }


  if (xAtLeastOneAllowedChar)
  {
    int xSavOffset=0;
    if (pBinRepr==0)
      xSavOffset=fAllowed[0];

    if (pBinRepr==1)
      xTrgt[fAllowed.length()+xFirstAllowed]+=1;
    else
      xTrgt[fAllowed.length()]=xSavOffset+xFirstAllowed;//beginning

    if (pBinRepr==0)
      xTrgt[fAllowed.length()+xLastAllowed]+=1;
    else
    {
      xTrgt[fAllowed.length()+1]=xSavOffset+xLastAllowed;//ending
      //swapping, is it sufficient to store them ordered?
      if (!fBegEndSeparately && xTrgt[fAllowed.length()+1] < xTrgt[fAllowed.length()])
      {
        xTrgt[fAllowed.length()+1]=xTrgt[fAllowed.length()];
        xTrgt[fAllowed.length()]=xSavOffset+xLastAllowed;
      }
    }
  }

  return xAtLeastOneAllowedChar;
}
char *TWordyTransformator::FindWord(char *xSrc)
{
  bool xEatenNormal=false;
  bool xWasSep=false;
  for (int i=0;i<strlen(xSrc);i++)
  {
    xWasSep=false;
    for (int j=0;j<sizeof(fSeparators);j++)
    {
      if ((xSrc)[i]==fSeparators[j])        //we have the word
      {
        //ProcWord(*xSrc,i-1,xTrgt);//i is separator, co the max length is i
        if (xEatenNormal)
          return (xSrc)+i;//and return position of separator

        xWasSep=true;
        break;
      }
    }
    if (!xWasSep)
      xEatenNormal=true;
  }
  return NULL;
}
char *TWordyTransformator::FindCSVCol(char *xSrc,int xLen, int xIthCol)
{
  const char xIndentables[]="\"\'";
  const int xIndtSz=2;
  int xIndented=0;
  int xDepth=0;

  for (int i=0;i<xLen-1;i++)
  {
    if (xIthCol<=0) return xSrc;
    if (xSrc[0]==0) return NULL;
    //jumping over inputs in commas
    if (xDepth==0)
    {
      for (int j=0;j<xIndtSz;j++)
      {
        if (xIndentables[j]==xSrc[0])
        {
          xIndented=j;
          xDepth++;
          break;
        }
      }
    }
    else
    {
      if (xSrc[0]==xIndentables[xIndented])
        xDepth--;
    }
    //end jumps

    if (xDepth==0 && (xSrc[0]==';' || xSrc[0]==','))        //the pattern is either ;" or ,"
    {
      xSrc=xSrc+1;
      xIthCol--;
    }
    else
      xSrc=xSrc+1;
  }
  return NULL;
}
char *TWordyTransformator::FindTextCol(char *xSrc,int xLen, int xIthTextCol)
{
  if (xIthTextCol<=0) return xSrc;
  int xF=0;
  do
  {
    bool xFnd=false;
    for (int i=0;i<xLen-1;i++)//xFt=strstr(xSrc,xCSVDel);
    {
      if (xSrc[0]==0 || xSrc[1]==0) return NULL;
      if (i>0 && xSrc[1]=='\"' && (xSrc[0]==';' || xSrc[0]==','))        //the pattern is either ;" or ,"
      {
        xSrc=xSrc+2;
        xFnd=true;
        break;
      }
      else
        xSrc=xSrc+1;
    }
    if (!xFnd || !xSrc || xSrc[0]==0) return NULL;

    xF++;
    if (xF==xIthTextCol) return xSrc;
  } while (true);
}

int Procstr(string &l, string &r)
{
  return strcmp(l.c_str(),r.c_str());
}
int Proctup(STupl &l, STupl &r)
{
  return strcmp(l.xRep.c_str(),r.xRep.c_str());
}

