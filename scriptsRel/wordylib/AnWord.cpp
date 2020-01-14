//---------------------------------------------------------------------------
/*
knowhow:
it is enough to represent a word this way:

get first and last character
get frequencies of all characters (26 numbers) *
and then write last and first character [VARIANT B such that lexicographically first is first in the tuple]

*max number of characters in any word is less then 8 (cze, eng)

collisions - czech 63, eng - 6

the collisions dont change in variant B!!!

Number of neurons:
26+26+26 variant A
26+26 variant B (frequencies and zeroes where )

So things to try:
variantA x VariantB
Czech x English
Czech with diacritics x Czech without diacritics

*/
/*

  TODO:
  -unicode and diacritics

  g++ neumi output pro nenulovej out file!
  neumime kopirovat outputy knihovnou


  builder - 431821 processed, zahodil 21749
  22520 paragrafu
  nej input 11
  410072 in one input


  g++:
  431297 22604 reject
  */


#pragma hdrstop

#include "uWordy.h"
#include "uWordyReader.h"
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
#pragma argsused
int _tmain(int argc, _TCHAR* argv[])
{
  /*int xColumnProc=0;
  int xMode=0;          //0-process dictionary, 1-make dictionary (can eat CSV)
                        //todo decipher text option
  if (argc>=3)
    xMode=atoi(argv[2]);

  string xFileIn="Czech.3-2-3.dic";
  if (argc>=2)
    xFileIn=argv[1];

  if (argc>=4)
    xColumnProc=atoi(argv[3]);

  switch (xMode)
  {
    default:
    case 0:
    {
      FILE *f=fopen(xFileIn.c_str(),"r");
      FILE *t=fopen(string(xFileIn+string("Proc.dic")).c_str(),"w");

      vector<STupl> xSort;
      char xStor[64];
      char xProc[256];
      while (fgets(xStor,sizeof(xStor),f))
      {
        ProcWordDict(xStor,0,xProc,false);
        xSort.push_back(STupl(xProc,xStor));
        fputs(xProc,t);
        fputs("\n",t);
      }

      fclose(f);
      fclose(t);

      sort(xSort.begin(),xSort.end(),Proctup);
      t=fopen(string(xFileIn+"ProcSort.dic").c_str(),"w");
      FILE *coli=fopen(string(xFileIn+"ProcCollisions.dic").c_str(),"w");
      int xSame=0;
      for (int i=0;i<xSort.size();i++)
      {
        if (i<xSort.size()-1 && strcmp((char*)xSort[i].xRep.c_str(),(char*)xSort[i+1].xRep.c_str())==0 && IsDiff((char*)xSort[i].xOrig.c_str(),(char*)xSort[i+1].xOrig.c_str()))
        {
          fputs(xSort[i].xOrig.c_str(),coli);
          fputs(" ",coli);
          fputs(xSort[i+1].xOrig.c_str(),coli);
          fputs("\n",coli);
        }
        fputs(xSort[i].xRep.c_str(),t);
        fputs(" ",t);
        fputs(xSort[i].xOrig.c_str(),t);
        fputs("\n",t);
      }
      fclose(t);
      fclose(coli);
    }
    break;
    /*case 1://make dict. (todo: append)
    {
      FILE *f=fopen(xFileIn.c_str(),"r");
      char xStor[4*4096];//long lines of text possible?
      char xWord[512];
      TWordList xList;
      while (fgets(xStor,sizeof(xStor),f))
      {
        char *xCur=xStor;
        while (FindWord(&xCur,xWord,sizeof(xWord)))//found a word.
        {
          procword...
          xList.Append(xWord);               //yea yea all these things need to fit into memory.
        }
      }
      fclose(f);
      xList.Append(string(xFileIn+"-proc.dic").c_str());//print dictionary word by word.
    }  */                                                                      /*
    case 2://make into transformed data (not with all permutations. Yay.)
    case 3://uses like writes CSV thingies
    {
      unsigned int xWords=0;
      unsigned int xRejected=0;
      FILE *f=fopen(xFileIn.c_str(),"r");
      FILE *t=fopen(string(xFileIn+string("BinRep.bin")).c_str(),"w");
      FILE *tind=fopen(string(xFileIn+string("BinInd.bin")).c_str(),"w");
      char xStor[4*4096];//long lines of text possible?
      char xWord[512];
      char xProcced[128];
      const int xCodwordlen=2*strlen(xAllowed);//for bin repr.
      //unsigned int xLongest=0;
      //unsigned int xOver=0;

      while (fgets(xStor,sizeof(xStor),f))
      {
        char *xCur=xStor;
        if (xMode==3)
          xCur=FindTextCol(xCur,sizeof(xStor),xColumnProc);
        char *xEnd=FindWord(xCur);//,xWord,sizeof(xWord)

        //stat
        //if (strlen(xStor)>xLongest)
        //  xLongest=strlen(xStor);
        //if (xStor[xLongest-1]!='\n') xOver++;
        //endstat
        unsigned int xWordsNow=0;
        while (xEnd)//found a word.
        {
          if (ProcWordDict(xCur,xEnd-xCur,xWord,true))//to binary    true == has at least one allowed character
          {
            fwrite(xWord,xCodwordlen,1,t);
            xWordsNow++;
          }
          else xRejected++;

          xCur=xEnd+1;
          xEnd=FindWord(xCur);
          xWords++;
        }

        //save index:
        long xNow=ftell(t);//where are we now
        fprintf(tind,"%li %u \n",xWordsNow,xNow);//at this position in file begins this much words, strlen(xAllocated)*2 is the length per word.
      }
      fclose(f);
      fclose(t);
      fclose(tind);
      printf("done %i words, %i rejected \n",xWords,xRejected);
      //printf("longest line %u characters",xLongest);
    }
    break;
  }                                                                          */


  //test

  int batch_rows = 200;
  string OrigFileIn = "..\\Data\\doucka\\procords.csv";
  unsigned int CSVTextCol = 1;
  unsigned int NumInputWords = 64;
  unsigned int OutputCol=1;
  unsigned int NumOutputDatas=1;
  int handlein = -1;
  int totalinputs = -1;
  int nullflags=0;

  handlein = ExternInitNew((char*)OrigFileIn.c_str(),CSVTextCol,NumInputWords,OutputCol,NumOutputDatas);
  totalinputs = ExternCallback(handlein,NULL,NULL,0,nullflags);

  TYPERET xI[64*52];
  TYPERET xO[10];
  ExternMove(handlein,400);//so that pPosition is read next callback
  for (int i=0;i<100;i++)
  {
    ExternCallback(handlein,xI,xO,1,0);
    //if (xO[0]>0.1)
      printf("%f ",xO[0]);
  }

  ExternFree(handlein);

  getch();

  return 0;
}
//---------------------------------------------------------------------------
