all: wordylib64.dll wordylib32.dll

wordylib64.dll: .\Release\wordydll64.o .\Release\uWordyReader64.o .\Release\uWordy64.o
	 g++ -shared -m64 -imacros wordydll.h -o wordylib64.dll .\Release\wordydll64.o .\Release\uWordyReader64.o .\Release\uWordy64.o

.\Release\wordydll64.o: wordydll.cpp
	 g++ -shared -m64 -imacros wordydll.h -c wordydll.cpp -o .\Release\wordydll64.o

.\Release\uWordyReader64.o: uWordyReader.cpp
	 g++ -shared -m64 -imacros wordydll.h -c uWordyReader.cpp -o .\Release\uWordyReader64.o

.\Release\uWordy64.o: uWordy.cpp
	 g++ -shared -m64 -imacros wordydll.h -c uWordy.cpp -o .\Release\uWordy64.o
   
wordylib32.dll: .\Release\wordydll32.o .\Release\uWordyReader32.o .\Release\uWordy32.o
	 g++ -shared -m32 -imacros wordydll.h -o wordylib32.dll .\Release\wordydll32.o .\Release\uWordyReader32.o .\Release\uWordy32.o

.\Release\wordydll32.o: wordydll.cpp
	 g++ -shared -m32 -imacros wordydll.h -c wordydll.cpp -o .\Release\wordydll32.o

.\Release\uWordyReader32.o: uWordyReader.cpp
	 g++ -shared -m32 -imacros wordydll.h -c uWordyReader.cpp -o .\Release\uWordyReader32.o

.\Release\uWordy32.o: uWordy.cpp
	 g++ -shared -m32 -imacros wordydll.h -c uWordy.cpp -o .\Release\uWordy32.o   
     
clean:
	 rm wordylib64.dll .\Release\wordydll64.o .\Release\uWordyReader64.o .\Release\uWordy64.o
   rm wordylib32.dll .\Release\wordydll32.o .\Release\uWordyReader32.o .\Release\uWordy32.o