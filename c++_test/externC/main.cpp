#include <iostream>
#include "myfun.h"
using namespace std;
int main(){
   display();
   return 0;
}

/*
gcc -c myfun.c
g++ main.cpp  -c
g++ main.o myfun.o


gcc -c myfun.c --shared -fpic -o libmyfun.so
g++ main.cpp  -c
g++ main.o -lmyfun -L.
*/