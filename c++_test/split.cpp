#include<iostream>
#include<string.h>

using namespace std;

int main() {
    char s[100];
    cin.getline(s, 100);

    // Pointer to point the word returned by the strtok() function.
    char * p;
    // Here, the delimiter is white space.
    p = strtok(s, "->");
    
    while( p != NULL ) {
        printf( "%s\n", p );
        p = strtok(p, s);
    }
}