#include<iostream>
#include<cstring>
using namespace std;

int main (int argc, char** argv) {
    auto i = [&] {
        for (int i = 0; i < argc; i++) if (!strcmp(argv[i], "-h")) return i;
        return -1;
    } ();
    cout << i << endl;
    return 0;
}