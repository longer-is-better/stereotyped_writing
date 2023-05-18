#include<iostream>
using namespace std;

int main() {
    int a = 3;
    int& r_a = a;
    a = a >> 1;
    cout << a;
}