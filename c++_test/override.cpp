#include <iostream>
#include <string>

using namespace std;

class A
{
private:
    int data;
public:
    A(int a);
};

A::A(int a)
{
    data = a;
}

class B
{
private:
    string data;
public:
    B(string b);
};

B::B(string b):data(b)
{

}


// A CreateAB() // the first function...
// {
// cout << "call A";
//     return A(3);
// }

// B CreateAB() // ... and its "underload"
// {
// cout << "call B";
// return B("3");
// }

struct CreateAB
{
    A operator()() // the first function...
    {
        cout << "call A";
        return A(3);
    }

    B operator()() // ... and its "underload"
    {
        cout << "call B";
        return B("3");
    }
};
 
int main(int argc, char* argv[])
{
    A a = CreateAB();
    B b = CreateAB();
    return 0;
}