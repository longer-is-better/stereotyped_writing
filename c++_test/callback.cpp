#include<iostream>
using namespace std;
void Add(int(*callbackfun)(int, int), int a, int b)
{
	cout << callbackfun(a, b) << endl;
}
int add(int a, int b)
{
	return a + b;
}
int main()
{
	Add(add, 1, 2);
    return 0;
}
