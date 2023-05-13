#include<iostream>
using namespace std;

class Base
{
public:
	virtual void fun()
	{
		cout << "Base" << endl;
	}
};
class Derive :public Base
{
public:
	virtual void fun()
	{
		cout << "Derive" << endl;
	}
};
int main(void)
{
	//都是安全的
	Base* base = new Base();
	Derive* derive = new Derive();
	base = reinterpret_cast<Base*>(derive);
	base->fun();
	derive->fun();
	derive = reinterpret_cast<Derive*>(base);
	base->fun();
	derive->fun();
	return 0;
}