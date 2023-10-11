#include <iostream>
#include <string>
#include <algorithm>
using namespace std;
 
bool isSpace(char x) { return x == ' '; }
 
int main()
{
	string s2("1 1 1 1 1 ");
	cout << "remove_if之前"<<s2 << endl;
	auto i = remove_if(s2.begin(), s2.begin() + 6, isSpace);
	cout <<"remove_if之后"<< s2 << endl;
    cout <<"i之后"<< string(i, s2.end()) << endl;
    i->erase()
	return 0;
}