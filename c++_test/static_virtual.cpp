#include <iostream>
using namespace std;
class Base {
public:
    Base(int a) {
        val = a;
        cout << "Base construct fun" << endl;
    }
    static virtual void fun() {
        cout << "Base static fun" << endl;
    }
    int val;
};
class Son : public Base {
public:
    Son(int a) : Base(a) {//由于父类构造函数立有参数，这里必须采用列表初始化调用基类，构建基类
        cout << "Son construct fun" << endl;
    }
    static void fun() {
        cout << "Son static fun" << endl;
    }
};
int main() {
    Base base(1);
    Son son(2);
    Base* p = new Son(3);
    base.fun();
    son.fun();
    p->fun();
    return 0;
}

