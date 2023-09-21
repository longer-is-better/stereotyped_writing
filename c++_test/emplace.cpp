#include <vector> 
#include <iostream> 
using namespace std;
class testDemo
{
public:
    testDemo(int num):num(num){
        std::cout << "调用构造函数" << endl;
    }
    testDemo(const testDemo& other) :num(other.num) {
        std::cout << "调用拷贝构造函数" << endl;
    }
    testDemo(testDemo&& other) :num(other.num) {
        std::cout << "调用移动构造函数" << endl;
    }



    int num;
};
int main()
{
    cout << "emplace_back:" << endl;
    std::vector<testDemo> demo1;
    demo1.emplace_back(2);  
    testDemo t1(2);
    // demo1.emplace_back(move(t1));
    testDemo t2 = 222;
    t2 = move(t1);
    // demo1.emplace_back(testDemo(1));
    t1.num = 999;
    cout << "t1 " << t1.num << endl;
    cout << "t2 " << t2.num << endl;
    // cout << demo1.front().num << endl;

    int T[] = {1,2,4,5,6};

    // cout << "push_back:" << endl;
    // std::vector<testDemo> demo2;
    // demo2.push_back(2);
}