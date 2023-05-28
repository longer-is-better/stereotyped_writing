# C++和C语言的区别
1.兼容：C语言是C++的子集，C++可以很好兼容C语言。但是C++又有很多新特性；  
2.又好又快：C++是面对对象的编程语言，支持封装、继承、多态特性，程序结构清晰、易于扩充、程序可读性好，运行效率高、仅比汇编语言慢10%~20%；C语言是面对过程的编程语言；  
3.又安全：C语言有一些不安全的语言特性，如指针使用的潜在危险、强制转换的不确定性、内存泄露等。而C++对此增加了不少新特性来改善安全性，如四类 cast 转换、智能指针、try-catch等等；  
4.可复用性高：C++引入了模板的概念，后面在此基础上，实现了方便开发的标准模板库STL。C++的STL库相对于C语言的函数库更灵活、更通用。   
5.C++是不断发展的语言，C++11中新引入了nullptr、auto变量、Lambda匿名函数、右值引用、智能指针。   
# C++中 struct 和 class 的区别
1.意义上：struct 一般用于描述一个数据结构集合，而 class 是对一个对象数据的封装；  
2.默认权限：struct 中默认的访问控制权限是 public 的，而 class 中默认的访问控制权限是 private 的；  
默认继承：在继承关系中，struct 默认是公有继承，而 class 是私有继承；  
3.模板：class 关键字可以用于定义模板参数，就像 typename，而 struct 不能用于定义模板参数。  
# include头文件的顺序以及双引号""和尖括号<>的区别
## 区别：
尖括号<>的头文件是系统文件，双引号""的头文件是自定义文件；
编译器预处理阶段查找头文件的路径不一样；
## 查找路径;
使用尖括号<>的头文件的查找路径：编译器设置的头文件路径–>系统变量;  
使用双引号""的头文件的查找路径：当前头文件目录–>编译器设置的头文件路径–>系统变量。
# C++结构体和C结构体的区别
C的结构体内不允许有函数存在，C++允许有内部成员函数，且允许该函数是虚函数；  
C的结构体对内部成员变量的访问权限只能是public，而C++允许public,protected,private三种；  
C中使用结构体需要加上 struct 关键字，或者对结构体使用 typedef 取别名，而 C++ 中可以省略 struct 关键字直接使用；  
C语言的结构体是不可以继承的，C++的结构体是可以从其他的结构体或者类继承过来的。  
# 导入C函数的关键字是什么，C++编译时和C有什么不同？
关键字：在C++中，导入C函数的关键字是extern，表达形式为extern “C”， extern "C"的主要作用就是为了能够正确实现C++代码调用其他C语言代码。加上extern "C"后，会指示编译器这部分代码按C语言的进行编译，而不是C++的。  
编译区别：由于C++支持函数重载，因此编译器编译函数的过程中会将函数的参数类型也加到编译后的代码中，而不仅仅是函数名；而C语言并不支持函数重载，因此编译C语言代码的函数时不会带上函数的参数类型，一般只包括函数名。
# C++ 函数重载不能区分返回值类型
## 原因：无法进行类型推导
## trick
但是通过重载 struct operator X () 可以
```
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
   operator A () // the first function...
   {
    cout << "call A";
      return A(3);
   }

   operator B () // ... and its "underload"
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
```
其实是骗你的，这个是 cast type operator 类型转换操作
# 简述C++从代码到可执行二进制文件的过程
预编译、编译、汇编、链接
## 预编译：这个过程主要的处理操作如下： gcc -E demo.c -o demo.i
（1） 将所有的#define删除，并且展开所有的宏定义  
（2） 处理所有的条件预编译指令，如#if、#ifdef  
（3） 处理#include预编译指令，将被包含的文件插入到该预编译指令的位置。  
（4） 过滤所有的注释  
（5） 添加行号和文件名标识  
## 编译：这个过程主要的处理操作如下：gcc -S demo.c(i) -o demo.s
（1） 词法分析：将源代码的字符序列分割成一系列的记号。  
（2） 语法分析：对记号进行语法分析，产生语法树。  
（3） 语义分析：判断表达式是否有意义。  
（4） 源代码优化：  
（5） 目标代码生成：生成汇编代码。  
（6） 目标代码优化  
## 汇编：gcc -c demo.s -o test.o
这个过程主要是将汇编代码转变成机器可以执行的指令。
## 链接：gcc democ.o -o democ.exe
将不同的源文件产生的目标文件进行链接，从而形成一个可以执行的程序。
​ 链接分为静态链接和动态链接。  
​ (1) 静态链接，是在链接的时候就已经把要调用的函数或者过程链接到了生成的可执行文件中，就算你在去把静态库删除也不会影响可执行程序的执行；生成的静态链接库，Windows下以.lib为后缀，Linux下以.a为后缀。  
​ (2) 而动态链接，是在链接的时候没有把调用的函数代码链接进去，而是在执行的过程中，再去找要链接的函数，生成的可执行文件中没有函数代码，只包含函数的重定位信息，所以当你删除动态库时，可执行程序就不能运行。生成的动态链接库，Windows下以.dll为后缀，Linux下以.so为后缀。  
# 全局和局部，static和非static
https://www.runoob.com/w3cnote/cpp-static-usage.html  
## 什么是static?
static 是 C/C++ 中很常用的修饰符，它被用来控制变量的存储方式和可见性。  
## 为什么引入 static
想将函数某变量的值保存至下一次调用，又需要将变量的访问范围控制在函数内部。  

## 一览表

|存储位置<br>初始化|全局|函数内|类内|
|:---:|:---:|:---:|:---:|
|static|变量：全局数据区<br>（非0初始化在data,<br>否则初始化为0且在bss）<br>函数：代码段|变量：同左，<br>（若多次执行初始化语<br>句仅第一次生效）<br>函数：不能嵌套定义|变量：同左，<br>类内定义，类外初始化<br>函数：代码段|
|非 static|变量：全局数据区<br>函数：代码段|（普通）变量：随所属栈帧<br>函数：不能嵌套定义|（普通）变量：随所属对象(堆/栈)<br>函数：代码段|

|变量：生存周期|全局|函数内|类内|
|:---:|:---:|:---:|:---:|
|static|程序始终|程序始终|程序始终|
|非 static|程序始终|随所属栈帧|随所属对象(堆/栈)|

|作用域|全局|函数内|类内|
|:---:|:---:|:---:|:---:|
|static|文件内部，**即使用 extern 修饰时也不可以跨文件访问**<br>除非拿到指针或者通过非 static 间接调用|函数内|类内<br>派生类若不覆盖定义则和父类共享<br>派生类若覆盖定义则各自控制自己的那个<br>不能为虚函数（编译阶段报错）|
|非 static|外部链接性，默认文件内部， extern 修饰时可以跨文件访问|函数内|普通|

# 可执行程序组成及内存布局
## 代码段（Code）
由机器指令组成，该部分是不可改的，编译之后就不再改变，放置在文本段（.text）。

## 数据段（Data）
常量（constant），通常放置在只读read-only的文本段（.text）  
静态数据（static data），初始化的放置在数据段（.data）；未初始化的放置在（.bss，Block tarted by Symbol，BSS段的变量只有名称和大小却没有值）  
动态数据（dynamic data），这些数据存储在堆（heap）或栈（stack）
# 数组和指针的区别
http://c.biancheng.net/view/vip_2018.html  
http://c.biancheng.net/view/vip_2019.html
## 概念：
C语言标准规定，当数组名作为数组定义的标识符（也就是定义或声明数组时）、sizeof 或 & 的操作数时，它才表示整个数组本身，在其他的表达式中，数组名会被转换为指向第 0 个元素的指针（地址）。
（1）数组：数组是用于储存多个相同类型数据的集合。数组名是首元素的地址。  
（2）指针：指针相当于一个变量，但是它和不同变量不一样，它存放的是其它变量在内存中的地址。指针名指向了内存的首地址。
## 区别：
数组和指针不等价的一个典型案例就是求数组的长度
```
#include <stdio.h>

int main(){
    int a[6] = {0, 1, 2, 3, 4, 5};
    int *p = a;
    int len_a = sizeof(a) / sizeof(int);
    int len_p = sizeof(p) / sizeof(int);
    printf("len_a = %d, len_p = %d\n", len_a, len_p);
    return 0;
}
// len_a = 6, len_p = 1 
```
赋值：同类型指针变量可以相互赋值；数组不行，只能一个一个元素的赋值或拷贝；
类型：int[] 和 int*  
存储方式：  
数组：数组在内存中是连续存放的，开辟一块连续的内存空间。数组是根据数组的下标进行访问的，数组的存储空间，不是在静态区就是在栈上。  
指针：指针很灵活，它可以指向任意类型的数据。指针的类型说明了它所指向地址空间的内存。由于指针本身就是一个变量，再加上它所存放的也是变量，所以指针的存储空间不能确定。
# 什么是函数指针，如何定义函数指针，有什么使用场景
概念：函数指针就是指向函数的指针变量。每一个函数都有一个入口地址，该入口地址就是函数指针所指向的地址。  
```
int func(int a); 
int (*f)(int a); 
f = &func;
```
使用场景： 回调（callback）。我们调用别人提供的 API函数(Application Programming Interface,应用程序编程接口)，称为Call；如果别人的库里面调用我们的函数，就叫Callback。
```
//以库函数qsort排序函数为例，它的原型如下：
void qsort(void *base,//void*类型，代表原始数组
           size_t nmemb, //第二个是size_t类型，代表数据数量
           size_t size, //第三个是size_t类型，代表单个数据占用空间大小
           int(*compar)(const void *,const void *)//第四个参数是函数指针
          );
//第四个参数告诉qsort，应该使用哪个函数来比较元素，即只要我们告诉qsort比较大小的规则，它就可以帮我们对任意数据类型的数组进行排序。在库函数qsort调用我们自定义的比较函数，这就是回调的应用。

//示例
int num[100];
int cmp_int(const void* _a , const void* _b){//参数格式固定
    int* a = (int*)_a;    //强制类型转换
    int* b = (int*)_b;
    return *a - *b;　　
}
qsort(num,100,sizeof(num[0]),cmp_int); //回调
```
# 静态变量什么时候初始化
对于C语言的全局和静态变量，初始化发生在任何代码执行之前，属于编译期初始化。
而C++标准规定：全局或静态对象当且仅当对象首次用到时才进行构造。
# nullptr调用成员函数可以吗？为什么？
可以。因为在编译时对象就绑定了函数地址，和指针空不空没关系。   
C++程序在内存中执行时候，类方法和普通函数没有太大区别，都是在代码段，只是成员函数通过C++名字mangle算法生成了新的名字。对象执行方法时候 对象->方法 其实是 类方法找对象数据（通过隐形的this指针）调用breath(*this), this就等于pAn。由于函数中没有需要解引用this的地方，所以函数运行不会出错，但是若用到this，因为this=nullptr，运行出错。
```
//给出实例
class animal{
public:
    void sleep(){ cout << "animal sleep" << endl; }
    void breathe(){ cout << "animal breathe haha" << endl; }
};
class fish :public animal{
public:
    void breathe(){ cout << "fish bubble" << endl; }
};
int main(){
    animal *pAn=nullptr;
    pAn->breathe();   // 输出：animal breathe haha
    fish *pFish = nullptr;
    pFish->breathe(); // 输出：fish bubble
    return 0;
}  
```
# 什么是野指针，怎么产生的，如何避免？
野指针就是指针指向的位置是不可知的（随机的、不正确的、没有明确限制的）  
## 原因：
（1）指针没有被初始化，任何指针变量被刚创建时不会被自动初始化为NULL指针。  
（2）释放内存后指针不及时置空（野指针），依然指向了该内存，那么可能出现非法访问的错误。  
（3）指针超过了变量的作用范围，越界或指向已被释放的临时对象  
## 避免办法：
（1）指针变量在创建的同时应当被初始化分配内存（使用malloc函数、calloc函数或new操作符），要么将指针设置为NULL，要么让它指向合法的内存。  
（2）指针释放后将指针置为NULL  
（3）使用智能指针
# 内联函数和宏函数的区别

||处理时机|本质|二义性|参数检查<br>类型转换等|实现方式|注意|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|宏函数|预编译|宏|很可能有|无|预编译器直接文本替换||
|内联函数|编译|函数|无|有|编译器谨慎的替换|仅节省出入栈时间，函数太大没必要内联，浪费内存|


# new和malloc的区别，各自底层实现原理
||本质|构造和析构|参数|返回|失败|
|:---:|:---:|:---:|:---:|:---:|:---:|
|new|操作符|是|类型|相应指针|异常|
|malloc|库函数|否|大小|void*|返回null|
# const 和 define
http://c.biancheng.net/view/vip_2198.html  
http://c.biancheng.net/view/2230.html  
http://c.biancheng.net/view/2232.html  
http://c.biancheng.net/view/vip_7678.html  
define是预编译指令，const是普通变量的定义，define定义的宏是在预处理阶段展开的，而const定义的只读变量是在编译运行阶段使用的。  
const定义的是变量，而define定义的是常量。define定义的宏在编译后就不存在了，它不占用内存，因为它不是变量，系统只会给变量分配内存。但const定义的常变量本质上仍然是一个变量，具有变量的基本属性，有类型、占用存储单元。可以说，常变量是有名字的不变量，而常量是没有名字的。有名字就便于在程序中被引用，所以从使用的角度看，除了不能作为数组的长度，用const定义的常变量具有宏的优点，而且使用更方便。所以编程时在使用const和define都可以的情况下尽量使用常变量来取代宏。  
const定义的是变量，而宏定义的是常量，所以const定义的对象有数据类型，而宏定义的对象没有数据类型。所以编译器可以对前者进行类型安全检查，而对后者只是机械地进行字符替换，没有类型安全检查。这样就很容易出问题，即“边际问题”或者说是“括号问题”。

C++中的 const 更像编译阶段的 #define  
C++中全局 const 变量的可见范围是当前文件  
C++ const常量如何在多文件编程中使用？  
1. 将const常量定义在.h头文件中  
2. 借助extern先声明再定义const常量  
3. 0借助extern直接定义const常量  

# const 和引用
const 能加尽加  
http://c.biancheng.net/view/vip_2254.html  
http://c.biancheng.net/view/vip_2255.html  
给引用添加 const 限定后，不但可以将引用绑定到临时数据 
还可以将引用绑定到类型相近的数据，这使得引用更加灵活和通用，它们背后的机制都是临时变量。  
## 引用类型的函数形参请尽可能的使用 const
使用 const 可以避免无意中修改数据的编程错误；  
使用 const 能让函数接收 const 和非 const 类型的实参，否则将只能接收非 const 类型的实参；  
使用 const 引用能够让函数正确生成并使用临时变量。  


# C++中函数指针和指针函数的区别
指针函数本质是一个函数，其返回值为指针。
```
int *fun(int x,int y);
```
函数指针本质是一个指针，其指向一个函数。
```
int (*fun)(int x,int y)
```
# 常量指针和指针常量
```
1. const int a;     //指的是a是一个常量，不允许修改。
2. const int *a;    //a指针所指向的内存里的值不变，即（*a）不变
3. int const *a;    //同const int *a;
4. int *const a;    //a指针所指向的内存地址不变，即a不变
5. const int *const a;   //都不变，即（*a）不变，a也不变
```
# 使用指针需要注意什么？
定义指针时，先初始化为NULL。  
用malloc申请内存之后，应该立即检查指针值是否为NULL。防止使用指针值为NULL的内存。  
不要忘记为数组和动态内存赋初值。防止将未被初始化的内存作为右值使用。  
避免数字或指针的下标越界，特别要当心发生“多1”或者“少1”操作  
动态内存的申请与释放必须配对，防止内存泄漏  
用free或delete释放了内存之后，立即将指针设置为NULL，防止“野指针”  
# 指针和引用
http://c.biancheng.net/view/vip_2252.html  
其实引用只是对指针进行了简单的封装，它的底层依然是通过指针实现的，引用占用的内存和指针占用的内存长度一样，在 32 位环境下是 4 个字节，在 64 位环境下是 8 个字节，之所以不能获取引用的地址，是因为编译器进行了内部转换。  
不是引用不占用内存，而是编译器不让获取它的地址。  
C++ 的发明人 Bjarne Stroustrup 也说过，他在 C++ 中引入引用的直接目的是为了让代码的书写更加漂亮，尤其是在运算符重载中，不借助引用有时候会使得运算符的使用很麻烦。
## 其他区别
1) 引用必须在定义时初始化，并且以后也要从一而终，不能再指向其他数据；而指针没有这个限制，指针在定义时不必赋值，以后也能指向任意数据。  
2) 可以有 const 指针，但是没有 const 引用。因为 r 本来就不能改变指向，加上 const 是多此一举。  
3) 指针可以有多级，但是引用只能有一级，例如，int **p是合法的，而int &&r是不合法的。如果希望定义一个引用变量来指代另外一个引用变量，那么也只需要加一个&  
4) 指针和引用的自增（++）自减（--）运算意义不一样。对指针使用 ++ 表示指向下一份数据，对引用使用 ++ 表示它所指代的数据本身加 1；自减（--）也是类似的道理。  

# 运算符重载
## 为什么要以全局函数的形式重载 +
这样做是为了保证 + 运算符的操作数能够被对称的处理，C++ 只会对成员函数的参数进行类型转换，而不会对调用成员函数的对象进行类型转换
```
    Complex c2 = c1 + 15.6;
    Complex c3 = 28.23 + c1;
    /* 全局： Complex(28.23) + c1
       成员函数： (28.23).operator+(c1) */
```

# 赋值和初始化
在定义的同时进行赋值叫做初始化（Initialization），定义完成以后再赋值（不管在定义的时候有没有赋值）就叫做赋值（Assignment）。

# 构造函数
## 拷贝控制：三五法则
C++89 拷贝构造 赋值 析构 C++11 转移构造 转移赋值； 一般，需要自定义析构函数的类，都需要其他几个。
默认  
初始化
拷贝
转换
转移
# 左值和右值
右值就是不能寻址的值？  
左值的英文简写为“lvalue”，右值的英文简写为“rvalue”。很多人认为它们分别是"left value"、"right value" 的缩写，其实不然。lvalue 是“loactor value”的缩写，可意为存储在内存中、有明确存储地址（可寻址）的数据，而 rvalue 译为 "read value"，指的是那些可以提供数据值的数据（不一定可以寻址，例如存储于寄存器中的数据）。
# 简述C++有几种传值方式，之间的区别是什么？
值传递：形参即使在函数体内值发生变化，也不会影响实参的值；  
引用传递：形参在函数体内值发生变化，会影响实参的值；  
指针传递：在指针指向没有发生改变的前提下，形参在函数体内值发生变化，会影响实参的值；  
# 四种cast类型转换
http://c.biancheng.net/view/2343.html

|关键字|说明|
|----|----|
|static_cast|用于良性转换，一般不会导致意外发生，风险很低。|
|const_cast|用于 const 与非 const、volatile 与非 volatile 之间的转换。|
|reinterpret_cast|高度危险的转换，这种转换仅仅是对二进制位的重新解释，不会借助已有的转换规则对数据进行调整，但是可以实现最灵活的 C++ 类型转换。|
|dynamic_cast|借助 RTTI，用于类型安全的向下转型（Downcasting）。|

## const_cast
1.去除const属性，将只读变为读写  
2.针对常量指针、常量引用和常量对象
```
const char *p;
char *p1 = const_cast<char*>(p);
```
## static_cast
static_cast 在编译期间转换，转换失败的话会抛出一个编译错误，能够更加及时地发现错误。
原理：继承链（Inheritance Chain）。只能用于良性转换，这样的转换风险较低，一般不会发生什么意外，例如：  
1. 原有的自动类型转换，例如 short 转 int、int 转 double、const 转非 const、向上转型等；  
2. void 指针和具体类型指针之间的转换，例如void *转int *、char *转void *等；  
3. 有转换构造函数或者类型转换函数的类与其它类型之间的转换，例如 double 转 Complex（调用转换构造函数）、Complex 转 double（调用类型转换函数）。  
```
#include <iostream>
#include <cstdlib>
using namespace std;

class Complex{
public:
    Complex(double real = 0.0, double imag = 0.0): m_real(real), m_imag(imag){ }
public:
    operator double() const { return m_real; }  //类型转换函数
private:
    double m_real;
    double m_imag;
};

int main(){
    //下面是正确的用法
    int m = 100;
    Complex c(12.5, 23.8);
    long n = static_cast<long>(m);  //宽转换，没有信息丢失
    char ch = static_cast<char>(m);  //窄转换，可能会丢失信息
    int *p1 = static_cast<int*>( malloc(10 * sizeof(int)) );  //将void指针转换为具体类型指针
    void *p2 = static_cast<void*>(p1);  //将具体类型指针，转换为void指针
    double real= static_cast<double>(c);  //调用类型转换函数
   
    //下面的用法是错误的
    float *p3 = static_cast<float*>(p1);  //不能在两个具体类型的指针之间进行转换
    p3 = static_cast<float*>(0X2DF9);  //不能将整数转换为指针类型

    return 0;
}
```
## reinterpret_cast
reinterpret 是“重新解释”的意思，顾名思义，reinterpret_cast 这种转换仅仅是对二进制位的重新解释，不会借助已有的转换规则对数据进行调整，非常简单粗暴，所以风险很高。  
reinterpret_cast 可以认为是 static_cast 的一种补充，一些 static_cast 不能完成的转换，就可以用 reinterpret_cast 来完成，例如两个具体类型指针之间的转换、int 和指针之间的转换（有些编译器只允许 int 转指针，不允许反过来）。  
```
#include <iostream>
using namespace std;

class A{
public:
    A(int a = 0, int b = 0): m_a(a), m_b(b){}
private:
    int m_a;
    int m_b;
};

int main(){
    //将 char* 转换为 float*
    char str[]="http://c.biancheng.net";
    float *p1 = reinterpret_cast<float*>(str);
    cout<<*p1<<endl;
    //将 int 转换为 int*
    int *p = reinterpret_cast<int*>(100);
    //将 A* 转换为 int*
    p = reinterpret_cast<int*>(new A(25, 96));
    cout<<*p<<endl;
   
    return 0;
}
```

## dynamic_cast
dynamic_cast 用于在类的继承层次之间进行类型转换，它既允许向上转型（Upcasting），也允许向下转型（Downcasting）。向上转型是无条件的，不会进行任何检测，所以都能成功；向下转型的前提必须是安全的，要借助 RTTI 进行检测，所有只有一部分能成功。  
dynamic_cast 与 static_cast 是相对的，dynamic_cast 是“动态转换”的意思，static_cast 是“静态转换”的意思。dynamic_cast 会在**程序运行期间**借助 RTTI 进行类型转换，这就要求**基类必须包含虚函数**；
# 继承中的访问
编译器通过指针来访问成员变量，指针指向哪个对象就使用哪个对象的数据；  
编译器通过指针的类型来访问成员函数，指针属于哪个类的类型就使用哪个类的函数。
# 虚析构和内存泄漏
基类和派生类都申请了资源  
基类指针指向派生类对象  
delete 释放这个指针指向的对象的时候  
因为析构函数不是虚函数  
只会调用基类的析构函数释基类的资源，派生类申请的资源会泄露  
# 浅拷贝还是深拷贝
如果一个类拥有指针类型的成员变量，那么绝大部分情况下就需要深拷贝，因为只有这样，才能将指针指向的内容再复制出一份来，让原有对象和新生对象相互独立，彼此之间不受影响。如果类的成员变量没有指针，一般浅拷贝足以。
浅拷贝和double free
# 拷贝构造函数参数为什么一定是引用
函数调用的值传递形式是一次复制，传递过程本身就会调用复制构造函数，导致死循环
# 检查内存泄漏
## valgrid
## asan






















# 移动构造函数
http://c.biancheng.net/view/7847.html




# 内存重叠的情况下，如何实现strcpy？
```
// 无内存重叠
char * strcpy(char *dst,const char *src)   // remember const
{
    assert(dst != NULL && src != NULL);    // check
    char *ret = dst;  // dst need ++
    while ((*dst++=*src++)!='\0'); // 赋值，检查‘\0’，++
    return ret;  // 支持链式表达式
}

内存可能重叠
char * strcpy(char *dst,const char *src)
{
    assert(dst != NULL && src != NULL);
    char *ret = dst;
    if (dst >= src) {  // dst 在 src 之后，可能存在重叠，从后往前复制
        while (src++ != '\0') dst++;
        while (dst >= ret) {
            *dst-- = *src--;
        }
    } else {
        while ((*dst++=*src++)!='\0');
    }
    return ret;
}
```
# c++进程间通信？
管道(Pipe)：管道可用于具有亲缘关系进程间的通信，允许一个进程和另一个与它有共同祖先的进程之间进行通信。  
命名管道(named pipe)：命名管道克服了管道没有名字的限制，因此，除具有管道所具有的功能外，它还允许无亲缘关系进程间的通信。命名管道在文件系统中有对应的文件名。命名管道通过命令mkfifo或系统调用mkfifo来创建。  
信号(Signal)：信号是比较复杂的通信方式，用于通知接受进程有某种事件发生，除了用于进程间通信外，进程还可以发送信号给进程本身;Linux除了支持Unix早期信号语义函数sigal外，还支持语义符合Posix.1标准的信号函数sigaction(实际上，该函数是基于BSD的，BSD为了实现可靠信号机制，又能够统一对外接口，用sigaction函数重新实现了signal函数)。  
信号量(semaphore)：主要作为进程间以及同一进程不同线程之间的同步手段。   
消息(Message)队列：消息队列是消息的链接表，包括Posix消息队列system V消息队列。有足够权限的进程可以向队列中添加消息，被赋予读权限的进程则可以读走队列中的消息。消息队列克服了信号承载信息量少，管道只能承载无格式字节流以及缓冲区大小受限等缺  
共享内存：使得多个进程可以访问同一块内存空间，是最快的可用IPC形式。是针对其他通信机制运行效率较低而设计的。往往与其它通信机制，如信号量结合使用，来达到进程间的同步及互斥。  
套接字(Socket)：更为一般的进程间通信机制，可用于不同机器之间的进程间通信。起初是由Unix系统的BSD分支开发出来的，但现在一般可以移植到其它类Unix系统上：Linux和System V的变种都支持套接字。  


## C++ 实现单例模式
