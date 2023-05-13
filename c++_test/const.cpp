class A
{
private:
    int data;
public:
    A(/* args */);
    ~A();
    void get() const {
        data = 2;
    }
};

A::A(/* args */)
{
}

A::~A()
{
}
