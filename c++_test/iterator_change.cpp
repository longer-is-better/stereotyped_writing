#include<iostream>
#include<vector>
using namespace std;


ostream& operator<<(ostream& os, const vector<string>& vs) {
    os << "["; for (auto& s: vs) os << s << ", "; os << "]";
    return os;
}

int main() {
    vector<string> v{"1", "22", "333", "4444", "55555", "666666"};
    auto i1 = v.begin();
    auto i2 = i1 + 1;
    auto i3 = i2 + 1;
    
    cout << v << *i1 << *i2 << " " << *i3;
    v.erase(i1);
    cout << v << *i1 << *i2 << " " << *i3;
}