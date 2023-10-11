#include<iostream>
#include<string>
#include<algorithm>
using namespace std;

int a[100000];

int init = []{
    ios_base::sync_with_stdio(false);
    // cin.tie(nullptr);
    
    string s = "[-3,1,5,6,8,9,4,52,26,5,7,-5]";
    // for (string s;  cout << '\n') {
        // if (s.length() == 2) {
        //     cout << 0;
        //     continue;
        // }
        int n = 0;
        for (int _i = 1, _n = s.length(); _i < _n; ++_i) {
            bool _neg = false;
            if (s[_i] == '-') ++_i, _neg = true;
            int v = s[_i++] & 15;
            while ((s[_i] & 15) < 10) v = v * 10 + (s[_i++] & 15);
            if (_neg) v = -v;
            a[n++] = v;
        }
        sort(a, a + n);
        int ans = 0;
        for (int i = 0; i < n;) {
            int i0 = i;
            for (++i; i < n && a[i - 1] + 1 >= a[i]; ++i);
            ans = max(ans, a[i - 1] - a[i0] + 1);
        }
        cout << ans;
    // }
    cout.flush();
    exit(0);
    return 0;
}();

int main() {

}
// [1,5,6,8,9,4,52,26,5,7,-5]