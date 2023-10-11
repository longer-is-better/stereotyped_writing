#include<iostream>
#include<map>
#include<vector>
using namespace std;

int main() {
    map<int, int> height_map{
        {1, 111},
        {5, 555},
        {8, 555},
        {10, 555}
    };
    cout << height_map.lower_bound(2)->first << endl;
    cout << height_map.upper_bound(6)->first << endl;

}





// #include <iostream>
// #include <map>
// using namespace std;

// int main ()
// {
//   std::map<char,int> mymap;
//   std::map<char,int>::iterator itlow,itup;

//   mymap['a']=20;
//   mymap['b']=40;
//   mymap['c']=60;
//   mymap['d']=80;
//   mymap['e']=100;

//   itlow=mymap.lower_bound ('b');  // itlow points to b
//   cout << itlow->first << endl;
//   itup=mymap.upper_bound ('d');   // itup points to e (not d!)
//   cout << itup->first << endl;

//   mymap.erase(itlow,itup);        // erases [itlow,itup)

//   // print content:
//   for (std::map<char,int>::iterator it=mymap.begin(); it!=mymap.end(); ++it)
//     std::cout << it->first << " => " << it->second << '\n';

//   return 0;
// }