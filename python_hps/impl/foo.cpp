//libfibonacci
#include <cmath>

#define API_DESC extern "C"

API_DESC int foo(const int val)
{
    float result = 0.0f;
    for(int c=0;c<1000;c++)
    {
        for(int i=0;i<val;i++)
        {
            result += (i);
            result += sqrt((float)(i*i));
            result += pow((float)(i*i*i),0.1f);
        }
    }
    return (int)result;
}
