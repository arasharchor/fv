#pragma warning(disable:4996)

#include <iostream>

using namespace std;

void prefeat(void);
void train(void);
void expect(void);

int main(void)
{
    //system("dir ORL\\*.pgm /a-d /o-n /b /s >datalist.ORL");

    //prefeat();

    train();

    //expect();

    system("shutdown -s -t 120");
    system("pause");

    return 0;
}
