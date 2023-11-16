#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int float_power_of_2(int x) {
    int e;
    float f = frexp((float)x, &e);
    return 1 << (e - 1);
}

int main() {
    int x[] = {5, 7, 8, 9, 15, 16, 17};

    for (int a : x) {
        int y = float_power_of_2(a);
        printf("(%d, %d) ", a, y);
    }
    printf("\n");
    return 0;
}