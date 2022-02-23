#include <iostream>
#include <stdio.h>
int main(){
    int a = 0;
    cudaGetDeviceCount(&a);
    int n = 000000010;
    asm("addc.u32 %0, %1, %2;" : "=r"(*n) : "r"(*m) , "r"(*n));

    printf("\n%d\n", a);
}