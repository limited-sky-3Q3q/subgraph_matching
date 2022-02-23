#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
__global__ void kernel(void) {}
int main() {
    kernel << <1, 1 >> > ();
    printf("Hello world!\n");
    return 0;
}