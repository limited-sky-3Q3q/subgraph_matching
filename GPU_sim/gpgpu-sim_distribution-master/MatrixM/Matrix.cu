
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#define Row 100
#define Col 100

__global__ void matrix_mul_gpu(int *M, int* N, int* P, int width)
{
    int sumNum = threadIdx.x + threadIdx.y*10 ;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int sum = 0;

    for(int k=0;k<width;k++)
    {
        int a = M[j*width+k];
        int b = N[k*width+i];
        sum += a*b;
    }
    P[sumNum] = sum;
}
__global__ void passMessage(int dstX, int dstY, int srcX,int srcY,int* data, int dataSize){
    int para1 = srcX *10000000 + srcY*100000 + dstX*1000+dstY * 10 ;
    for(int i = 0; i<dataSize;i++){
        // printf("\n[%d %d] -> [%d %d]\n", srcX, srcY, dstX, dstY);
        printf("\npara: %d\n", para1);
        asm("addc.s32 %0, %1, %2;" : "=r"(data[i]) : "r"(para1) , "r"(data[i]));//函数有问题，写入的数据会自动覆盖原数据
    }
}
void readMessage( int srcX,int srcY,int dstX,int dstY,int*data,int dataSize){
   
    char * fileName = new char[100];
    sprintf(fileName,"./buffer%d_%d_%d_%d",srcX,srcY,dstX,dstY);
    std::ifstream file(fileName);
    int tmpdata = 0;
    for(int i = 0;i<dataSize;i++)
    {
        file>>tmpdata;
        printf("\ndata[%d]: %d", i, tmpdata);
        data[i] += tmpdata;
    }
    file.close();
}

int srcX,srcY;
int main(int argc, char** argv)
{
    printf("\n---------------------start---------------------\n");
    srcX=atoi(argv[1]);
    srcY=atoi(argv[2]);

    struct timeval start, end;
    gettimeofday( &start, NULL );

    int *A = (int *)malloc(sizeof(int) * Row * Col);
    int *B = (int *)malloc(sizeof(int) * Row * Col);
    int *C = (int *)malloc(sizeof(int) * Row * Col);
    //malloc device memory
    int *d_dataA, *d_dataB, *d_dataC;
    cudaMalloc((void**)&d_dataA, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataB, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataC, sizeof(int) *Row*Col);
    //set value
    for (int i = 0; i < Row*Col; i++) {
        A[i] = rand() % 51;
        B[i] = rand() % 51;
    }

    cudaMemcpy(d_dataA, A, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, B, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(10,10);
    dim3 blockNumber(1);
    printf("\n---------------------matrix_mul_gpu run---------------------\n");
    matrix_mul_gpu << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);
    printf("\n---------------------matrix_mul_gpu done---------------------\n");

    cudaMemcpy(C, d_dataC, sizeof(int) * Row * Col, cudaMemcpyDeviceToHost);
    printf("\n---------------------passMessage/readMessage run---------------------\n");

    if(srcX != 0 || srcY != 0)
    {
        passMessage << <1,2>> > (0,0,srcX,srcY,d_dataC,5);
    }
    else
    {
        char ready = '0';
        printf("\n---------------------0---------------------\n");
        std::cin >>ready;
        printf("\n---------------------1---------------------\n");
        readMessage(srcX,srcY,0,1,C,5);
        printf("\n---------------------2---------------------\n");
        readMessage(srcX,srcY,1,0,C,5);
        printf("\n---------------------3---------------------\n");
        readMessage(srcX,srcY,1,1,C,5);
        printf("\n---------------------4---------------------\n");
    }
    printf("\n---------------------passMessage/readMessage done---------------------\n");
    //拷贝计算数据-一级数据指针

    // for(int ii=0; ii<100; ii++){
    //     printf("\ndata[%d]: %d", ii, C[ii]);
    // }

    //释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);
    printf("\n---------------------done---------------------\n");
    return 0;
}
