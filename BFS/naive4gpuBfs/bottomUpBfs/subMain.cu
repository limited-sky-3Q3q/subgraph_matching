/*
位于chiplet1，只负责转发信息到其余chiplet中，不负责核心计算
其它chiplet都含有完整的图信息
输入参数：srcX srcY inputFilePath
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// #include "kernel.cu"

#define MAX_THREADS_PER_BLOCK 256

struct Node{
    int u;//即节点ID
    int edges;//该节点边的数量
};

int num_of_nodes;
int edge_list_size;
FILE *fp;

void BFSGraph(int argc, char** argv);

int main(int argc, char** argv){
    printf("\n-----------------------------chiplet1 code start-----------------------------\n");
    if (!(argc==4)){
        printf("srcX, srcY, input file is needed!\n");
	    return 0;
    }
    
    if(!(atoi(argv[1])==0 && atoi(argv[2])==0)){
        printf("\nthis is chiplet1, both srcX and srcY should be 0");
        return;
    }
	num_of_nodes=0;
	edge_list_size=0;
    
    BFSGraph(argc, argv);

    printf("\n-----------------------------chiplet1 code end -----------------------------\n");
	return 0;
}


void BFSGraph(int argc, char** argv){
    printf("\n-----------------------------chiplet1 Reading File-----------------------------\n");
	static char *input_file_name;

    input_file_name = argv[3];
    printf("Input file: %s\n", input_file_name);
	fp = fopen(input_file_name,"r");
    if(!fp){
        printf("Error Reading graph file\n");
        return;
	}

    //init
    int source = 0;
	fscanf(fp,"%d",&num_of_nodes);//节点数量
    int num_of_blocks = 1;
	int num_of_threads_per_block = num_of_nodes;
    if(num_of_nodes>MAX_THREADS_PER_BLOCK){
        num_of_blocks = (int)ceil(num_of_nodes/(double)MAX_THREADS_PER_BLOCK);//向上取整
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

    bool *d_change;
    cudaMalloc( (void**) &d_change, sizeof(bool));
    bool change;
    int k=0;
	//Call the Kernel untill all the elements of Frontier are not false
    do{//每次循环搜索同一层
		//if no thread changes this value then the loop stops
        change=false;
		cudaMemcpy(d_change, &change, sizeof(bool), cudaMemcpyHostToDevice);
		assignJobs<<<3, 1>>>(d_change);//chiplet2,3,4

		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cudaMemcpy(&change, d_change, sizeof(bool), cudaMemcpyDeviceToHost);
        k++;
	}
    while(change);

    return;
}