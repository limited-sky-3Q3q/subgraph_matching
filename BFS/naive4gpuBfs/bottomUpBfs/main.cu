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


__global__ void
assignJobs(bool *g_change, int num_of_nodes){
	int chipletId = blockIdx.x;// 0, 1, 2
    int subSegmentBegin, subSegmentEnd;//子chiplet要处理的部分, [subSegmentBegin, subSegmentEnd)
    int parameters=0;
    switch(chipletId){
        case 0:// chiplet2(0, 1)
            subSegmentBegin = 0;
            subSegmentEnd = num_of_nodes/3;
            parameters = 000000010;
            break;
        case 1:// chiplet3(1, 0)
            subSegmentBegin = num_of_nodes/3;
            subSegmentEnd = 2*num_of_nodes/3;
            parameters = 000001000;
            break;
        case 2:// chiplet4(1, 1)
            subSegmentBegin = 2*num_of_nodes/3;
            subSegmentEnd = num_of_nodes;
            parameters = 000001010;
            break;
        default:
            break;
    }

    asm("addc.s32 %0, %1, %2;" : "=r"(subSegmentBegin) : "r"(parameters) , "r"(subSegmentBegin));
    asm("addc.s32 %0, %1, %2;" : "=r"(subSegmentEnd) : "r"(parameters) , "r"subSegmentEnd);
}

__global__ void
synchronizeGraphInfo(int *d_graph_visited){
	int chipletId = blockIdx.x;// 0, 1, 2
    int subSegmentBegin, subSegmentEnd;//子chiplet要处理的部分, [subSegmentBegin, subSegmentEnd)
    int parameters=0;
    switch(chipletId){
        case 0:// chiplet2(0, 1)
            subSegmentBegin = 0;
            subSegmentEnd = num_of_nodes/3;
            parameters = 000000010;
            break;
        case 1:// chiplet3(1, 0)
            subSegmentBegin = num_of_nodes/3;
            subSegmentEnd = 2*num_of_nodes/3;
            parameters = 000001000;
            break;
        case 2:// chiplet4(1, 1)
            subSegmentBegin = 2*num_of_nodes/3;
            subSegmentEnd = num_of_nodes;
            parameters = 000001010;
            break;
        default:
            break;
    }

    asm("addc.s32 %0, %1, %2;" : "=r"(subSegmentBegin) : "r"(parameters) , "r"(subSegmentBegin));
    asm("addc.s32 %0, %1, %2;" : "=r"(subSegmentEnd) : "r"(parameters) , "r"subSegmentEnd);
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
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*num_of_nodes);
    for(int i=0; i<num_of_nodes; i++)
        h_graph_visited[i]=false;
    //Copy the Visited nodes array to device memory
    bool* d_graph_visited;
    cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*num_of_nodes) ;
    cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*num_of_nodes, cudaMemcpyHostToDevice) ;

    bool *d_change;//记录是否发生状态更新
    cudaMalloc( (void**) &d_change, sizeof(bool));
    bool change;
    int k=0;
	//Call the Kernel untill all the elements of Frontier are not false
    do{//每次循环搜索同一层
		//if no thread changes this value then the loop stops
        change=false;
		cudaMemcpy(d_change, &change, sizeof(bool), cudaMemcpyHostToDevice);
		assignJobs<<<3, 1>>>(d_change, num_of_nodes);//chiplet2,3,4
        synchronizeGraphInfo<<<1, 3>>>(d_graph_visited);//将frontier set 同步更新

		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cudaMemcpy(&change, d_change, sizeof(bool), cudaMemcpyDeviceToHost);
        k++;
	}
    while(change);

    return;
}