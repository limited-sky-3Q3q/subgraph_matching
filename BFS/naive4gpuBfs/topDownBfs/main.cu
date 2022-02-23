/***********************************************************************************
简单的top-down BFS算法
并行运算单个节点的多个子节点
本程序为了简化代码，使用邻接矩阵来存储图信息
************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 256

int no_of_nodes;
int edge_list_size;
FILE *fp;

struct Node{// 存储边的起点和数量
    int starting;
    int no_of_edges;
    int farther;
};


int **edges = NULL;// 图的邻接矩阵

#include "kernel.cu"

void TopDownBFS(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	printf("\n-----------------------------top-down BFS start-----------------------------\n");
	no_of_nodes=0;
	edge_list_size=0;
	
    TopDownBFS(argc, argv);// 运行BFS

	printf("\n----------------------------- top-down BFS end -----------------------------\n");
	return 0;
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void TopDownBFS( int argc, char** argv) {
    //init
	printf("\n-----------------------------Reading File-----------------------------\n");
	static char *input_file_name;
	if (argc == 2 ) {
		input_file_name = argv[1];
		printf("Input file: %s\n", input_file_name);
	}
	else{// 未指定文件路径时默认读取SampleGraph.txt
        printf("Without specific, use default input file: SampleGraph.txt\n");
		input_file_name = "/home/sr/GPGPU-Sim/Chiplet-GPGPU-Sim-MessagePassing/BFS_singal/data/SampleGraph.txt";
		printf("No input file specified, defaulting to SampleGraph.txt\n");
	}
    // read graph from input_file_name
    fp = fopen(input_file_name,"r");
	if(!fp){
	printf("Error Reading graph file\n");
	return;
	}
	
	int source = 0;// 源节点
	fscanf(fp,"%d",&no_of_nodes);//读取节点数量

    edges = new int* [no_of_nodes];// 创建邻接表
    for(int i=0; i<no_of_nodes; i++)
        edges[i] = new int[no_of_nodes];

    for(int i=0; i<no_of_nodes; i++)//初始化邻接表
        for(int j=0; j<no_of_nodes; j++)
            edges[i][j]=0;
    
    
	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;
	
	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK){
	num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK);//向上取整
	num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}
		
	// allocate host memory
    Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
    bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
    
    int start, edgeno;   
    // initalize the memory
    for( unsigned int i = 0; i < no_of_nodes; i++){
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_nodes[i].farther = -1;
        h_graph_mask[i]=false;
        h_graph_visited[i]=false;
    }
    
    //read the source node from the file
    fscanf(fp,"%d",&source);//读取源节点
    //set the source node as true in the mask
    h_graph_mask[source]=true;
    
    fscanf(fp,"%d",&edge_list_size);//读取边的数量
     
    int id,cost;
    // 边的信息按照一维数组存储
    int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
    for(int i=0; i < edge_list_size ; i++){
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
    }
     
	if(fp)fclose(fp);    
	
	/************
    复制数据到显存
    *************/ 

	//Copy the Node list to device memory
    Node* d_graph_nodes;
    cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
    cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice);

	//Copy the Edge List to device Memory
	int* d_graph_edges;
    cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
    cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);

    //Copy the Mask to device memory
    bool* d_graph_mask;
    cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
    cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);
    
    //Copy the Visited nodes array to device memory
    bool* d_graph_visited;
    cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
    cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
    
    //记录最短距离
    // allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
	h_cost[i]=-1;
	h_cost[source]=0;

	// allocate device memory for result
    int* d_cost;
    cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
    cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

    //make a bool to check if the execution is over
    bool *d_over;
    cudaMalloc( (void**) &d_over, sizeof(bool));

	printf("-----------------------------Copied Everything to GPU memory-----------------------------\n");
    
    // setup execution parameters
    // dim3  grid( num_of_blocks, 1, 1);
    // dim3  threads( num_of_threads_per_block, 1, 1);
	dim3  grid(num_of_blocks);
	dim3  threads(num_of_threads_per_block);
	// printf("\grid: %d", grid);
	// printf("\threads: %d", threads);
	printf("\nnum_of_blocks: %d", num_of_blocks);
	printf("\nnum_of_threads_per_block: %d\n", num_of_threads_per_block);

	int k=0;
	printf("-----------------------------！！！！！！！！！！！！！！！！！！！！！！！！-----------------------------\n");
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
    do
    {
		//if no thread changes this value then the loop stops
		stop=false;
		cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;
		Kernel<<< grid, threads>>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_graph_visited, d_cost, d_over, no_of_nodes, num_of_threads_per_block);

		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
        k++;
	}
    while(stop);
    
    
    printf("Kernel Executed %d times\n",k);

    // copy result from device to host
    cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);

	//Stop the Timer
    //printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer));
    
	
	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
	fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");
	
	
    // cleanup memory
    free( h_graph_nodes);
    free( h_graph_edges);
    free( h_graph_mask);
    free( h_graph_visited);
    free( h_cost);
    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);
}
