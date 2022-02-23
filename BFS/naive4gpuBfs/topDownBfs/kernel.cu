#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void 
Kernel( Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_graph_visited, int* g_cost, bool *g_over, int no_of_nodes, int num_of_threads_per_block) {
    /*****************************
    1*no_of_nodes/4
    2*no_of_nodes/4
    3*no_of_nodes/4
    4*no_of_nodes/4
    *****************************/
    printf("\n %d", num_of_threads_per_block);
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;//在block里的id
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		g_graph_visited[tid]=true;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
			{
			int id = g_graph_edges[i];
			if(!g_graph_visited[id])
				{
				g_cost[id]=g_cost[tid]+1;
				g_graph_mask[id]=true;
                if(tid < no_of_nodes/4){// 0 0 
                    printf("\n 0 0!");
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(10) , "r"(id));
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(1000) , "r"(id));
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(1010) , "r"(id));
                }
                else if(tid >= no_of_nodes/4 && tid < 2*no_of_nodes/4){// 0 1
                    printf("\n 0 1!");
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(100000) , "r"(id));
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(101000) , "r"(id));
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(101010) , "r"(id));
                }
                else if(tid >= 2*no_of_nodes/4 && tid < 3*no_of_nodes/4){// 1 0
                    printf("\n 1 0!");
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(10000000) , "r"(id));
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(10000010) , "r"(id));
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(10001010) , "r"(id));
                }
                else if(tid >= 3*no_of_nodes/4 && tid < 4*no_of_nodes/4){// 1 1
                    printf("\n 1 1!");
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(10100000) , "r"(id));
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(10100010) , "r"(id));
                    asm("addc.s32 %0, %1, %2;" : "=r"(id) : "r"(10101000) , "r"(id));
                }

				//Change the loop stop value such that loop continues
				*g_over=true;
				}
			}
	}

}

#endif 
