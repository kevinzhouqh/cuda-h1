#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <sys/time.h>
double getTimeStamp() {
	struct timeval tv;
	gettimeofday( &tv, NULL );
	return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

void h_addmat(float*A, float*B, float*C, int nx, int ny)
{
	int total = nx*ny;
	int count = 0;
	int i;
	for(i=0; i<total; i++)
	{
		C[count] = A[count] + B[count];
		count++;
	}
	return;
}

__global__ void f_addmat( float*A, float*B, float*C, int nx, int ny) {
	int ix = threadIdx.x + blockIdx.x*blockDim.x ;
	int iy = threadIdx.y + blockIdx.y*blockDim.y ;
	int idx = ix*ny + iy;
	if ( (ix<nx) && (iy<ny))
		C[idx] = A[idx] + B[idx];
}

int main( int argc, char *argv[] ) {
	if (argc != 3)
	{
		printf("Error: wrong number\n");
		exit(0);
	}

	int nx = atoi ( argv[1] );
	int ny = atoi (argv[2] );
	int noElems = nx*ny;
	int bytes = noElems * sizeof(float);
	int i,j, count;
	count = 0;
	float *h_A = (float *) malloc ( bytes );
	float* h_B = (float *) malloc ( bytes );
	float *h_hC = (float *) malloc ( bytes );
	float *h_dC = (float *) malloc ( bytes );
	
	for (i=0; i<nx; i++)
		for (j=0; j<ny; j++)
		{
			h_A[count] = (float)(i+j)/3.0;
			count++;
		}
	count = 0;
	for (i=0; i<nx; i++)
		for (j=0; j<ny; j++)
		{
			h_B[count]= (float)3.14*(i+j);
			count++;
		}
        float *d_A, *d_B, *d_C ;
	cudaMalloc( (void **) &d_A, bytes);
	cudaMalloc( (void **) &d_B, bytes);
	cudaMalloc( (void **) &d_C, bytes);
	double timeStampA = getTimeStamp();
	cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice);
	double timeStampB = getTimeStamp();
	dim3 block(128, 8);
	dim3 grid((nx + block.x-1)/block.x, (ny+block.y-1)/block.y);
	f_addmat<<<grid, block>>>( d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();
	double timeStampC = getTimeStamp();
	cudaMemcpy(h_dC, d_C, bytes, cudaMemcpyDeviceToHost );
	double timeStampD = getTimeStamp();
	cudaFree( d_A ); cudaFree( d_B); cudaFree( d_C);
	cudaDeviceReset();
	h_addmat(h_A, h_B, h_hC, nx, ny);
	count = 0;
	bool s = true;
	for(i=0; i<noElems; i++)
	{
		if( h_hC[i] != h_dC[i] )
		{
			s = false;
			printf("%d \n", i);
			break;
		}
	}
	if(s)
	{
		printf("total time is %f, CPU GPU transfer time is %f, kernel time is %f, GPU CPU transfer time is %f\n ", timeStampD-timeStampA, timeStampB - timeStampA, timeStampC- timeStampB, timeStampD - timeStampC);
		exit(0);
	}
	printf("finished");
	return 0;

}
