/*
 * ChunkTesting.cuh
 *
 *  Created on: Jan 5, 2016
 *      Author: matan
 */

#ifndef CHUNKTESTING_CUH_
#define CHUNKTESTING_CUH_

#include "ChunkPOC.cuh"

#include <cstdio>

#define IRRED_SIZE 32

extern unsigned int irreducibles[][5];


unsigned int calcLog(unsigned int n)
{
	unsigned int i = 0;
	while (n != 1)
	{
		n /= 2;
		++i;
	}
	return i;
}

unsigned int irreducibleIndex(unsigned int fieldDegree)
{
	for (unsigned int i = 0 ; i < IRRED_SIZE ; ++i)
	{
		if (irreducibles[i][4] == fieldDegree)
		{
			return i;
		}
	}
	printf("Field degree %d is not set in irreducibles!\n",fieldDegree);
	exit(-1);
}

template <unsigned int N>
float testChunksMultiply(unsigned int size)
{
	unsigned int * pentanomialCoefficients = irreducibles[irreducibleIndex(SIZE)];
	setPentanomial(pentanomialCoefficients);
	unsigned int (*chunksAHost)[ROUNDED(N)] = new unsigned int[size][ROUNDED(N)];
	unsigned int (*chunksBHost)[ROUNDED(N)] = new unsigned int[size][ROUNDED(N)];

	for(unsigned int i = 0 ; i < size ; ++i)
	{
		for(unsigned int j = 0 ; j < ROUNDED(N) ; ++j)
		{
			chunksAHost[i][j]=0;
			chunksBHost[i][j]=0;
		}
		chunksAHost[i][N/2]=2;
		chunksBHost[i][N/2]=2;
	}

	unsigned int (*chunksA)[ROUNDED(N)];
	unsigned int (*chunksB)[ROUNDED(N)];

	cudaMalloc(&chunksA, sizeof(unsigned int)*ROUNDED(N)*size);
	cudaMalloc(&chunksB, sizeof(unsigned int)*ROUNDED(N)*size);

	cudaMemcpy(chunksA, chunksAHost, sizeof(unsigned int)*ROUNDED(N)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(chunksB, chunksBHost, sizeof(unsigned int)*ROUNDED(N)*size, cudaMemcpyHostToDevice);

	unsigned int blocksNum = (size*GROUP_SIZE(N)+THREAD_BLOCK_SIZE(N)-1)/THREAD_BLOCK_SIZE(N);
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	finiteFieldMultiplyArrays<N><<<
			blocksNum,
			THREAD_BLOCK_SIZE(N)>>>
			((unsigned int (*)[ROUNDED(N)])chunksA, reinterpret_cast<unsigned int (*)[ROUNDED(N)]>(chunksB), (unsigned int (*)[ROUNDED(N)])chunksA, size);
	cudaEventRecord(end);
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(chunksAHost, chunksA, sizeof(unsigned int)*N*size, cudaMemcpyDeviceToHost);


	float ms;
	cudaEventElapsedTime(&ms, start, end);
	delete[] chunksAHost;
	delete[] chunksBHost;

	cudaFree(chunksA);
	cudaFree(chunksB);
	return ms;
}

float testChunksMultiply64Ring(unsigned int size)
{
	unsigned int * pentanomialCoefficients = irreducibles[irreducibleIndex(SIZE)];
	setPentanomial(pentanomialCoefficients);
	unsigned int (*chunksAHost)[64] = new unsigned int[size][64];
	unsigned int (*chunksBHost)[64] = new unsigned int[size][64];
	unsigned int (*chunksCHost)[128] = new unsigned int[size][128];

	for(unsigned int i = 0 ; i < size ; ++i)
	{
		for(unsigned int j = 0 ; j < 64 ; ++j)
		{
			chunksAHost[i][j]=0;
			chunksBHost[i][j]=0;
		}
		chunksAHost[i][32]=2;
		chunksBHost[i][32]=2;
	}

	unsigned int (*chunksA)[64];
	unsigned int (*chunksB)[64];
	unsigned int (*chunksC)[128];

	cudaMalloc(&chunksA, sizeof(unsigned int)*64*size);
	cudaMalloc(&chunksB, sizeof(unsigned int)*64*size);
	cudaMalloc(&chunksC, sizeof(unsigned int)*128*size);

	cudaMemcpy(chunksA, chunksAHost, sizeof(unsigned int)*64*size, cudaMemcpyHostToDevice);
	cudaMemcpy(chunksB, chunksBHost, sizeof(unsigned int)*64*size, cudaMemcpyHostToDevice);

	unsigned int blocksNum = (size*GROUP_SIZE_RING+MAX_THREADBLOCK_SIZE-1)/MAX_THREADBLOCK_SIZE;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	performMult64<<<
			blocksNum,
			MAX_THREADBLOCK_SIZE>>>
			((unsigned int (*)[64])chunksA, reinterpret_cast<unsigned int (*)[64]>(chunksB), (unsigned int (*)[128])chunksC, size);
	cudaEventRecord(end);
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(chunksCHost, chunksC, sizeof(unsigned int)*128*size, cudaMemcpyDeviceToHost);

	float ms;
	cudaEventElapsedTime(&ms, start, end);
	delete[] chunksAHost;
	delete[] chunksBHost;
	delete[] chunksCHost;

	cudaFree(chunksA);
	cudaFree(chunksB);
	cudaFree(chunksC);
	return ms;
}
#endif /* CHUNKTESTING_CUH_ */

