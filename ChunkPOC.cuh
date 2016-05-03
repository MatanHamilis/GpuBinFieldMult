/*
 * ChunkPOC.cuh
 *
 *  Created on: Jan 7, 2016
 *      Author: matan
 */

#ifndef CHUNKPOC_CUH_
#define CHUNKPOC_CUH_



#define TYPE 1
#if TYPE == 2
#define GROUP_SIZE_RING 32
#else
#define GROUP_SIZE_RING 64
#endif
#define GROUP_SIZE(T) (WARP_SIZE*(((T)+63)/64))
#define GROUPS_PER_THREADBLOCK(T) (MAX_THREADBLOCK_SIZE / GROUP_SIZE(T))
#define THREAD_BLOCK_SIZE(T) (GROUP_SIZE(T)*GROUPS_PER_THREADBLOCK(T))
#define ROUNDED(T) ((((T)+63)/64)*64)


#include "utils.h"


extern __constant__ unsigned int pentanomial[5];

/*
 * @Brief: Sets a polynomial as the irreducible polynomial to be used in multiplication.
 * @Param: pentanomialInput - an array of length 5 that represents the powers i of x for which
 * 			the coefficient of x^i is not zero.
 */
__host__ void setPentanomial(unsigned int pentanomialInput[5]);

/*
 * @Brief: Internal multiplication function.
 * 		   performs the core shuffle-based polynomial multiplication of degree 64.
 * @Param cChunk - The output chunk to write the result to.
 * @Param myIdx - The index of the thread in the warp with which the function was called.
 * @Param a0 - The first entry of chunk A kept by this thread.
 * @Param a1 - The second entry of chunk A kept by this thread.
 * @Param b0 - The first entry of chunk B kept by this thread.
 * @Param b1 - The second entry of chunk B kept by this thread.
 */
__device__ inline void multiply64Shuffle(
		unsigned int* cChunk,
		const unsigned int myIdx,
		const unsigned int a0,
		const unsigned int a1,
		const unsigned int b0,
		const unsigned int b1);

/*
 * @Brief: Internal multiplication function.
 * 		   performs the core shared memory-based polynomial multiplication of degree 64.
 * @Param cChunk - The output chunk to write the result to.
 * @Param aChunk - The first chunk to multiply.
 * @Param bChunk - The second chunk to multiply.
 * @Param myIdx - The index of the thread in the warp that calls this function.
 */
__device__ inline void multiply64Shmem(
		unsigned int* aChunk,
		unsigned int* bChunk,
		unsigned int* cChunk,
		const unsigned int myIdx);

/*
 * @Brief: Multiplies cChunk by the pentanomial of degree 64. Writes result to resChunk.
 * @Param cChunk - The chunk to multiply.
 * @Param resChunk - The output chunk to which the result is written.
 * @Param myIdx - The index of the thread within the warp.
 */
__device__ inline void multiplyByPentanomial64(unsigned int* cChunk, unsigned int* resChunk, const unsigned int myIdx);

/*
 * @Brief: Multiplies cChunk by the pentanomial. Writes result to resChunk.
 * @Param cChunk - The chunk to multiply.
 * @Param resChunk - The output chunk to which the result is written.
 * @Param myIdx - The index of the thread within the warp.
 */
__device__ inline void multiplyByPentanomial(unsigned int* cChunk, unsigned int* resChunk, const unsigned int myIdx);

/*
 * @Brief: General finite field multiplication function.
 * @Param: aChunk - first chunk to multiply.
 * @Param: bChunk - second chunk to multiply.
 * @Param: cChunk - The chunk to which output is written.
 * @Param: temp - Temporary memory allocated by user to be used by this function.
 * @Param: myIdxInGroup - Index of the thread within the group of threads that perform the multiplication.
 * @Param: myIdxInWarp - Index of the thread within its warp.
 * @Param: warpInGroup - Index of the warp amongst all warps in the group.
 */
template<unsigned int N>
__device__ inline void finiteFieldMultiply(
	unsigned int aChunk[ROUNDED(N)],
	unsigned int bChunk[ROUNDED(N)],
	unsigned int cChunk[ROUNDED(N)],
	unsigned int temp[2*ROUNDED(N)],
	unsigned int myIdxInGroup,
	unsigned int myIdxInWarp,
	unsigned int warpInGroup);

template<>
__device__ inline void finiteFieldMultiply<32>(
	unsigned int aChunk[64],
	unsigned int bChunk[64],
	unsigned int cChunk[64],
	unsigned int temp[128],
	unsigned int myIdxInGroup,
	unsigned int myIdx,
	unsigned int warpInGroup)
{
	unsigned int a_reg;
	unsigned int e_reg;

	a_reg = aChunk[myIdx];

	e_reg = bChunk[myIdx];

	unsigned my_ans[2]={0};
	int b;
	for(unsigned k = 0 ; k < WARP_SIZE ; ++k){
		b= (myIdx>= k);
		my_ans[0] ^= (b*__shfl_up(a_reg,k)) & __shfl(e_reg,k);
		my_ans[1] ^= ((1-b)*__shfl_down(a_reg,WARP_SIZE-k))& __shfl(e_reg,k);
	}

	unsigned int c = 0;
	myIdx = (myIdx + 16) & (WARP_SIZE - 1);
#pragma unroll 4
	for(unsigned int i = 0 ; i < 4 ; ++i){
		b = (myIdx < 16);
		c = (myIdx >= pentanomial[i]) & (myIdx < pentanomial[i] + 16);

		my_ans[1] ^= ( b * c *__shfl(my_ans[1],16+myIdx-pentanomial[i]));
		my_ans[0]^=((1-b)* c *__shfl(my_ans[1],pentanomial[i]));
	}
	myIdx = (myIdx + 16) & (WARP_SIZE - 1);
#pragma unroll 4
	for(unsigned int i = 0 ; i < 4 ; ++i){
		c = (myIdx >= pentanomial[i]) & (myIdx < pentanomial[i] + 16);
		my_ans[0] ^= (c * __shfl(my_ans[1],myIdx-pentanomial[i]));
	}
		cChunk[myIdx] = my_ans[0];
}




template<unsigned int N>
__device__ inline void finiteFieldMultiply(
	unsigned int aChunk[ROUNDED(N)],
	unsigned int bChunk[ROUNDED(N)],
	unsigned int cChunk[ROUNDED(N)],
	unsigned int temp[2*ROUNDED(N)],
	unsigned int myIdxInGroup,
	unsigned int myIdxInWarp,
	unsigned int warpInGroup)
{
#ifdef SHUFFLE
	unsigned int my_a[2];
	unsigned int my_b[2];

	my_a[0] = aChunk[warpInGroup * 64 + myIdxInWarp];
	my_a[1] = aChunk[warpInGroup * 64 + myIdxInWarp + WARP_SIZE];

	aChunk[warpInGroup * 64 + myIdxInWarp] = 0;
	aChunk[warpInGroup * 64 + myIdxInWarp + WARP_SIZE] = 0;
#endif

#pragma unroll
	for (unsigned int i = 0 ; i < ROUNDED(N)/64 ; ++i)
	{
#ifdef SHUFFLE
		my_b[0] = bChunk[64*i + myIdxInWarp];
		my_b[1] = bChunk[64*i + myIdxInWarp + WARP_SIZE];
		__syncthreads();
		if(warpInGroup == 0)
		{
			bChunk[64*i + myIdxInWarp] = 0;
			bChunk[64*i + myIdxInWarp + WARP_SIZE] = 0;
		}
		__syncthreads();
		multiply64Shuffle(&aChunk[64*(i+warpInGroup)], myIdxInWarp, my_a[0], my_a[1], my_b[0], my_b[1]);
#else
		multiply64Shmem(&aChunk[warpInGroup * 64], &bChunk[64*i], &temp[64*(i+warpInGroup)], myIdxInWarp);
#endif
	}

	if(myIdxInGroup < (N-(N/2)))
	{
		multiplyByPentanomial(&temp[N - 1 + ((N)/2)], &temp[((N)/2)-1], myIdxInGroup);
	}
if(myIdxInGroup < (N/2)-1)
	{
		multiplyByPentanomial(&temp[N],&temp[0], myIdxInGroup);
	}

if(myIdxInGroup < ((N+1)/2))
	#pragma unroll 2
	for (unsigned int i = 0 ; i < 2 ; ++i)
	{
		cChunk[myIdxInGroup + i*(N/2)] = temp[myIdxInGroup + i*(N/2)];
	}
}

/*
 * @Brief: Multiplies an array of chunks.
 * @Param: aChunk - An array of chunks.
 * @Param: bChunk - An array of chunks.
 * @Param: cChunk - An array of chunks to which the output of the multiplication of a Chunk
 * 			and bChunk will be written.
 */
template<unsigned int N>
__global__ void finiteFieldMultiplyArrays(
		unsigned int aChunk[][ROUNDED(N)],
		unsigned int bChunk[][ROUNDED(N)],
		unsigned int cChunk[][ROUNDED(N)],
		unsigned int size);



template<unsigned int N>
__global__ void finiteFieldMultiplyArrays(
		unsigned int aChunk[][ROUNDED(N)],
		unsigned int bChunk[][ROUNDED(N)],
		unsigned int cChunk[][ROUNDED(N)],
		unsigned int size)
{
	__shared__ unsigned int aShared[(THREAD_BLOCK_SIZE(N)*4)/ROUNDED(N)][ROUNDED(N)];
	unsigned int (*bShared)[ROUNDED(N)] = reinterpret_cast<unsigned int (*)[ROUNDED(N)]>(&(aShared[(THREAD_BLOCK_SIZE(N)*2)/ROUNDED(N)][0]));
#ifndef SHUFFLE
	__shared__ unsigned int tempShared[(THREAD_BLOCK_SIZE(N)*2)/ROUNDED(N)][2*ROUNDED(N)];
#endif
	unsigned int warpGroupSize = GROUP_SIZE(N);
	unsigned int warpGroupIdx = (threadIdx.x + blockIdx.x * blockDim.x) / (warpGroupSize);
	unsigned int warpTBIdx = threadIdx.x / (warpGroupSize);
	unsigned int threadIdxInWarpGroup = threadIdx.x % (warpGroupSize);

	if(warpGroupIdx >= size)
	{
		return;
	}

	// Loading input into shared memory.
#pragma unroll 2
	for (unsigned int i = 0 ; i < 2 ; ++i)
	{
		aShared[warpTBIdx][threadIdxInWarpGroup + i*warpGroupSize] = aChunk[warpGroupIdx][threadIdxInWarpGroup + i*warpGroupSize];
		bShared[warpTBIdx][threadIdxInWarpGroup + i*warpGroupSize] = bChunk[warpGroupIdx][threadIdxInWarpGroup + i*warpGroupSize];
#ifndef SHUFFLE
		tempShared[warpTBIdx][threadIdxInWarpGroup + i*warpGroupSize] = 0;
		tempShared[warpTBIdx][threadIdxInWarpGroup + i*warpGroupSize + N] = 0;
#endif
	}
	__syncthreads();
	finiteFieldMultiply<N>(
			aShared[warpTBIdx],
			bShared[warpTBIdx],
			cChunk[warpGroupIdx],
#ifdef SHUFFLE
			aShared[warpTBIdx],
#else
			tempShared[warpTBIdx],
#endif
			threadIdxInWarpGroup,
			threadIdxInWarpGroup & (WARP_SIZE - 1),
			threadIdxInWarpGroup / WARP_SIZE);
}

__host__ void setPentanomial(unsigned int pentanomialInput[5])
{
	cudaMemcpyToSymbol(pentanomial, pentanomialInput, sizeof(unsigned int)*5);
}

__device__ inline void multiplyByPentanomial64(unsigned int* cChunk, unsigned int* resChunk, const unsigned int myIdx)
{
	unsigned int myC[2];
	myC[0] = cChunk[myIdx];
	myC[1] = cChunk[myIdx + WARP_SIZE];

#pragma unroll
	for (unsigned int i = 0  ; i < 4 ; ++i)
	{
		resChunk[pentanomial[i] + myIdx] ^= myC[0];
		resChunk[pentanomial[i] + myIdx + WARP_SIZE] ^= myC[1];
	}
}
__device__ inline void multiply64Shuffle(
		unsigned int* cChunk,
		const unsigned int myIdx,
		const unsigned int a0,
		const unsigned int a1,
		const unsigned int b0,
		const unsigned int b1)
{
	unsigned my_ans[2][2]={0};

	int b;
	for(unsigned k = 0 ; k < WARP_SIZE ; ++k){
		b= (myIdx>= k);
		my_ans[0][0] ^= (b*__shfl_up(a0,k)) & __shfl(b0,k);
		my_ans[0][1] ^= ((1-b)*__shfl_down(a0,WARP_SIZE-k))& __shfl(b0,k);
		my_ans[0][1] ^= (b*__shfl_up(a1,k)) & __shfl(b0,k);
		my_ans[1][0] ^= ((1-b)*__shfl_down(a1,32-k))& __shfl(b0,k);

		my_ans[0][1] ^= (b*__shfl_up(a0,k)) & __shfl(b1,k);
		my_ans[1][0] ^= ((1-b)*__shfl_down(a0,32-k))& __shfl(b1,k);
		my_ans[1][0] ^= (b*__shfl_up(a1,k)) & __shfl(b1,k);
		my_ans[1][1] ^= ((1-b)*__shfl_down(a1,32-k))& __shfl(b1,k);
	}

	cChunk[myIdx] ^= my_ans[0][0];
	cChunk[myIdx + warpSize] ^= my_ans[0][1];
	__syncthreads();
	cChunk[myIdx + 2*warpSize] ^= my_ans[1][0];
	cChunk[myIdx + 3*warpSize] ^= my_ans[1][1];
	__syncthreads();
}

__device__ inline void multiplyByPentanomial(unsigned int* cChunk, unsigned int* resChunk, const unsigned int myIdx)
{
	unsigned int myC;
	myC = cChunk[myIdx];
#pragma unroll
	for (unsigned int i = 0  ; i < 4 ; ++i)
	{
		resChunk[pentanomial[i] + myIdx] ^= myC;
		__syncthreads();
	}

}

__device__ inline void multiply64Shmem(
		unsigned int* aChunk,
		unsigned int* bChunk,
		unsigned int* cChunk,
		const unsigned int myIdx)
{
	unsigned int b;
	unsigned int my_ans[2][2] = {0};
	for(unsigned k = 0 ; k < WARP_SIZE ; ++k){
		b= (myIdx>= k);
		if (b)
		{
			my_ans[0][0] ^= (b*aChunk[myIdx - k]) & bChunk[k];
		}
		my_ans[0][1] ^= (aChunk[myIdx + 32-k])& bChunk[k];
		if (1-b)
		{
			my_ans[1][0] ^= ((1-b)* aChunk[myIdx + 64 - k])& bChunk[k];
		}
		else
		{
			my_ans[0][1] ^= (b*aChunk[myIdx -k]) & bChunk[k + WARP_SIZE];
		}
		my_ans[1][0] ^= (aChunk[myIdx + 32 - k])& bChunk[k + WARP_SIZE];
		if (1-b)
		{
			my_ans[1][1] ^= ((1-b)*aChunk[myIdx + 64 - k])& bChunk[k + WARP_SIZE];
		}
	}
	cChunk[myIdx] ^= my_ans[0][0];
	cChunk[myIdx + warpSize] ^= my_ans[0][1];
	__syncthreads();
	cChunk[myIdx + 2*warpSize] ^= my_ans[1][0];
	cChunk[myIdx + 3*warpSize] ^= my_ans[1][1];
	__syncthreads();
}
//----------------------------------------------------------------------
// Polynomials multiplication implementation for GF(2^64)
__device__ void polynomialMultiplication32Shmem(
		unsigned int a[32],
		unsigned int b[32],
		unsigned int c[64],
		unsigned int myIdxInWarp)
{
	for (unsigned int i = 0 ; i < 32 ; ++i)
	{
		if (i <= myIdxInWarp)
		{
			c[myIdxInWarp] ^= a[i]&b[myIdxInWarp-i];
		}
		else
		{
			c[myIdxInWarp + 32] ^= a[i] & b[myIdxInWarp + 32 - i];
		}
	}
}
__device__ void polynomialMultiplication32Shuffle(
		unsigned int a[32],
		unsigned int b[32],
		unsigned int c[64],
		unsigned int myIdxInWarp)
{
	unsigned int aReg = a[myIdxInWarp];
	unsigned int bReg = b[myIdxInWarp];
	unsigned int cReg[2] = {0};
	for (unsigned int i = 0  ; i < 32 ; ++i)
	{
		unsigned int readB = __shfl(bReg, (myIdxInWarp - i) % WARP_SIZE);
		unsigned int readA = __shfl(aReg, i);
		if (i <= myIdxInWarp)
		{
			cReg[0] ^= readB&readA;
		}
		else
		{
			cReg[1] ^= readB&readA;
		}
	}
	c[myIdxInWarp] = cReg[0];
	c[myIdxInWarp + 32] = cReg[1];
}




__device__ void polynomialMultiplication64Based32Shmem(
		unsigned int a[64],
		unsigned int b[64],
		unsigned int c[128],
		unsigned int myIdxInGroup)
{
	unsigned int myIdxInWarp = myIdxInGroup & (WARP_SIZE - 1);
	unsigned int cRegs[2] = {0};
#pragma unroll
	for (unsigned int j = 0 ; j < 2 ; ++j)
	{
#pragma unroll
		for (unsigned int i = 0 ; i < 32 ; ++i)
		{
			if (i <= myIdxInWarp)
			{
				cRegs[0] ^= a[32*j + i]&b[myIdxInGroup-i];
			}
			else
			{
				cRegs[1] ^= a[32*j + i] & b[myIdxInGroup + 32 - i];
			}
		}
		c[myIdxInGroup + j * 32] ^= cRegs[0];
		__syncthreads();
		c[myIdxInGroup + j * 32 + 32] ^= cRegs[1];
		__syncthreads();
	}
}




__device__ void polynomialMultiplication64Based32Shuffle(
		unsigned int a[64],
		unsigned int b[64],
		unsigned int c[128],
		unsigned int myIdxInGroup)
{
	unsigned int myIdxInWarp = myIdxInGroup & (WARP_SIZE - 1);
	unsigned int aReg = a[myIdxInGroup];
//#pragma unroll
	for (unsigned int j = 0 ; j < 2 ; ++j)
	{
		unsigned int cReg[2] = {0};
		unsigned int bReg = b[myIdxInWarp + 32 * j];
//#pragma unroll
		for (unsigned int i = 0 ; i < 32 ; ++i)
		{
			unsigned int readB = __shfl(bReg, (myIdxInWarp - i) % WARP_SIZE);
			unsigned int readA = __shfl(aReg, i);
			if (i <= myIdxInWarp)
			{
				cReg[0] ^= readB&readA;
			}
			else
			{
				cReg[1] ^= readB&readA;
			}
		}
		c[myIdxInGroup + j * 32] ^= cReg[0];
		__syncthreads();
		c[myIdxInGroup + j * 32 + 32] ^= cReg[1];
		__syncthreads();
	}

}


__device__ void polynomialMultiplication64Shuffle(
		unsigned int aInput[64],
		unsigned int bInput[64],
		unsigned int cOutput[128],
		unsigned int myIdx)
{
	unsigned int a1 = aInput[myIdx + 32];
	unsigned int b1 = bInput[myIdx + 32];
	unsigned int a0 = aInput[myIdx];
	unsigned int b0 = bInput[myIdx];
	unsigned int my_ans[2][2] = {0};
#pragma unroll
	for(unsigned k = 0 ; k < WARP_SIZE ; ++k){
		unsigned int bLo = __shfl(b0,k);
		unsigned int bHi = __shfl(b1,k);
		unsigned int aLok = __shfl_up(a0,k);
		unsigned int aHik = __shfl_up(a1,k);
		unsigned int aLoInv = __shfl_down(a0,WARP_SIZE-k);
		unsigned int aHiInv = __shfl_down(a1,32-k);
		if (myIdx >= k)
		{
			my_ans[0][0] ^= aLok & bLo;
			my_ans[0][1] ^= aHik & bLo;
			my_ans[0][1] ^= aLok & bHi;
			my_ans[1][0] ^= aHik & bHi;
		}
		else
		{
			my_ans[0][1] ^= aLoInv& bLo;
			my_ans[1][0] ^= aHiInv& bLo;
			my_ans[1][0] ^= aLoInv& bHi;
			my_ans[1][1] ^= aHiInv& bHi;
		}
	}

	cOutput[myIdx] ^= my_ans[0][0];
	cOutput[myIdx + warpSize] ^= my_ans[0][1];
	cOutput[myIdx + 2*warpSize] ^= my_ans[1][0];
	cOutput[myIdx + 3*warpSize] ^= my_ans[1][1];
}
__global__ void performMult64(
		unsigned int a[][64],
		unsigned int b[][64],
		unsigned int c[][128],
		unsigned int size)
{
#if TYPE == 2
		unsigned int inWarpIdx = (threadIdx.x + blockDim.x * blockIdx. x) & (WARP_SIZE - 1);
		unsigned int warpIdx = (threadIdx.x + blockDim.x * blockIdx.x) / WARP_SIZE;
		polynomialMultiplication64Shuffle(a[warpIdx],b[warpIdx],c[warpIdx], inWarpIdx);
		return;
#else
		__shared__ unsigned int aShmem[16][64];
		__shared__ unsigned int bShmem[16][64];
		__shared__ unsigned int cShmem[16][128];
		unsigned int inGroupIdx = (threadIdx.x + blockDim.x * blockIdx. x) & (63);
		unsigned int warpIdx = (threadIdx.x + blockDim.x * blockIdx.x) / 64;

		if(warpIdx >= size)
			return;

		unsigned int inTBwarpIdx = warpIdx & (15);
		aShmem[inTBwarpIdx][inGroupIdx] = a[warpIdx][inGroupIdx];
		bShmem[inTBwarpIdx][inGroupIdx] = b[warpIdx][inGroupIdx];
		cShmem[inTBwarpIdx][inGroupIdx] = 0;
		cShmem[inTBwarpIdx][inGroupIdx + 64] = 0;
#if TYPE == 1
		polynomialMultiplication64Based32Shuffle(aShmem[inTBwarpIdx], bShmem[inTBwarpIdx], cShmem[inTBwarpIdx], inGroupIdx);
#else
		polynomialMultiplication64Based32Shmem(aShmem[inTBwarpIdx], bShmem[inTBwarpIdx], cShmem[inTBwarpIdx], inGroupIdx);
#endif
		c[warpIdx][inGroupIdx] = cShmem[inTBwarpIdx][inGroupIdx];
		c[warpIdx][inGroupIdx + 64] = cShmem[inTBwarpIdx][inGroupIdx + 64];
#endif
}

#endif /* CHUNKPOC_CUH_ */


