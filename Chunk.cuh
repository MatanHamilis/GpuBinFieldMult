/*
 * Chunk.h
 *
 *  Created on: Jan 3, 2016
 *      Author: matan
 */

#ifndef CHUNK_H_
#define CHUNK_H_

#include <cuda_runtime.h>
#include <iostream>

#include "utils.h"

#define BASIC_MULT_SIZE 32
#define logWarpSize 5

#ifdef __CUDACC__
extern __constant__ unsigned int pentanomialCoefficients[];
#endif

/*
 * A class for GF(2^N) chunks.
 * A chunk represents a set of 32 elements in the field of GF(2^k) for some k.
 * This implementation currently support 32 <= k < 2048.
 */
template <unsigned int N>
class Chunk {
public:
	/// A chunks with all elements set to zero.
	HOST_FUNCTION Chunk();

	/*
	 * @brief Copy c'tor.
	 * @param chunk - The chunk to be copied.
	 */
	HOST_FUNCTION Chunk(const Chunk& chunk);

	static const unsigned int degree = N;

	/// D'tor.
	HOST_FUNCTION ~Chunk(){};

	// Assignment of chunks.
	HOST_DEVICE_FUNCTION void operator=(const Chunk<N>& c);

	// Pretty prints a chunk, useful for debugging.
	HOST_FUNCTION void print();

	/*
	 * @brief Returns the coefficients of x^idx of all polynomials in the chunk.
	 * @param Idx - The power of x such that the function returns the coefficients of x^idx.
	 * @return The coefficients of x^idx of all polynomials in the chunk.
	 */
	HOST_DEVICE_FUNCTION inline unsigned int getCoefficient(unsigned int idx) const;

	/*
	 * @brief Returns the pointer to the coefficients of x^idx of all polynomials in the chunk.
	 * @param Idx - The power of x such that the function returns the pointer to the coefficients of x^idx.
	 * @return The pointer to the coefficients of x^idx of all polynomials in the chunk.
	 */
	HOST_DEVICE_FUNCTION unsigned int& getCoefficientPtr(unsigned int idx) const;

	/*
	 * @brief Sets the coefficients of x^idx of all polynomials in the chunk.
	 * @param Idx - The power of x such that the function sets the coefficients of x^idx.
	 * @param value - The value of the coefficients of x^idx.
	 */
	HOST_DEVICE_FUNCTION inline void setCoefficient(unsigned int idx, unsigned int value);

	/*
	 * @brief Xors the coefficients of x^idx of all polynomials in the chunk.
	 * @param Idx - The power of x such that the function sets the coefficients of x^idx.
	 * @param value - The value of the coefficients of x^idx.
	 */
	HOST_DEVICE_FUNCTION inline void xorCoefficient(unsigned int idx, unsigned int value);
#ifdef __CUDACC__
	HOST_FUNCTION static void setPentanomial(unsigned int coefficients[5]);
#endif
	static void* operator new(std::size_t size);
	static void* operator new[](std::size_t size);
	static void operator delete(void * ptr, std::size_t size);
	static void operator delete[] (void* ptr, std::size_t size);

	/// Array of coefficients of size N, represents elements in GF(2^N).
	unsigned int coefficients[N];

};

template <unsigned int N>
void Chunk<N>::print()
{
	for (unsigned int i = 0 ; i < N ; ++i)
	{
		std::cout << this->coefficients[i] << std::endl;
	}
}

#ifdef __CUDACC__
template<unsigned int N>
void Chunk<N>::setPentanomial(unsigned int pentanomial[5])
{
	cudaMemcpyToSymbol(pentanomialCoefficients, pentanomial, sizeof(unsigned int)*5);
}
#endif

template <unsigned int N>
void Chunk<N>::operator=(const Chunk<N>& c)
{
#ifdef __CUDACC__ //<- GPU Version
	unsigned int lindex = threadIdx.x & (warpSize-1);

	for (unsigned int i = lindex ; i < N ; i += warpSize)
	{
		this->coefficients[i] = c.coefficients[i];
	}

#else //<- CPU Version
	for (unsigned int i = 0 ; i < N ; ++i)
	{
		this->coefficients[i] = c.coefficients[i];
	}
#endif
}

template <unsigned int N>
unsigned int& Chunk<N>::getCoefficientPtr(unsigned int idx) const
{
	return this->coefficients[idx];
}

template <unsigned int N>
void Chunk<N>::setCoefficient(unsigned int idx, unsigned int value)
{
	this->coefficients[idx] = value;
}

template <unsigned int N>
void Chunk<N>::xorCoefficient(unsigned int idx, unsigned int value)
{
	this->coefficients[idx] ^= value;
}

template <unsigned int N>
unsigned int Chunk<N>::getCoefficient(unsigned int idx) const
{
	return this->coefficients[idx];
}

template <unsigned int N>
Chunk<N>::Chunk()
{
	for (unsigned int i = 0 ; i < N ; ++i)
	{
		setCoefficient(i,0);
	}

	if(N > 32)
		setCoefficient(32,2);
}

template <unsigned int N>
void * Chunk<N>::operator new(std::size_t size)
{
	void *ptr = NULL;
	cudaMallocManaged(&ptr, size);
	cudaDeviceSynchronize();
	return ptr;
}

template <unsigned int N>
void * Chunk<N>::operator new[](std::size_t size)
{
	void *ptr = NULL;
	cudaMallocManaged(&ptr, size);
	cudaDeviceSynchronize();
	return ptr;
}

template <unsigned int N>
void Chunk<N>::operator delete(void *ptr, std::size_t size)
{
	cudaDeviceSynchronize();
	cudaFree(ptr);
}

template <unsigned int N>
void Chunk<N>::operator delete[](void *ptr, std::size_t size)
{
	cudaDeviceSynchronize();
	cudaFree(ptr);
}

template <unsigned int N>
Chunk<N>::Chunk(const Chunk& chunk)
{
	for (unsigned int i = 0 ; i < N ; ++i)
	{
		setCoefficient(i,chunk.getCoefficient(i));
	}
}



#ifdef __CUDACC__

/**
 * @brief XORs chunks a & b, stores the result in both of them.
 */
__device__ inline void chunkMutualXorBasic(
		Chunk<BASIC_MULT_SIZE>& a,
		Chunk<BASIC_MULT_SIZE>& b
);


/*
 *	@brief Multiplies a chunk by b, output is written into chunk c.
 *	@param a, b - The chunks being multiplied.
 *	@param c - The chunk into which the multiplication is written to.
 */
template <unsigned int N>
__device__ inline void chunkMultiply (Chunk<N>& a, Chunk<N>& b, Chunk<2*N>& c);

/*
 *	@brief Multiplies a chunk by a pentanomial stored in constant memory.
 *	@param a - The chunks being multiplied.
 *	@param c - The chunk into which the multiplication is written to.
 */
template <unsigned int N>
__device__ void chunkMultiplyPentanomial(Chunk<N>&a, Chunk<2*N>& c)
{
	if (N <= 64)
	{
		unsigned int lindex = threadIdx.x & (warpSize -1);
		unsigned int inputRegs[regsPerThread(N)];

#pragma unroll
		for (int i = 0 ; i < regsPerThread(N) ; ++i)
		{
			inputRegs[i] = a.coefficients[lindex + i*warpSize];
		}
#pragma unroll
		for (int i = 0 ; i < 4 ; ++i)
		{
#pragma unroll
			for (int j = 0 ; j < regsPerThread(N) ; ++j)
			{
				c.coefficients[pentanomialCoefficients[i] + lindex + j*warpSize] ^= inputRegs[j];
			}
		}
	}
	else
	{
// To be implemented....
	}
}

/*
 * @brief Adds rhs chunk to this chunk, output is written into lhs.
 * @param rhs - The chunks to be added to this chunk.
 * @param lhs - Left hand side of the addition.
 */
template <unsigned int N>
__device__ void chunkAdd (Chunk<N>& lhs, const Chunk<N>& rhs);

template <unsigned int N>
__device__ void chunkAdd (Chunk<N>& lhs, const Chunk<N>& rhs)
{
	unsigned int lindex = threadIdx.x & (warpSize-1);
	for (unsigned int i = lindex ; i < N ; i+=32)
	{
		lhs.xorCoefficient(i, rhs.getCoefficient(i));
	}
}

template <>
__device__ inline void chunkMultiply<32> (Chunk<32>& a, Chunk<32>& b, Chunk<64>& c)
{
	unsigned int lindex = threadIdx.x & (warpSize-1);
	unsigned int aCoeff = a.getCoefficient(lindex);
	unsigned int bCoeff = b.getCoefficient(lindex);
	unsigned int cCoeff[2] = {0};

#pragma unroll
	for (unsigned int i = 0 ; i < BASIC_MULT_SIZE ; ++i)
	{
		unsigned int indexAdd = (i>lindex);
		cCoeff[0] ^= (1-indexAdd)*__shfl(bCoeff, lindex - i) * __shfl(aCoeff,i);
		cCoeff[1] ^= indexAdd * __shfl(bCoeff, lindex + warpSize - i) * __shfl(aCoeff, i);
	}
	c.xorCoefficient(lindex, cCoeff[0]);
	c.xorCoefficient(lindex + warpSize, cCoeff[1]);
}

// Specialization of degree 64 multiplication - optimized to be used as a building block for smaller multiplications.
template <>
__device__ inline void chunkMultiply<64> (Chunk<64>& a, Chunk<64>& b, Chunk<128>& c)
{
	unsigned int lindex = threadIdx.x & (warpSize-1);
	unsigned int aCoeff[2];
	unsigned int cCoeff[2][2] = {0};

	aCoeff[0] = a.getCoefficient(lindex);
	aCoeff[1] = a.getCoefficient(lindex + warpSize);

	// output of the polynomial multiplication.
	unsigned int  my_ans[2][2]={0};
	int t;
	unsigned int bCoeff;

	// This loop perform the shuffle. Efficiently and in parallel multiplying the polynomials.
	// Distributed in the registers.
	for(unsigned int k = 0 ; k < warpSize; ++k){
		bCoeff = b.coefficients[k];
		t = (lindex>= k);
		cCoeff[0][0] ^= (t*__shfl_up(aCoeff[0],k)) & bCoeff;
		cCoeff[0][1] ^= ((1-t)*__shfl_down(aCoeff[0],warpSize-k))& bCoeff;
		cCoeff[0][1] ^= (t*__shfl_up(aCoeff[1],k)) & bCoeff;
		cCoeff[1][0] ^= ((1-t)*__shfl_down(aCoeff[1],warpSize-k))& bCoeff;

		bCoeff=b.coefficients[k+warpSize];
		cCoeff[0][1] ^= (t*__shfl_up(aCoeff[0],k)) & bCoeff;
		cCoeff[1][0] ^= ((1-t)*__shfl_down(aCoeff[0],warpSize-k))& bCoeff;
		cCoeff[1][0] ^= (t*__shfl_up(aCoeff[1],k)) & bCoeff;
		cCoeff[1][1] ^= ((1-t)*__shfl_down(aCoeff[1],warpSize-k))& bCoeff;
	}
#pragma unroll
	for (unsigned int i = 0 ; i < 2 ; ++i)
	{
#pragma unroll
		for (unsigned int j = 0 ; j < 2 ; ++j)
		{
			c.xorCoefficient(lindex+i*64 + j * warpSize, cCoeff[i][j]);
		}
	}

}

template <unsigned int N>
__device__ inline void chunkMultiply (Chunk<N>& a, Chunk<N>& b, Chunk<2*N>& c)
{
	return;
}

__device__ inline void chunkMutualXorBasic(
		Chunk<BASIC_MULT_SIZE>& a,
		Chunk<BASIC_MULT_SIZE>& b)
{
	unsigned int lindex = threadIdx.x & (warpSize-1);
	unsigned int xored = a.getCoefficient(lindex) ^ b.getCoefficient(lindex);
	a.setCoefficient(lindex, xored);
	b.setCoefficient(lindex, xored);
}


#endif



#endif /* CHUNK_H_ */
