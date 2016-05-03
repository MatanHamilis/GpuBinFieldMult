#ifndef SIZE
#define SIZE 32
#endif

#ifndef REPETITIONS
#define REPETITIONS 5
#endif

#include <iostream>
#include "ChunkTesting.cuh"

float getSum(float* array, unsigned int size)
{
	float sum = 0;
	for (unsigned int i = 0 ; i < size ; ++i )
	{
		sum += array[i];
	}

	return sum;
}

float getMax(float* array, unsigned int size)
{
	float max = 0;
	for (unsigned int i = 0 ; i < size ; ++i)
	{
		if (array[i] > max)
		{
			max = array[i];
		}
	}
	return max;
}

float getMin(float* array, unsigned int size)
{
	float min = array[0];
	for (unsigned int i = 0 ; i < size ; ++i)
	{
		if (array[i] < min)
		{
			min = array[i];
		}
	}
	return min;
}

float performTest(unsigned int mults)
{
	float results[REPETITIONS] = {0};
	for (unsigned int i = 0 ; i < REPETITIONS ; ++i)
	{
		results[i] = testChunksMultiply<SIZE>(mults);
	}
	if (REPETITIONS >= 3)
	{
		float choppedSum = getSum(results, REPETITIONS) - getMax(results, REPETITIONS) - getMin(results, REPETITIONS);
		return choppedSum/(REPETITIONS - 2);
	}
	return 0;
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		std::cerr << "Usage: " << argv[0] << " <log of minimal multiplications> <log of maximal multiplications>" << std::endl;
	}

	int minimalMults = atoi(argv[1]);
	int maximalMults = atoi(argv[2]);

	cudaSetDevice(0);
	for (unsigned int i = minimalMults ; i <= maximalMults ; ++i)
	{
		std::cout << performTest((1<<i)) << std::endl;
	}
}
