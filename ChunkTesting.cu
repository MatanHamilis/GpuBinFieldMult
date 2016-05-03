#include "utils.h"
//#include "ChunkTesting.cuh"

// This is a simple list of irreducibles, one can add his own irreducibles here to be used.
unsigned int irreducibles[][5] =
{
		{0}, //2^0 irreducible
		{0, 0, 0, 1, 2}, // <-- Represents a polynomial of the form 1 + x^1 + x^2.
		{0, 0, 0, 1, 4}, // <-- Represents a polynomial of the form 1 + x^1 + x^4.
		{0, 1, 3, 4, 8}, // <-- Represents a polynomial of the form 1 + x^1 + x^3 + x^4 + x^8.
		{0, 1, 3, 5, 16},
		{0, 2, 3, 7, 32},
		{0, 1, 3, 4, 64},
		{0, 1, 4, 9, 80},
		{0, 0, 0, 11, 95},
		{0, 6, 9, 10, 96},
		{0, 3, 4, 5, 112},
		{0, 0, 0, 1, 127},
		{0, 1, 2, 7, 128},
		{0, 2, 4, 7, 144},
		{0, 2, 3, 5, 160},
		{0, 2, 3, 11, 176},
		{0, 1, 2, 7, 192},
		{0, 1, 3, 9, 208},
		{0, 3, 8, 9, 224},
		{0, 3, 5, 8, 240},
		{0, 2, 5, 10, 256},
		{0, 2, 3, 9, 272},
		{0, 1, 10 ,11 ,288},
		{0, 1, 2, 11, 304},
		{0, 1, 3, 4, 320},
		{0, 1, 4, 7, 336},
		{0, 6, 11, 13, 352},
		{0, 2, 3, 7,368},
		{0, 2, 3, 12, 384},
		{0, 2, 5, 8, 512},
		{0, 1, 6, 19, 1024},
		{0, 13, 14, 19, 2048}
};
