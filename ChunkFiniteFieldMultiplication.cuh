///*
// * ChunkFiniteFieldMultiplication.h
// *
// *  Created on: Jan 6, 2016
// *      Author: matan
// */
//
//#ifndef CHUNKFINITEFIELDMULTIPLICATION_H_
//#define CHUNKFINITEFIELDMULTIPLICATION_H_
//
//#include "Chunk.cuh"
//
//template <unsigned int N>
//__device__ void chunkFiniteFieldMultiply(Chunk<N>& a, Chunk<N>& b, Chunk<N>& c, Chunk<2*N>& temp);
//
//template <>
//void chunkFiniteFieldMultiply<64>(Chunk<64>& a, Chunk<64>& b, Chunk<64>& c, Chunk<128>& temp)
//{
//	chunkMultiply(a,b,temp);
//	chunkMultiplyPentanomial(
//			reinterpret_cast<Chunk<32>& >(temp.coefficients[96]),
//			reinterpret_cast<Chunk<64>& >(temp.coefficients[32]));
//	chunkMultiplyPentanomial(
//				reinterpret_cast<Chunk<32>& >(temp.coefficients[64]),
//				reinterpret_cast<Chunk<64>& >(temp));
//
//	c = reinterpret_cast<Chunk<64>& >(temp);
//}
//
//template <unsigned int N>
//__device__ void chunkFiniteFieldMultiply(Chunk<N>& a, Chunk<N>& b, Chunk<N>& c, Chunk<2*N>& temp)
//{
//	chunkMultiply(a,b,temp);
//
//
//}
//
//#endif /* CHUNKFINITEFIELDMULTIPLICATION_H_ */
