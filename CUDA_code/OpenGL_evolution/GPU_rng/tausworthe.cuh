#ifndef TAUSWORTHE_CUH
#define TAUSWORTHE_CUH

#include <iostream>
#include <cstdlib>

#define N_BLOCKS				32	// number of blocks
#define N_THREADS_PER_BLOCK		32	// number of threads per block

#define N_RN	(50 * N_BLOCKS * N_THREADS_PER_BLOCK)

#define ITERATIONS	10000	// number of runs

/*
Generator of random numbers Tausworthe in the device:
	Three-step generator with period 2^88.

Keyword Arguments:
z1: unsigned (first step storage, is a random numbert)
z2: unsigned (second step storage, is a random numbert)
z3: unsigned (third step storage, is a random numbert)
*/
__device__ unsigned Tausworthe88(unsigned &z1, unsigned &z2, unsigned &z3)
{
	unsigned b = (((z1 << 13) ^ z1) >> 19);
	z1 = (((z1 & 4294967294) << 12) ^ b);

	b = (((z2 << 2) ^ z2) >> 25);
	z2 = (((z2 & 4294967288) << 4)  ^ b);

	b = (((z3 << 3) ^ z3) >> 11);
	z3 = (((z3 & 4294967280) << 17) ^ b);

	return z1 ^ z2 ^ z3;
}


/*
Generator of random numbers LCRNG in the device:
	Linear congruential random number generators with period 2^32.

Keyword Arguments:
z: unsigned (is a random numbert)
*/
__device__ unsigned LCRNG(unsigned &z)  
{  
	const unsigned a = 1664525, c = 1013904223;
	return z = a * z + c;
}


/*
Combination of a Tausworthe generator with a LCRNG, esulting in a 
generator with a period of about 2^120.

Keyword Arguments:
z1: unsigned (is a random numbert)
z2: unsigned (is a random numbert)
z3: unsigned (is a random numbert)
z: unsigned (is a random numbert)
*/
__device__ float TauswortheLCRNG(unsigned &z1, unsigned &z2, unsigned &z3, unsigned &z)
{
	// combine both generators and normalize 0...2^32 to 0...1
	return (Tausworthe88(z1, z2, z3) ^ LCRNG(z)) * 2.3283064365e-10;
}

#endif // TAUSWORTHE_CUH