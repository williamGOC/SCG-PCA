#ifndef INTERFAZ_OPEN_GL_TO_CUDA_H
#define INTERFAZ_OPEN_GL_TO_CUDA_H

#include "./common/book.h"
#include "./common/cpu_bitmap.h"

#include <cuda.h>
#include <cuda_gl_interop.h>

#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<cmath>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include<thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include<thrust/device_vector.h>
#include <thrust/functional.h>
#include<thrust/host_vector.h>
#include<thrust/for_each.h>
#include <thrust/tuple.h>
#include<thrust/reduce.h>
#include <thrust/fill.h>
#include<thrust/copy.h>


typedef unsigned short ubyte;

#define ALIVE 1
#define DEAD 0

#define DIM    1024
#define MESH_DIM 1024
//#define N DIM * DIM

class game{

private:
	thrust::host_vector<ubyte> hostWorld;
	thrust::device_vector<ubyte> devWorld;

	thrust::host_vector<ubyte> hostFancyWorld;
	thrust::device_vector<ubyte> devFancyWorld;

	ubyte *World_ptr;
	ubyte *Fancy_ptr;

	int L, N;
	float RHO;

public:
	game(int _L, float _RHO);
	void addsquare(int x, int y, int z, int w, int sign);

	ubyte *dptr_W;
	ubyte *dptr_F;
	//~game();
	
};

game::game(int _L, float _RHO):L(_L),RHO(_RHO){

	N = L * L;

	hostWorld.resize(N);
	hostFancyWorld.resize(N);

	thrust::fill(hostWorld.begin(), hostWorld.end(), DEAD);
	thrust::fill(hostFancyWorld.begin(), hostFancyWorld.end(), DEAD);

	unsigned Nalive = static_cast<unsigned>(RHO * N);
	unsigned n = 0;

	while(n < Nalive) {
		unsigned id = static_cast<unsigned>(floor(N * drand48()));
		if(hostWorld[id] == DEAD){
			hostWorld[id] = ALIVE;
			n++;
		}
	}

	devWorld.resize(N);
	devFancyWorld.resize(N);

	thrust::copy(hostWorld.begin(), hostWorld.end(), devWorld.begin());
	thrust::copy(hostFancyWorld.begin(), hostFancyWorld.end(), devFancyWorld.begin());

	World_ptr = thrust::raw_pointer_cast(&devWorld[0]);
	dptr_W = World_ptr;

	Fancy_ptr = thrust::raw_pointer_cast(&devFancyWorld[0]);
	dptr_F = Fancy_ptr;
}


void game::addsquare(int x, int y, int z, int w, int sign){
	int ww = (w > DIM)?(DIM):(w);
	for(int j=y-ww/2;j<y+ww/2;j++){
		for(int i=x-ww/2;i<x+ww/2;i++){
			devWorld[(i+DIM)%DIM+((j+DIM)%DIM)*DIM]=sign;
	 	}   
	}
}
#endif // INTERFAZ_OPEN_GL_TO_CUDA_H
