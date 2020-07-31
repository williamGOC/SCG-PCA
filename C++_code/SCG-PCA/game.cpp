#include "game.h"

short kronneker(int a, int b){
	return (a == b)?(1):(0);
}

game::game(int _L, float _RHO):L(_L),RHO(_RHO){

	N = L * L;

	world.resize(N);

	// Create dead world:
	std::uninitialized_fill(world.begin(), world.end(), DEAD);
	
}

unsigned game::getId(int x, int y){

	// periodic bonduary conditions:
	while(x >= L)
		x -= L;

	while(x < 0)
		x += L;

	while(y >= L)
		y -= L;

	while(y < 0)
		y += L;

	return x + y * L;
}

void game::randomWorld(){

	int Nalive = static_cast<int>(N * RHO);
	int n = 0;

	while(n < Nalive){

		unsigned index = static_cast<unsigned>(floor(N * drand48()));
		if(world[index] == DEAD){
			world[index] = ALIVE;
			n++;
		}
	}
}


float game::getRHO(){
	float density = std::accumulate(world.begin(), world.end(), 0.0);
	return density / static_cast<float>(N);
}


int game::getL(){
	return L;
}


float game::getAlphas(){
	return T;
}


unsigned short game::cell(unsigned index){
	return world[index];
}


void game::step(){

	std::vector<unsigned short> fancyWorld(N); 
    double w1;

	for (int k = 0; k < N; k++){
		
		int x = k % L;
		int y = k / L;

		// determine new state by rules of Conway's Game of Life:
		unsigned short state     = world[k];
		unsigned short newstate  = DEAD;

		// calculate number of alive neihbors:
		unsigned short aliveNeighbors = 0;

		for(short x_offset = -1; x_offset <= 1; x_offset++){
			for(short y_offset = -1; y_offset <= 1; y_offset++){
				if(x_offset != 0 || y_offset != 0){		// don't count itself

					unsigned neihborId = getId(x + x_offset, y + y_offset);
					aliveNeighbors += world[neihborId];
				}
			}
		}

		w1 = ALPHA_1 * state*kronneker(2,aliveNeighbors) + ALPHA_2 * (1-state)*kronneker(3,aliveNeighbors) + ALPHA_3 * state*kronneker(3, aliveNeighbors);

		if(drand48() < w1){
			newstate = ALIVE;
		}

		fancyWorld[k] = newstate;
	}

	std::copy(fancyWorld.begin(), fancyWorld.end(), world.begin());
}
