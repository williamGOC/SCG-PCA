#include "game.h"

game::game(int _L, float _RHO):L(_L),RHO(_RHO){

	N = L * L;

	world.resize(N);

	// Create dead world:
	for (int i = 0; i < N; ++i)
		world[i] = DEAD; 
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

double game::getRHO(){
	return RHO;
}

int game::getL(){
	return L;
}

unsigned short game::cell(unsigned index){
	return world[index];
}



void game::step(){

	std::vector<unsigned short> fancyWorld(N);

	for (int k = 0; k < N; k++){
		
		int x = k % L;
		int y = k / L;

		// determine new state by rules of Conway's Game of Life:
		unsigned short state    = world[k];
		unsigned short newstate = state;

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

		// decide about new state:
		if(state == ALIVE){
			if(aliveNeighbors < 2 || aliveNeighbors > 3)
				newstate = DEAD;
			else
				newstate = ALIVE;
		}
		else {		// if DEAD
			if(aliveNeighbors == 3)
				newstate = ALIVE;
		}

		fancyWorld[k] = newstate;
	}

	for(int k = 0; k < N; k++)
		world[k] = fancyWorld[k];

}
