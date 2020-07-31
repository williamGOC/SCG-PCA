#ifndef GAME_H
#define GAME_H

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <ctime>

#define ALIVE 1
#define DEAD  0

class game {

private:
	
	float RHO;
	float T;
	int L, N;

	std::vector<unsigned short> world;

public:
	game(int _L, float _RHO, float _T);

	unsigned getId(int x, int y);
	void randomWorld();
	float getRHO();
	float getT();
	int getL();

	unsigned short cell(unsigned index);
	void step();
	
	//~game();
	
};


#endif // GAME_H