#include "game.h"

game *pG;


int main(int argc, char **argv) {
	
	int args = 1;
	// Size of the system (default 500)
	int L = (argc > args)?(atoi(argv[args])):(500);
	if(L <= 0){
		std::cout << "error: L must be positive" << std::endl;
		exit(1);
	}
	args++;

	// Density of system (default 0.5)
	float RHO = (argc > args)?(atof(argv[args])):(0.5);
	args++;

	srand48(time(NULL));

	pG = new game(L, RHO);
	pG -> randomWorld();

	for (int i = 0; i < 100000; ++i){
		std::cout << i << '\t' << pG -> getRHO() << std::endl;
		pG -> step();
		
	}


	return 0;
}

