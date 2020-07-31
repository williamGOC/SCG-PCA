#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define ALIVE 1
#define DEAD  0

// world size in x and y direction:
#define L 500

#define N L*L		// number of sites


# define IM1 2147483563
# define IM2 2147483399
# define AM (1.0/IM1)
# define IMM1 (IM1-1)
# define IA1 40014
# define IA2 40692
# define IQ1 53668
# define IQ2 52774
# define IR1 12211
# define IR2 3791
# define NTAB 32
# define NDIV (1+IMM1/NTAB)
# define EPS 1.2e-7
# define RNMX (1.0-EPS)



typedef struct system{
	unsigned short *world;
	double RHO;
	long seed;
} *game;

// System Structuring Functions:
game builderSystem(double RHO);
void printerSystem(game pG, FILE * gnuplotPIPE);
void freeSystem(game pG);


float rand2(long *idum);
void step(game pG);
unsigned getId(int x, int y);
//Evolution System of Conway's Game of Life:

// Random world:
void randomWorld(game pG);

// Still lifef:
void Block(game pG);
void BeeHive(game pG);
//void Loaf(game pG);
//void Boad(game pG);
//void Tub(game pG);

// Oscillators
void Blinker(game pG);

// Spaceships
void Glider(game pG);

/*int main(int argc, char const *argv[]) {

	FILE * pipe = popen("gnuplot", "w");
	assert(pipe);
	
	double RHO = 0.2;
	unsigned iterations = 1000000;

	game pG = builderSystem(RHO);
	//randomWorld(pG);
	Blinker(pG);
	Glider(pG);
	

    int k;
    for(k = 0; k < iterations; k++){
    	step(pG);
    	printerSystem(pG, pipe);
    }

	
	freeSystem(pG);
	pclose(pipe);

	return 0;
}*/

game builderSystem(double RHO){

	game pG = (game) malloc(sizeof(struct system));		// Memory allocation for the central structure of the program.
	assert(pG);

	pG -> world = (unsigned short *) malloc(N * sizeof(unsigned short));	// Creation of dynamic array member of the central structure.
	assert(pG -> world);

	// Create dead world.
	int i;
	for(i = 0; i < N; i++)
		pG -> world[i] = DEAD;

	pG -> RHO = RHO;
	pG -> seed = -time(NULL);
}


void printerSystem(game pG, FILE * gnuplotPIPE){

	int i, j;

	fprintf(gnuplotPIPE, "set title '{/=20 Conway's Game of Life}'\n");
	fprintf(gnuplotPIPE, "unset xtics; unset ytics\n");
	fprintf(gnuplotPIPE, "unset border\n");
	fprintf(gnuplotPIPE, "set xrange [0:%d]; set yrange [0:%d]\n", L, L);
	fprintf(gnuplotPIPE, "plot '-' u 1:2:3 w p pt 5 ps 3 lc variable title ''\n");

	for(i = 0; i < N; i++)
		fprintf(gnuplotPIPE, "%d\t%d\t%d\n", i%L, i/L, pG -> world[i]);

	fprintf(gnuplotPIPE, "e\n");
	fflush(gnuplotPIPE);
}


void freeSystem(game pG){

	free(pG -> world);
	free(pG);
}


void randomWorld(game pG){

	int Nalive = (int)(N * pG -> RHO);
	int n = 0;

	while(n < Nalive){
		
		unsigned index = (unsigned)(N * rand2(&pG -> seed));
		if(pG -> world[index] == DEAD){
			pG -> world[index] = ALIVE;
			n++;
		}
	}
}




void step(game pG){

	int k;

	unsigned short fancyWorld[N]; 

	for (int k = 0; k < N; k++){
		
		int x = k % L;
		int y = k / L;

		// determine new state by rules of Conway's Game of Life:
		unsigned short state    = pG -> world[k];
		unsigned short newstate = state;

		// calculate number of alive neihbors:
		unsigned short aliveNeighbors = 0;

		for(short x_offset = -1; x_offset <= 1; x_offset++){
			for(short y_offset = -1; y_offset <= 1; y_offset++){
				if(x_offset != 0 || y_offset != 0){		// don't count itself

					unsigned neihborId = getId(x + x_offset, y + y_offset);
					aliveNeighbors += pG -> world[neihborId];
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

	for(k = 0; k < N; k++)
		pG -> world[k] = fancyWorld[k];

}




// Get world array index from world coordinates
unsigned getId(int x, int y){

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


void Block(game pG){

	int index = (unsigned)(N * rand2(&pG -> seed));
	
	int x = index % L;
	int y = index / L;
	
	// 00000000000
	// 00000xx0000
	// 00000xx0000
	// 00000000000
	
	pG -> world[getId(x, y)] = ALIVE;
	pG -> world[getId(x + 1, y)] = ALIVE;
	pG -> world[getId(x, y + 1)] = ALIVE;
	pG -> world[getId(x + 1, y + 1)] = ALIVE;
}


void BeeHive(game pG){

	int index = (unsigned)(N * rand2(&pG -> seed));
	
	int x = index % L;
	int y = index / L;

	// 00000000000
	// 0000xx00000
	// 000x00x0000
	// 0000xx00000
	// 00000000000

	pG -> world[getId(x, y)] = ALIVE;
	pG -> world[getId(x + 1, y)] = ALIVE;
	pG -> world[getId(x + 2, y - 1)] = ALIVE;
	pG -> world[getId(x - 1, y - 1)] = ALIVE;
	pG -> world[getId(x, y - 2)] = ALIVE;
	pG -> world[getId(x + 1, y - 2)] = ALIVE;
}

void Blinker(game pG){

	int index = (unsigned)(N * rand2(&pG -> seed));
	
	int x = index % L;
	int y = index / L;

	pG -> world[getId(x + 1, y)] = ALIVE;
	pG -> world[getId(x, y)] = ALIVE;
	pG -> world[getId(x - 1, y)] = ALIVE;
}


void Glider(game pG){

	int index = (unsigned)(N * rand2(&pG -> seed));
	
	int x = index % L;
	int y = index / L;

	pG -> world[getId(x, y)] = ALIVE;
	pG -> world[getId(x, y - 1)] = ALIVE;
	pG -> world[getId(x, y - 2)] = ALIVE;
	pG -> world[getId(x - 1, y - 2)] = ALIVE;
	pG -> world[getId(x - 2, y - 1)] = ALIVE;

}


float rand2(long *idum){
    
  int j;
  long k;
  
  static long idum2=123456789;
  static long iy=0;
  static long iv[NTAB];
  float temp;

  if (*idum <= 0){                        //Initialize.
  if (-(*idum) < 1) *idum=1;              //Be sure to prevent idum = 0.
  else *idum = -(*idum);
  idum2=(*idum);
  for (j=NTAB+7;j>=0;j--){                //Load the shuffle table (after 8 warm-ups).
  k=(*idum)/IQ1;
  *idum=IA1*(*idum-k*IQ1)-k*IR1;
  if (*idum < 0) *idum += IM1;
  if (j < NTAB) iv[j] = *idum;
  }
  iy=iv[0];
  }
  k=(*idum)/IQ1;                           //Start here when not initializing.
  *idum=IA1*(*idum-k*IQ1)-k*IR1;           //Compute idum=(IA1*idum) % IM1 without
  if (*idum < 0) *idum += IM1;             //overflows by Schrage’s method.
  k=idum2/IQ2;
  idum2=IA2*(idum2-k*IQ2)-k*IR2;           //Compute idum2=(IA2*idum) % IM2 likewise.
  if (idum2 < 0) idum2 += IM2;
  j=iy/NDIV;                               //Will be in the range 0..NTAB-1.
  iy=iv[j]-idum2;                          //Here idum is shuffled, idum and idum2 are
  iv[j] = *idum;                           //combined to generate output.
  if (iy < 1) iy += IMM1;
  if ((temp=AM*iy) > RNMX) return RNMX;    //Because users don’t expect endpoint values.
  else return temp;
}