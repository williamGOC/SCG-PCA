#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "game.h"

#define WHITE 1.0, 1.0, 1.0
#define BLACK 0.0, 0.0, 0.0

GLint windowWidth  = 600;
GLint windowHeight = 600;

GLint FPS = 24;

GLfloat left	= 0.0;
GLfloat right	= 1.0;
GLfloat bottom	= 0.0;
GLfloat top		= 1.0;

game *pG;


void display();
void reshape(int w, int h);
void update(int value); 


int main(int argc, char **argv) {
	
	int args = 1;
	// Size of the system (default 500)
	int L = (argc > args)?(atoi(argv[args])):(100);
	if(L <= 0){
		std::cout << "error: L must be positive" << std::endl;
		exit(1);
	}
	args++;

	// Initial density of system (default 0.5)
	float RHO = (argc > args)?(atof(argv[args])):(0.3);
	args++;


	srand48(time(NULL));

	glutInit(&argc, argv);
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Aplication: Conway's Game of Life.");
	glClearColor(1, 1, 1, 1);

	glutReshapeFunc(reshape);
	glutDisplayFunc(display);


	pG = new game(L, RHO);
	pG -> randomWorld();

	update(0);
	glutMainLoop();

	return 0;
}

void display(){

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	GLint L = static_cast<GLint>(pG -> getL());

	GLfloat xSize = (right - left) / L;
	GLfloat ySize = (top - bottom) / L;

	glBegin(GL_QUADS);
		for(GLint x = 0; x < L; x++){
			for(GLint y = 0; y < L; y++){

				(pG -> cell(pG -> getId(x,y)) == ALIVE)?(glColor3f(BLACK)):(glColor3f(WHITE));

				glVertex2f(x * xSize + left, y * ySize + bottom);
				glVertex2f((x + 1) * xSize + left, y * ySize + bottom);
				glVertex2f((x + 1) * xSize + left, (y + 1) * ySize + bottom);
				glVertex2f(x * xSize + left,(y + 1) * ySize + bottom);
			}
		}
	glEnd();

	glFlush();
	glutSwapBuffers();
}


void reshape(int w, int h){

	windowWidth	 = w;
	windowHeight = h;

	glViewport(0, 0, windowWidth, windowHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(left, right, bottom, top);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutPostRedisplay();
}


void update(int value){

	pG -> step();
	glutPostRedisplay();
	glutTimerFunc(1000 / FPS, update, 0);
}

