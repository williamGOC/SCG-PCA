#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>


#include "game_of_life.c"

#define WHITE 1.0, 1.0, 1.0
#define BLACK 0.0, 0.0, 0.0

GLint windowWidth  = 1000;
GLint windowHeight = 1000;

GLint FPS = 24;
GLfloat left = 0.0;
GLfloat right = 1.0;
GLfloat bottom = 0.0;
GLfloat top = 1.0;
GLint game_width = 500;
GLint game_height = 500;

game pG;

void display(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	GLfloat xSize = (right - left) / game_width;
	GLfloat ySize = (top - bottom) / game_height;

	glBegin(GL_QUADS);
		for(GLint x = 0; x < game_width; ++x) {
			for(GLint y = 0; y < game_height; ++y){

				pG -> world[getId(x,y)]?glColor3f(BLACK):glColor3f(WHITE);

				glVertex2f(    x*xSize+left,    y*ySize+bottom);
				glVertex2f((x+1)*xSize+left,    y*ySize+bottom);
				glVertex2f((x+1)*xSize+left,(y+1)*ySize+bottom);
				glVertex2f(    x*xSize+left,(y+1)*ySize+bottom);
			}
		}
	glEnd();
	
	glFlush();
	glutSwapBuffers();	
}


void reshape(int w, int h){

	windowWidth = w;
	windowHeight = h;

	glViewport(0, 0, windowWidth, windowHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(left, right, bottom, top);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutPostRedisplay();
}


void update(int value) {

	step(pG);

	glutPostRedisplay();
	glutTimerFunc(1000 / FPS, update, 0);
}


int main(int argc, char **argv) {
	
	glutInit(&argc, argv);
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Aplication: Game of Life");
	glClearColor(1, 1, 1, 1);
	
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);

	pG = builderSystem(0.5);
	randomWorld(pG);
		
	update(0);
	glutMainLoop();
		
	return 0;
}
