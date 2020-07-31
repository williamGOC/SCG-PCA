#include "Interfaz_OpenGL_to_CUDA.h"

PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

#define REFRESH_DELAY 1 //ms

GLuint  bufferObj;
struct cudaGraphicsResource *resource;

game * pG;

int paso;

int sqx, sqy;
int signo;
float fieldmax, fieldmin;
int planoz;

void initializeSystem(){
 	
 	std::cout << "inicializando variables del sistema..."; 
  	paso = 1;
  	signo = 1;
  	planoz = 0;		
  	pG = new game(DIM, 0.5);		
  	std::cout << "done" << std::endl;
}

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;

float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;


// rendering callbacks
static void display();
static void keyboard(unsigned char key, int x, int y);
void timerEvent(int value);
void motion(int x, int y);
void mouse(int button, int state, int x, int y);


//GL funcionality
void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_res, unsigned int pbo_res_flags);
void deletePBO(GLuint *pbo, struct cudaGraphicsResource *pbo_res);

// Cudaa functionality
void changePixels(struct cudaGraphicsResource **pbo_res);



__device__ unsigned getId(int x, int y){

	// periodic bonduary conditions:
	while(x >=DIM)
		x -=DIM;

	while(x < 0)
		x +=DIM;

	while(y >=DIM)
		y -=DIM;

	while(y < 0)
		y +=DIM;

	return x + y *DIM;
}

// based on ripple code, but uses uchar4 which is the type of data
// graphic inter op uses. see screenshot - basic2.png
__global__ void kernel(ubyte* world, ubyte * fancyWorld, uchar4 *ptr){

	// get the world coordinate:
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// id in 1D array which represents the world:
	unsigned id = getId(x, y);

	// create a copy of the world.
	fancyWorld[id] = world[id];

	// wait for all threads to finish copying:
	__syncthreads();


	// run a steps:
	// determine new state by rules of Conway's Game of Life
	ubyte state = fancyWorld[id];
	ubyte newstate = state;

	// calculate number of alive neihbors:
	unsigned short aliveNeighbors = 0;

	for(short x_offset = -1; x_offset <= 1; x_offset++){
		for(short y_offset = -1; y_offset <= 1; y_offset++){
			aliveNeighbors += (x_offset != 0 || y_offset != 0)?(world[getId(x + x_offset, y + y_offset)]):(0);
		}
	}
    
    newstate = (aliveNeighbors == 3 || (aliveNeighbors == 2 && state))?(1):(0);

    // wait for all threads to determine new state:
    __syncthreads();

    // save spins in shared memory:
    fancyWorld[id] = newstate;

    // wait for all threads to copy new state to shared memory:
    __syncthreads();

    world[id] = fancyWorld[id];


    ptr[id].x = 0;
    ptr[id].y = (fancyWorld[id] == DEAD)?(255):(0);
    ptr[id].z = 0;
    ptr[id].w = 0;
}

void launch_kernel(uchar4 *ptr){

    // execute the kernel
    dim3 block(16, 16, 1);
    dim3 grid(MESH_DIM / block.x, MESH_DIM/ block.y, 1);

    kernel<<<grid, block>>>(pG -> dptr_W, pG -> dptr_F, ptr);
    //std::swap(pG -> ptrWorld, pG -> ptrfancyWorld);
}


int main(int argc, char **argv){
    
    cudaDeviceProp  prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

    // tell CUDA which dev we will be using for graphic interop
    // from the programming guide:  Interoperability with OpenGL
    // requires that the CUDA device be specified by
    // cudaGLSetGLDevice() before any other runtime calls.

    HANDLE_ERROR(cudaGLSetGLDevice(dev));

    // these GLUT calls need to be made before the other OpenGL
    // calls, else we get a seg fault
    glutInit(&argc, argv);
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( DIM, DIM );
    glutCreateWindow( "bitmap" );

    glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

    createPBO(&bufferObj, &resource, cudaGraphicsMapFlagsNone);

    initializeSystem();


    // set up GLUT and kick off main loop
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutMainLoop();
}


static void keyboard(unsigned char key, int x, int y){
    switch (key) {
        case 27:
            // clean up OpenGL and CUDA
            HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
            glDeleteBuffers( 1, &bufferObj );
            exit(0);
    }
}

static void display(){
    // we pass zero as the last parameter, because out bufferObj is now
    // the source, and the field switches from being a pointer to a
    // bitmap to now mean an offset into a bitmap object
    changePixels(&resource);
    glDrawPixels( DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glutSwapBuffers();
}

void timerEvent(int value){

    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}


void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_res, unsigned int pbo_res_flags){

    // the first three are standard OpenGL, the 4th is the CUDA reg 
    // of the bitmap these calls exist starting in OpenGL 1.5
    assert(pbo);

    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(pbo_res, *pbo, pbo_res_flags));

}

void deletePBO(GLuint *pbo, struct cudaGraphicsResource *pbo_res){

    // unregister this buffer object with CUDA
    HANDLE_ERROR(cudaGraphicsUnregisterResource(pbo_res));

    glBindBuffer(1, *pbo);
    glDeleteBuffers(1, pbo);

    *pbo = 0;
}


void changePixels(struct cudaGraphicsResource **pbo_res){

    // map OpenGL buffer object for writing from CUDA
    uchar4 *devPtr;
    HANDLE_ERROR( cudaGraphicsMapResources( 1, pbo_res, NULL));
    size_t  size;
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, *pbo_res));

    launch_kernel(devPtr);

    // unmap buffer object
    HANDLE_ERROR(cudaGraphicsUnmapResources(1, pbo_res, NULL));
}

void mouse(int button, int state, int x, int y){
    
    if (state == GLUT_DOWN){
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP){
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    sqx=(x>0 && x<DIM)?(x):(sqx); 
    sqy=(y>0 && y<DIM)?(y):(sqy);
    
    std::cout << sqx << " (clicks) " << sqy << " " << signo << std::endl;   
}


void motion(int x, int y){

  	float dx, dy;
  	dx = (float)(x - mouse_old_x);
   	dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1){
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4){

        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    sqx=(x>0 && x<DIM)?(x):(sqx); 
    sqy=(y>0 && y<DIM)?(y):(sqy);

    std::cout << sqx << " (motion) " << sqy << std::endl;
    pG -> addsquare(sqx,DIM-sqy,planoz, 10, signo);
}