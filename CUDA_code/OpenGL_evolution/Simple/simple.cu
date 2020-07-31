#include "./common/book.h"
#include "./common/cpu_bitmap.h"

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <assert.h>

PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

#define DIM    512
#define MESH_DIM 512
#define REFRESH_DELAY 1 //ms

unsigned PASS = 0;

GLuint  bufferObj;
struct cudaGraphicsResource *resource;


//GL funcionality
void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_res, unsigned int pbo_res_flags);
void deletePBO(GLuint *pbo, struct cudaGraphicsResource *pbo_res);

// rendering callbacks
static void keyboard(unsigned char key, int x, int y);
static void display();
void timerEvent(int value);

// Cudaa functionality
void changePixels(struct cudaGraphicsResource **pbo_res);


// based on ripple code, but uses uchar4 which is the type of data
// graphic inter op uses. see screenshot - basic2.png
__global__ void kernel( uchar4 *ptr, unsigned step) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * DIM; 

    // accessing uchar4 vs unsigned char*
    ptr[offset].x = step;
    ptr[offset].y = 0;
    ptr[offset].z = 255;
    ptr[offset].w = 0;
}

void launch_kernel(uchar4 *ptr){

    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(MESH_DIM / block.x, MESH_DIM/ block.y, 1);
    PASS = (PASS > 255)?(0):(PASS);
    kernel<<<grid, block>>>(ptr, PASS);
}


int main( int argc, char **argv ) {
    
    cudaDeviceProp  prop;
    int dev;

    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 0;
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );

    // tell CUDA which dev we will be using for graphic interop
    // from the programming guide:  Interoperability with OpenGL
    //     requires that the CUDA device be specified by
    //     cudaGLSetGLDevice() before any other runtime calls.

    HANDLE_ERROR( cudaGLSetGLDevice( dev ) );

    // these GLUT calls need to be made before the other OpenGL
    // calls, else we get a seg fault
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( DIM, DIM );
    glutCreateWindow( "bitmap" );

    glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

    createPBO(&bufferObj, &resource, cudaGraphicsMapFlagsNone);

    // do work with the memory dst being on the GPU, gotten via mapping
    //changePixels(&resource);

    // set up GLUT and kick off main loop
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
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
    PASS++;
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