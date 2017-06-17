
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "glut.h"
#include "udray.cuh"

#include "glm.cuh"
#include <stdio.h>
#include <time.h>
#include <iostream>
using namespace std;
extern Camera *ray_cam;       // camera info

extern vector < GLMmodel * > model_list;
extern vector < Sphere * > sphere_list;
extern vector < Light * > light_list;
extern vector < Surface * > model_surf_list;


GLMmodel ** dev_model_list;
Sphere * dev_sphere_list;
Light *  dev_light_list;
Surface** dev_model_surf_list;
Camera * dev_ray_cam;



GLfloat* test_data;


GLfloat* dev_data;
Image *dev_im;
//Camera * ocam = make_camera();
//GLfloat *odata;
int flag = 0;
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void display(void)
{
	// draw it!
	

	printf("in display");
	printf("data[0] is :%f", ray_cam->im->data[0]);
	glPixelZoom(1, -1);
	glRasterPos2i(0, ray_cam->im->h);
	glDrawPixels(ray_cam->im->w, ray_cam->im->h, GL_RGBA, GL_FLOAT, ray_cam->im->data);

	glFlush();

}



int times = 0;
void idle2()
{
	//record time
	clock_t begin = clock();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, NULL);//记录开始
	cudaEventRecord(stop, NULL);//记录开始



	cudaError_t cudaStatus; 
	printf("\n in idle2\n");
	dim3 block(32, 32, 1);
	dim3 grid(ray_cam->im->w / block.x, ray_cam->im->h / block.y, 1);
	raytrace_one_pixel << < grid, block , 0, 0 >> > (dev_ray_cam, dev_model_list, dev_sphere_list, dev_light_list, dev_model_surf_list);


	///test

	if (times % 1 == 0) {
		Camera * trans_cam;
		Vect direct;
		direct[0] = 1.0;
		direct[1] = 0.0;
		direct[2] = 0.0;

		trans_cam = (Camera*)malloc(sizeof(Camera));
		cudaStatus = cudaMemcpy(trans_cam, dev_ray_cam, sizeof(Camera), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "stupid cuda failed!");
			goto Error;
		}

		VectAddS(0.005, direct, trans_cam->eye, trans_cam->eye);
		//VectSub(trans_cam->eye, trans_cam->eye, trans_cam->eye);
		VectPrint(trans_cam->eye);
		//copy camera
		printf("\n now copy camera \n");
		cudaStatus = cudaMemcpy(dev_ray_cam, trans_cam, sizeof(Camera), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}
	times++;



	////////////////////
	//copy data back
	printf("after display\n");
	if (test_data) {
		free(test_data);
	}
	test_data = (GLfloat *)malloc(4 * ray_cam->im->w * ray_cam->im->h * sizeof(GLfloat));


	printf("before copy \n");
	cudaStatus = cudaMemcpy(test_data, dev_data, 4 * ray_cam->im->w*ray_cam->im->h * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	printf("after copy \n");

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMcpy in main function  failed!");
		goto Error;
	}
	printf("copy success\n");
	ray_cam->im->data = test_data;
	//printf("color = %f %f %f %f\n", ray_cam->im->data[0], ray_cam->im->data[86401], ray_cam->im->data[86402], ray_cam->im->data[3]);
	printf("copy success\n");


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	//print running time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("total gpu time %f(ms) \n", elapsedTime);
	clock_t end = clock();
	cout << "Running time: " << (double)(end - begin) / CLOCKS_PER_SEC * 1000 << "ms" << endl;



	glutPostRedisplay();
	write_PPM("foutput.ppm", ray_cam->im);
	//free(test_data);
	//printf("after failed \n");
Error:
	return;
}


void init()
{
	printf("in init\n  im->w = %d", ray_cam->im->w);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, ray_cam->im->w, 0.0, ray_cam->im->h);
}


int main(int argc, char** argv)
{
    // Add vectors in parallel.
	cudaError_t cudaStatus;

	glutInit(&argc, argv);

	// initialize scene (must be done before scene file is parsed)

	init_raytracing();
	/////
	parse_scene_file("test.scene", ray_cam);

	Sphere * tL = (Sphere*)malloc(sizeof(Sphere*));

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSethost failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	
	//goto GPU buffers
	cudaStatus = cudaMalloc((void**)&dev_light_list, light_list.size() * sizeof(Light));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_sphere_list, sphere_list.size() * sizeof(Sphere));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	printf("test1: %lf\n", sphere_list[2]->radius);


	printf("\nshit here \n  ");
	printf("test2: %d %d %d", sizeof(dev_light_list), light_list.size(), sizeof(light_list));
	//copy GPU buffers

	for (size_t i = 0; i < light_list.size(); i++)
	{
		cudaStatus = cudaMemcpy(&dev_light_list[i], light_list[i], sizeof(Light), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}
	for (size_t i = 0; i < sphere_list.size(); i++)
	{
		Surface * dev_surf;
		cudaStatus = cudaMalloc((void**)&dev_surf, sizeof(Surface));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_surf, sphere_list[i]->surf, sizeof(Surface), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		sphere_list[i]->surf = dev_surf;
		//finish coping sphere

		cudaStatus = cudaMemcpy(&dev_sphere_list[i], sphere_list[i], sizeof(Sphere), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}




	//Camera	*test = make_camera();
	//cudaMemcpy(test, dev_ray_cam->im->data, sizeof(Camera), cudaMemcpyDeviceToHost);
	//printf("\n%lf \n", test->center[0]);
	//printf("\n%lf \n", ray_cam->center[0]);
	
	cudaStatus = cudaMalloc((void**)&dev_model_surf_list, sizeof(Surface*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}



	//finish copy GPU buffers
	//cudaStatus = cudaMemcpy(&tL, &dev_sphere_list[2], sizeof(Light*), cudaMemcpyDeviceToHost);


	printf("\n radius is %lf \n", tL->radius);
	//cuda///////////////////////////////////////////////////////////
	//parse_scene_file("test.scene", ray_cam);
	printf("size of Camera:%d\n", sizeof(Camera));
	//Sleep(1000);

	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(ray_cam->im->w, ray_cam->im->h);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("CudaRaytracing");
	init();

	//cudaError_t cudaStatus;
	printf("\n  calling glutidle \n");
	glutIdleFunc(idle2);
	//CreateThread(NULL, 0, idle, NULL, 0, NULL);
	//ifdo();

	//idle2();
	//Sleep(1000);
	glutDisplayFunc(display);

	glutPostRedisplay();
	printf("aaaa: %d %d %d", model_list.size(), light_list.size(), sphere_list.size() );




	////////////////////////////


	Image * backup_im;
	backup_im = ray_cam->im;
	GLfloat * backup_data;
	backup_data = ray_cam->im->data;

	cudaStatus = cudaMalloc((void**)&dev_data, 4 * ray_cam->im->w*ray_cam->im->h * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_data, ray_cam->im->data, 4 * ray_cam->im->w*ray_cam->im->h * sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMcpy failed!");
		goto Error;
	}


	ray_cam->im->data = dev_data;

	cudaStatus = cudaMalloc((void**)&dev_im, sizeof(Image));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_im, ray_cam->im, sizeof(Image), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMcpy failed!");
		goto Error;
	}

	ray_cam->im = dev_im;

	cudaStatus = cudaMalloc((void**)&dev_ray_cam, sizeof(Camera));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//copy camera
	printf("\n now copy camera \n");
	cudaStatus = cudaMemcpy(dev_ray_cam, ray_cam, sizeof(Camera), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//backup camera
	ray_cam->im = backup_im;
	ray_cam->im->data = backup_data;

	////////////////////////
	glutMainLoop();

	printf("after opengl\n");

	getchar();
Error:
	cudaFree(dev_light_list);
	cudaFree(dev_sphere_list);
	cudaFree(dev_model_list);
	cudaFree(dev_model_surf_list);
	cudaFree(dev_ray_cam);
	return cudaStatus;

	return 0;
}


