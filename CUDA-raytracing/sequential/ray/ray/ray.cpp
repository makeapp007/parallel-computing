#include <time.h>

#include <stdio.h>
#include <stdlib.h>

#include <Windows.h>

#include "udray.h"
#include "glm.h"
#include <iostream>
using namespace std;

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

extern Camera *ray_cam;       // camera info
extern int image_i, image_j;  // current pixel being shaded
extern bool wrote_image;      // has the last pixel been shaded?

							  // reflection/refraction recursion control

extern int maxlevel;          // maximum depth of ray recursion 
extern double minweight;      // minimum fracti`onal contribution to color

							  // these describe the scene

extern vector < GLMmodel * > model_list;
extern vector < Sphere * > sphere_list;
extern vector < Light * > light_list;

clock_t begin_time;

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// intersect a ray with the entire scene (.obj models + spheres)

// x, y are in pixel coordinates with (0, 0) the upper-left hand corner of the image.
// color variable is result of this function--it carries back info on how to draw the pixel

void trace_ray(int level, double weight, Ray *ray, Vect color)
{
	Intersection *nearest_inter = NULL;
	Intersection *inter = NULL;
	int i;

	// test for intersection with all .obj models
//#pragma omp parallel for 
	for (i = 0; i < model_list.size(); i++) {
		inter = intersect_ray_glm_object(ray, model_list[i]);
		update_nearest_intersection(&inter, &nearest_inter);
	}

	// test for intersection with all spheres
//#pragma omp parallel for 
	for (i = 0; i < sphere_list.size(); i++) {
		inter = intersect_ray_sphere(ray, sphere_list[i]);
		update_nearest_intersection(&inter, &nearest_inter);
	}

	// "color" the ray according to intersecting surface properties

	// choose one of the simpler options below to debug or preview your scene more quickly.
	// another way to render faster is to decrease the image size.

	if (nearest_inter) {
		//shade_ray_false_color_normal(nearest_inter, color);
		//    shade_ray_intersection_mask(color);  
		//shade_ray_diffuse(ray, nearest_inter, color);
		shade_ray_recursive(level, weight, ray, nearest_inter, color);
	}

	// color the ray using a default

	else
		shade_ray_background(ray, color);

	if (nearest_inter)
	{
		free(nearest_inter);
	}
	delete ray;// could be wrong
}

//----------------------------------------------------------------------------

// test for ray-sphere intersection; return details of intersection if true

Intersection *intersect_ray_sphere(Ray *ray, Sphere *S)
{
	// FILL IN CODE (line below says "no" for all spheres, so replace it)
	Intersection *interpoint;
	Vect C;
	double V, BB, delta, t;
	VectSub(S->P, ray->orig, C);
	V = VectDotProd(ray->dir, C);
	BB = VectDotProd(C, C) - V * V;
	if (S->radius * S->radius - BB< 0)
		return NULL;
	else
	{
		delta = S->radius * S->radius - BB;
		t = V - sqrt(delta);
	}
	interpoint = make_intersection();
	interpoint->t = t;
	interpoint->surf = S->surf;
	VectAddS(interpoint->t, ray->dir, ray->orig, interpoint->P);
	VectSub(interpoint->P, S->P, interpoint->N);
	//interpoint->N[0] = interpoint->N[0] / VectDotProd(interpoint->N, interpoint->N);
	//interpoint->N[1] = interpoint->N[1] / VectDotProd(interpoint->N, interpoint->N);
	//interpoint->N[2] = interpoint->N[2] / VectDotProd(interpoint->N, interpoint->N);
	VectUnit(interpoint->N);

	return interpoint;
	return NULL;
}


//----------------------------------------------------------------------------

// only local, ambient + diffuse lighting (no specular, shadows, reflections, or refractions)

void shade_ray_diffuse(Ray *ray, Intersection *inter, Vect color)
{
	Vect L;
	double diff_factor;

	// iterate over lights
//#pragma omp parallel for 
	for (int i = 0; i < light_list.size(); i++) {

		// AMBIENT

		color[R] += inter->surf->amb[R] * light_list[i]->amb[R];
		color[G] += inter->surf->amb[G] * light_list[i]->amb[G];
		color[B] += inter->surf->amb[B] * light_list[i]->amb[B];

		// DIFFUSE 

		// FILL IN CODE
		///////////////////////////
		VectSub(light_list[i]->P, inter->P, L);
		//L[0] = L[0] / VectDotProd(L, L);
		//L[1] = L[1] / VectDotProd(L, L);
		//L[2] = L[2] / VectDotProd(L, L);
		VectUnit(L);
		diff_factor = VectDotProd(L, inter->N);
		if (diff_factor > 0)
		{

			color[R] += diff_factor * inter->surf->diff[R] * light_list[i]->diff[R];
			color[G] += diff_factor * inter->surf->diff[G] * light_list[i]->diff[G];
			color[B] += diff_factor * inter->surf->diff[B] * light_list[i]->diff[B];
		}



		///////////////////////////

	}

	// clamp color to [0, 1]

	VectClamp(color, 0, 1);
}









//----------------------------------------------------------------------------

// same as shade_ray_diffuse(), but add specular lighting + shadow rays (i.e., full Phong illumination model)

void shade_ray_local(Ray *ray, Intersection *inter, Vect color)
{
	// FILL IN CODE 
	if (inter == NULL)
	{
		return;
	}
//#pragma omp parallel for
	for (size_t i = 0; i < light_list.size(); i++)
	{

		int ifskiplight = 0;
		Vect shadow_ori;
		Vect shadow_light;
		double shadow_t;
		VectSub(light_list[i]->P, inter->P, shadow_light);//
		shadow_t = VectMag(shadow_light);
		VectUnit(shadow_light);
		VectAddS(0.01, shadow_light, inter->P, shadow_ori);
		Ray* shadow_ray = make_ray(shadow_ori, shadow_light);

//#pragma omp parallel for 
		for (size_t j = 0; j < sphere_list.size(); j++)
		{
			Intersection* shadow_inter = intersect_ray_sphere(shadow_ray, sphere_list[j]);
			if (shadow_inter != NULL)
			{
				//printf("find shadow");
				//Vect shadow_inter_vec;
				//VectSub(shadow_ori, shadow_inter->P, shadow_inter_vec);
				double tempLen = shadow_inter->t;
				if (tempLen > 0 && tempLen < shadow_t)
				{
					free(shadow_inter);
					ifskiplight = 1;
					break;
				}
				else
				{
					free(shadow_inter);
					continue;
				}
			}
		}
		if (shadow_ray)
		{
			free(shadow_ray);
		}
//#pragma omp parallel for 
		for (size_t j = 0; j < model_list.size(); j++)
		{
			Intersection* shadow_inter = intersect_ray_glm_object(shadow_ray, model_list[j]);
			if (shadow_inter != NULL)
			{
				//Vect shadow_inter_vec;
				//VectSub(shadow_ori, shadow_inter->P, shadow_inter_vec);
				double tempLen = shadow_inter->t;
				if (tempLen > 0 && tempLen < shadow_t)
				{
					ifskiplight = 1;
					break;
				}
				else
				{
					continue;
				}
			}
		}

		////////////////////////////////////////////////////
		if (ifskiplight == 1)
		{
			//printf("skip");
			//ifskiplight = 0;
			continue;
		}
		// AMBIENT
		else
		{
			////////////////////////////////////////////////////////
			Vect L;
			double diff_factor;
			color[R] += inter->surf->amb[R] * light_list[i]->amb[R];
			color[G] += inter->surf->amb[G] * light_list[i]->amb[G];
			color[B] += inter->surf->amb[B] * light_list[i]->amb[B];

			// DIFFUSE 

			// FILL IN CODE
			///////////////////////////
			VectSub(light_list[i]->P, inter->P, L);
			//L[0] = L[0] / VectDotProd(L, L);
			//L[1] = L[1] / VectDotProd(L, L);
			//L[2] = L[2] / VectDotProd(L, L);
			VectUnit(L);
			diff_factor = VectDotProd(L, inter->N);
			if (diff_factor > 0)
			{

				color[R] += diff_factor * inter->surf->diff[R] * light_list[i]->diff[R];
				color[G] += diff_factor * inter->surf->diff[G] * light_list[i]->diff[G];
				color[B] += diff_factor * inter->surf->diff[B] * light_list[i]->diff[B];
			}
			//////////////////////////////////////////////////
			else
			{
				diff_factor = 0;
			}
			//specular

			Vect N, Re, temp, V, H;
			VectSub(ray->orig, inter->P, V);
			VectUnit(V);
			VectAddS(1, V, L, H);
			VectUnit(H);
			//if (VectDotProd(H,inter->N)<=0)
			//{
			//	printf("error");
			//}
			//VectCopy(inter->N, N);
			//VectUnit(N);
			//double cof = 2 * diff_factor;
			//VectAddS(cof-1, inter->N, inter->N, temp);
			//VectSub(temp, L, Re);
			//VectUnit(Re);

			//reflection_direction(N,L, Re);
			//double spec_factor = pow(VectDotProd(ray->dir, Re),inter->surf->spec_exp);
			double spec_factor = pow(VectDotProd(inter->N, H), inter->surf->spec_exp);
			color[R] += spec_factor * light_list[i]->spec[R] * inter->surf->spec[R];
			color[G] += spec_factor * light_list[i]->spec[G] * inter->surf->spec[G];
			color[B] += spec_factor * light_list[i]->spec[B] * inter->surf->spec[B];

			//VectClamp(color, 0, 1);
		}
		//VectClamp(color, 0, 1);
	}
	VectClamp(color, 0, 1);
}

//----------------------------------------------------------------------------

// full shading model: ambient/diffuse/specular lighting, shadow rays, recursion for reflection, refraction

// level = recursion level (only used for reflection/refraction)

void shade_ray_recursive(int level, double weight, Ray *ray, Intersection *inter, Vect color)
{
	Surface *surf = inter->surf;
	int i;


	// initialize color to Phong reflectance model
	
	shade_ray_local(ray, inter, color);

	//printf("professional!");
	// if not too deep, recurse
	//printf("level = %d", level);
	if (level + 1 < maxlevel) {

		// add reflection component to color
		//printf("%f",surf->reflectivity);
		if (surf->reflectivity * weight > minweight) {

			// FILL IN CODE
			Vect newray_dir;
			Vect newray_dir_neg;
			Vect newray_ori;
			//Vect L;
			//Vect reN;
			Vect newcolor;
			newcolor[R] = 0;
			newcolor[G] = 0;
			newcolor[B] = 0;

			//VectSub(ray->orig, inter->P, L);
			//VectUnit(L);
			reflection_direction(ray->dir, inter->N, newray_dir_neg);
			VectNegate(newray_dir_neg, newray_dir);
			VectUnit(newray_dir);
			VectAddS(0.01, newray_dir_neg, inter->P, newray_ori);
			//double temp = VectDotProd(inter->N, L);
			//VectAddS(2 * temp - 1, inter->N, inter->N, reN);
			//VectSub(reN,L, newray_dir);
			Ray* newray = make_ray(newray_ori, newray_dir);
			trace_ray(level + 1, weight, newray, newcolor);

			color[R] += newcolor[R] * surf->reflectivity*weight;
			color[G] += newcolor[G] * surf->reflectivity*weight;
			color[B] += newcolor[B] * surf->reflectivity*weight;
			VectClamp(color, 0, 1);
		}

		// add refraction component to color

		if (surf->transparency * weight > minweight) {

			Vect newray_dir, newray_ori;
			Vect temp, N;
			Vect newray_dir1;
			Vect newcolor;
			newcolor[R] = 0;
			newcolor[G] = 0;
			newcolor[B] = 0;

			//VectSub(ray->orig, inter->P, L);
			//VectUnit(L);
			//double setai = acos(VectDotProd(ray->dir, inter->N));
			//double setat = asin(sin(setai)*surf->transparency);

			VectAddS(surf->transparency - 1, ray->dir, ray->dir, newray_dir);
			//VectAddS((surf->transparency*cos(setai)+abs(cos(setat))) - 1, inter->N, inter->N, newray_dir2);
			//VectSub(newray_dir1, newray_dir2, term3);
			//VectCopy(newray_dir, newray_dir1);
			VectAddS(0.01, newray_dir, inter->P, newray_ori);
			VectUnit(newray_dir);

			Ray* newray = make_ray(newray_ori, newray_dir);
			trace_ray(level + 1, weight, newray, newcolor);


			color[R] += newcolor[R] * surf->transparency*weight;
			color[G] += newcolor[G] * surf->transparency*weight;
			color[B] += newcolor[B] * surf->transparency*weight;
			VectClamp(color, 0, 1);
			// GRAD STUDENTS -- FILL IN CODE

		}
	}
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// ray trace another pixel if the image isn't finished yet

void idle2()
{

	if (image_j < ray_cam->im->h) {

		raytrace_one_pixel(image_i, image_j);

		image_i++;

		if (image_i == ray_cam->im->w) {
			image_i = 0;
			image_j++;
		}
	}

	// write rendered image to file when done

	else if (!wrote_image) {

		clock_t end = clock();
		cout << "Running time: " << (double)(end - begin_time) / CLOCKS_PER_SEC * 1000 << "ms" << endl;


		write_PPM("output.ppm", ray_cam->im);

		wrote_image = true;
	}

	glutPostRedisplay();
}
 


DWORD WINAPI idle(LPVOID lpParam)
{
	clock_t begin2 = clock();
	//int max_threads = omp_get_max_threads();
//	printf(" Preparing for parallel computation...\n Max # of threads=%d\n", max_threads);
	//#pragma omp parallel for schedule(dynamic)
	for (image_i = 0; image_i < ray_cam->im->w; ++image_i)
	{
		for (int image_j = 0; image_j < ray_cam->im->h; ++image_j)
		{
			raytrace_one_pixel(image_i, image_j);
			//printf("%d %d\n", image_i, image_j);
		}
	}

	clock_t end = clock();
		cout << "Running time: " << (double)(end - begin2) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
		glutPostRedisplay();

		write_PPM("foutput.ppm", ray_cam->im);
	return NULL;
}

//----------------------------------------------------------------------------

// show the image so far

void display(void)
{
	// draw it!

	glPixelZoom(1, -1);
	glRasterPos2i(0, ray_cam->im->h);
	glDrawPixels(ray_cam->im->w, ray_cam->im->h, GL_RGBA, GL_FLOAT, ray_cam->im->data);

	glFlush();
}

//----------------------------------------------------------------------------

void init()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, ray_cam->im->w, 0.0, ray_cam->im->h);
}

//----------------------------------------------------------------------------

int main(int argc, char** argv)
{
	glutInit(&argc, argv);

	// initialize scene (must be done before scene file is parsed)

	init_raytracing();
	/////
	parse_scene_file("test.scene", ray_cam);
	/*
	if (argc == 2)
	parse_scene_file(argv[1], ray_cam);
	else {
	printf("missing .scene file\n");
	exit(1);
	}
	*/
	// opengl business

	begin_time = clock();

	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(ray_cam->im->w, ray_cam->im->h);
	glutInitWindowPosition(500, 300);
	glutCreateWindow("hw3");
	init();

	glutDisplayFunc(display);
	glutIdleFunc(idle2);
	//CreateThread(NULL, 0, idle, NULL, 0, NULL);
	glutMainLoop();

	return 0;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
