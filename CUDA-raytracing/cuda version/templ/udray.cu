//----------------------------------------------------------------------------
// UD Ray v2.0
// copyright 2008, University of Delaware
// Christopher Rasmussen
//----------------------------------------------------------------------------
#include <omp.h>
#include "udray.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <vector>

//----------------------------------------------------------------------------
// hosts
//----------------------------------------------------------------------------

#define model_list_size 2
#define sphere_list_size 8
#define light_list_size 3
#define maxlevel 2
#define minweight 0.01
#define weight 1
#define weight111 1


vector < GLMmodel * > model_list;
vector < Surface * > model_surf_list;   // material properties associated with .obj models (to avoid changing glm.hh)
vector < Sphere * > sphere_list;    // material properties are linked inside each sphere struct
vector < Light * > light_list;

Camera *ray_cam;

Ray *eye_ray;
int image_i, image_j;
bool wrote_image;

double rayeps;         // round-off error tolerance     

					   //----------------------------------------------------------------------------
					   // Functions
					   //----------------------------------------------------------------------------

					   // cross product v3 = v1 x v2




					   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void VectdCross(Vect v1, Vect v2, Vect v3)
{
	v3[X] = v1[Y] * v2[Z] - v1[Z] * v2[Y];
	v3[Y] = v1[Z] * v2[X] - v1[X] * v2[Z];
	v3[Z] = v1[X] * v2[Y] - v1[Y] * v2[X];

}

//----------------------------------------------------------------------------

__device__ void VectdPrint(Vect v)
{
	printf("%.2lf %.2lf %.2lf\n", v[X], v[Y], v[Z]);
}

//----------------------------------------------------------------------------

// dst = src

__device__ void VectdCopy(Vect dst, Vect src)
{
	dst[X] = src[X];
	dst[Y] = src[Y];
	dst[Z] = src[Z];
}

//----------------------------------------------------------------------------

// scaled addition v3 = t * v1 + v2

__device__ void VectdAddS(double t, Vect v1, Vect v2, Vect v3)
{
	v3[X] = t * v1[X] + v2[X];
	v3[Y] = t * v1[Y] + v2[Y];
	v3[Z] = t * v1[Z] + v2[Z];
}

//----------------------------------------------------------------------------

// v3 = v1 - v2

__device__ void VectdSub(Vect v1, Vect v2, Vect v3)
{
	v3[X] = v1[X] - v2[X];
	v3[Y] = v1[Y] - v2[Y];
	v3[Z] = v1[Z] - v2[Z];
}

//----------------------------------------------------------------------------

// vector length

__device__ float VectdMag(Vect v)
{
	return sqrt(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}

//----------------------------------------------------------------------------
__device__ void Vectdmul(double k, Vect v1, Vect v2)
{
	v2[X] = v1[X] * k;
	v2[Y] = v1[Y] * k;
	v2[Z] = v1[Z] * k;
}
// make vector have unit length; return original length

__device__ double VectdUnit(Vect v)
{
	double mag;

	mag = VectdMag(v);
	v[X] /= mag;
	v[Y] /= mag;
	v[Z] /= mag;

	return mag;
}

//----------------------------------------------------------------------------

// negate all components of vector

__device__   void VectdNegate(Vect v, Vect vneg)
{
	vneg[X] = -v[X];
	vneg[Y] = -v[Y];
	vneg[Z] = -v[Z];
}

//----------------------------------------------------------------------------

__device__ void VectdClamp(Vect v, double low, double high)
{
	for (int i = 0; i < 3; i++) {

		if (v[i] < low)
			v[i] = low;
		else if (v[i] > high)
			v[i] = high;
	}
}

//----------------------------------------------------------------------------

// dot product of two vectors

__device__ double VectdDotProd(Vect v1, Vect v2)
{
	return v1[X] * v2[X] + v1[Y] * v2[Y] + v1[Z] * v2[Z];
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////









__host__ void VectCross(Vect v1, Vect v2, Vect v3)
{
	v3[X] = v1[Y] * v2[Z] - v1[Z] * v2[Y];
	v3[Y] = v1[Z] * v2[X] - v1[X] * v2[Z];
	v3[Z] = v1[X] * v2[Y] - v1[Y] * v2[X];

}

//----------------------------------------------------------------------------

__host__ void VectPrint(Vect v)
{
	printf("%.2lf %.2lf %.2lf\n", v[X], v[Y], v[Z]);
}

//----------------------------------------------------------------------------

// dst = src

__host__ void VectCopy(Vect dst, Vect src)
{
	dst[X] = src[X];
	dst[Y] = src[Y];
	dst[Z] = src[Z];
}

//----------------------------------------------------------------------------

// scaled addition v3 = t * v1 + v2

__host__ void VectAddS(double t, Vect v1, Vect v2, Vect v3)
{
	v3[X] = t * v1[X] + v2[X];
	v3[Y] = t * v1[Y] + v2[Y];
	v3[Z] = t * v1[Z] + v2[Z];
}

//----------------------------------------------------------------------------

// v3 = v1 - v2

__host__ void VectSub(Vect v1, Vect v2, Vect v3)
{
	v3[X] = v1[X] - v2[X];
	v3[Y] = v1[Y] - v2[Y];
	v3[Z] = v1[Z] - v2[Z];
}

//----------------------------------------------------------------------------

// vector length

__host__ double VectMag(Vect v)
{
	return sqrt(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}

//----------------------------------------------------------------------------
__host__ void Vectmul(double k, Vect v1, Vect v2)
{
	v2[X] = v1[X] * k;
	v2[Y] = v1[Y] * k;
	v2[Z] = v1[Z] * k;
}
// make vector have unit length; return original length

__host__ double VectUnit(Vect v)
{
	double mag;

	mag = VectMag(v);
	v[X] /= mag;
	v[Y] /= mag;
	v[Z] /= mag;

	return mag;
}

//----------------------------------------------------------------------------

// negate all components of vector

__host__   void VectNegate(Vect v, Vect vneg)
{
	vneg[X] = -v[X];
	vneg[Y] = -v[Y];
	vneg[Z] = -v[Z];
}

//----------------------------------------------------------------------------

__host__ void VectClamp(Vect v, double low, double high)
{
	for (int i = 0; i < 3; i++) {

		if (v[i] < low)
			v[i] = low;
		else if (v[i] > high)
			v[i] = high;
	}
}

//----------------------------------------------------------------------------

// dot product of two vectors

__host__ double VectDotProd(Vect v1, Vect v2)
{
	return v1[X] * v2[X] + v1[Y] * v2[Y] + v1[Z] * v2[Z];
}


//----------------------------------------------------------------------------

// multiply vector by matrix transform

__host__ void TransformVect(Transform M, Vect V, Vect V_prime)
{
	V_prime[X] = M[0] * V[X] + M[4] * V[Y] + M[8] * V[Z] + M[12] * V[W];
	V_prime[Y] = M[1] * V[X] + M[5] * V[Y] + M[9] * V[Z] + M[13] * V[W];
	V_prime[Z] = M[2] * V[X] + M[6] * V[Y] + M[10] * V[Z] + M[14] * V[W];
	V_prime[W] = M[3] * V[X] + M[7] * V[Y] + M[11] * V[Z] + M[15] * V[W];
}

//----------------------------------------------------------------------------

__host__ void TransformPrint(Transform M)
{
	int r, c;

	for (r = 0; r < 4; r++) {
		for (c = 0; c < 4; c++)
			printf("%6.3lf ", MATRC(M, r, c));
		printf("\n");
	}
	printf("\n");
}

//----------------------------------------------------------------------------

__host__ void TransformIdentity(Transform M)
{
	int i, r;

	for (i = 0; i < 16; i++)
		M[i] = 0.0;

	for (r = 0; r < 4; r++)
		MATRC(M, r, r) = 1.0;
}

//----------------------------------------------------------------------------

// M3 = M1 * M2

__host__ void TransformProd(Transform M1, Transform M2, Transform M3)
{
	int r, c, k;

	for (r = 0; r < 4; r++)
		for (c = 0; c < 4; c++) {
			MATRC(M3, r, c) = 0.0;
			for (k = 0; k < 4; k++)
				MATRC(M3, r, c) += MATRC(M1, r, k) * MATRC(M2, k, c);
		}
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// set up some variables before we begin to draw

__host__ void init_raytracing()
{
	ray_cam = make_camera();

	rayeps = 1e-7;

	eye_ray = (Ray *)malloc(sizeof(Ray));

	image_i = 0;
	image_j = 0;

	wrote_image = false;
}

__host__ Image *make_image(int w, int h)
{
	int i, j;
	Image *im;
	Vect red;

	im = (Image *)malloc(sizeof(Image));
	im->w = w;
	im->h = h;

	im->data = (GLfloat *)calloc(4 * w * h, sizeof(GLfloat));

	red[0] = 1.0;
	red[1] = red[2] = 0.0;
	red[3] = 1.0;

	for (j = 0; j < h; j++)
		for (i = 0; i < w; i++)
			drawh_point(i, j, red, im);

	return im;
}
//----------------------------------------------------------------------------

__host__ void drawh_point(int x, int y, Vect color, Image *im)
{
	//      printf("%lf\n", color[0]);

	im->data[4 * (x + im->w * y)] = color[R];
	im->data[1 + 4 * (x + im->w * y)] = color[G];
	im->data[2 + 4 * (x + im->w * y)] = color[B];
	im->data[3 + 4 * (x + im->w * y)] = color[A];
}



// given a pixel location, turn it into an eye ray and trace it to get the color
//raytrace_one_pixel(image_i, image_j, ray_cam, wrote_image, maxlevel, minweight, model_list, sphere_list, light_list);
//s
//----------------------------------------------------------------------------

__global__ void raytrace_one_pixel(Camera* ray_cam, GLMmodel ** model_list, Sphere * sphere_list, Light * light_list, Surface** model_surf_list)
{
	int i, j; //cuda allocate
			  // assign a CUDA thread to every pixel (x,y) 
			  // blockIdx, blockDim and threadIdx are CUDA specific keywords
			  // replaces nested outer loops in CPU code looping over image rows and image columns 
	i = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;

	if (i >= ray_cam->im->w || j >= ray_cam->im->h)
		return;
	//unsigned int i = (height - y - 1)*width + x; // index of current pixel (calculated using thread index) 

	//printf("test if can print \n");


	double x, y;
	//double *eye_color=(double*)malloc(sizeof(double)*4);
	Vect eye_color;

	Ray* eye_ray = new Ray;//could be wrong
	x = 0.5 + (double)i;   // the "center" of the pixel
	y = 0.5 + (double)j;


	set_pixel_ray_direction(x, y, ray_cam, eye_ray);


	eye_color[R] = eye_color[G] = eye_color[B] = 0.0;
	//eye_color[A] = 0.0;
	//cuda////////////////////////////////////////////////////////////////////
	__syncthreads();
	trace_ray(0, eye_ray, eye_color, model_list, sphere_list, light_list, model_surf_list);

	//cuda////////////////////////////////////////////////////////////////////
	// could be wrong
	
	//eye_color[R] = eye_color[G] = eye_color[B] = 0.5;
	delete eye_ray;
	draw_point(i, j, eye_color, ray_cam->im);
	//__syncthreads();
	//free(eye_color);
}


__device__ void trace_ray(int level, Ray *ray, Vect color, GLMmodel ** model_list, Sphere * sphere_list, Light * light_list, Surface** model_surf_list)
{
	Intersection *nearest_inter = NULL;
	Intersection *inter = NULL;
	int i;

	// test for intersection with all .obj models
	//#pragma omp parallel for 


	// test for intersection with all .obj models
	//#pragma omp parallel for 
	//inter = intersect_ray_sphere(ray, &sphere_list[i]);
//	delete inter;
	for (i = 0; i < sphere_list_size; i++) {
		inter = intersect_ray_sphere(ray, &sphere_list[i]);
		//delete inter;
		update_nearest_intersection(&inter, &nearest_inter);
	}
	
	
	// choose one of the simpler options below to debug or preview your scene more quickly.
	// another way to render faster is to decrease the image size.

	if (nearest_inter) {
		//float j = nearest_inter->surf->reflectivity;
		//shade_ray_false_color_normal(nearest_inter, color);
		//    shade_ray_intersection_mask(color);  
		//shade_ray_diffuse(ray, nearest_inter, color);
		shade_ray_recursive(level, ray, nearest_inter, color, model_list, sphere_list, light_list, model_surf_list);
	}
	else{
		shade_ray_background(ray, color);
	}
	if (nearest_inter)
	{
		delete nearest_inter;
	}

	
}

__device__ void shade_ray_background(Ray *ray, Vect color)
{
	color[R] = color[G] = color[B] = 0.0;
}


__device__ void shade_ray_recursive(int level, Ray *ray, Intersection *inter, Vect color, GLMmodel ** model_list, Sphere * sphere_list, Light * light_list, Surface** model_surf_list)
{
	Surface *surf = inter->surf;
	int i;
	


	// initialize color to Phong reflectance model

	shade_ray_local(ray, inter, color, model_list, sphere_list, light_list, model_surf_list);
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
			VectdNegate(newray_dir_neg, newray_dir);
			VectdUnit(newray_dir);
			VectdAddS(0.01, newray_dir_neg, inter->P, newray_ori);
			//double temp = VectDotProd(inter->N, L);
			//VectAddS(2 * temp - 1, inter->N, inter->N, reN);
			//VectSub(reN,L, newray_dir);
			//Ray* newray = make_ray(newray_ori, newray_dir);
			Ray *newray = new Ray;
			VectdCopy(newray->dir,newray_dir);
			VectdCopy(newray->orig,newray_ori);
			//delete ray;
			trace_ray(level + 1, newray, newcolor, model_list, sphere_list, light_list, model_surf_list);
			//free(newray);
			delete newray;
			color[R] += newcolor[R] * surf->reflectivity*weight;
			color[G] += newcolor[G] * surf->reflectivity*weight;
			color[B] += newcolor[B] * surf->reflectivity*weight;
			VectdClamp(color, 0, 1);
		}



		// add refraction component to color
	}
}

__device__ void shade_ray_local(Ray *ray, Intersection *inter, Vect color, GLMmodel ** model_list, Sphere * sphere_list, Light * light_list, Surface ** model_surf_list)
{
	// FILL IN CODE 
	if (inter == NULL)
	{
		return;
	}
	//#pragma omp parallel for
	for (size_t i = 0; i < light_list_size; i++)
	{

		int ifskiplight = 0;
		Vect shadow_ori;
		Vect shadow_light;
		float shadow_t;
		VectdSub(light_list[i].P, inter->P, shadow_light);//
		shadow_t = VectdMag(shadow_light);
		VectdUnit(shadow_light);
		VectdAddS(0.01, shadow_light, inter->P, shadow_ori);
		Ray* shadow_ray = make_ray(shadow_ori, shadow_light);

		//#pragma omp parallel for 
		for (size_t j = 0; j < sphere_list_size; j++)
		{

			Intersection* shadow_inter = intersect_ray_sphere(shadow_ray, &sphere_list[j]);
			if (shadow_inter != NULL)
			{
				float tempLen = shadow_inter->t;
				if (tempLen > 0 && tempLen < shadow_t)
				{
					ifskiplight = 1;
					delete shadow_inter;
					break;
				}
				else
				{
					delete shadow_inter;
					continue;
				}
				//delete shadow_inter;
			}
			else {

			}
		}
		if(shadow_ray!=NULL){
			free(shadow_ray);
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
			__syncthreads();
			////////////////////////////////////////////////////////
			
			Vect L;
			float diff_factor;
			/*
			color[R] += inter->surf->amb[R] * light_list[i].amb[R];
			color[G] += inter->surf->amb[G] * light_list[i].amb[G];
			color[B] += inter->surf->amb[B] * light_list[i].amb[B];
			*/
			// DIFFUSE 

			// FILL IN CODE
			///////////////////////////
			VectdSub(light_list[i].P, inter->P, L);
			//L[0] = L[0] / VectDotProd(L, L);
			//L[1] = L[1] / VectDotProd(L, L);
			//L[2] = L[2] / VectDotProd(L, L);
			VectdUnit(L);
			diff_factor = VectdDotProd(L, inter->N);
			if (diff_factor > 0)
			{

				color[R] += diff_factor * inter->surf->diff[R] * light_list[i].diff[R];
				color[G] += diff_factor * inter->surf->diff[G] * light_list[i].diff[G];
				color[B] += diff_factor * inter->surf->diff[B] * light_list[i].diff[B];
			}
			//////////////////////////////////////////////////
			else
			{
				diff_factor = 0;
			}
			
			//specular

			Vect N, Re, temp, V, H;
			VectdSub(ray->orig, inter->P, V);
			VectdUnit(V);
			VectdAddS(1, V, L, H);
			VectdUnit(H);
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
			double spec_factor = pow(VectdDotProd(inter->N, H), inter->surf->spec_exp);
			
			color[R] += spec_factor * light_list[i].spec[R] * inter->surf->spec[R];
			color[G] += spec_factor * light_list[i].spec[G] * inter->surf->spec[G];
			color[B] += spec_factor * light_list[i].spec[B] * inter->surf->spec[B];

			

			//VectClamp(color, 0, 1);
		}

		//VectClamp(color, 0, 1);
	}
	VectdClamp(color, 0, 1);


}



__device__ void reflection_direction(Vect incoming, Vect norm, Vect outgoing)
{
	VectdAddS(-2 * VectdDotProd(incoming, norm), norm, incoming, outgoing);


}
__device__  Ray *make_ray()
{
	Ray *r;

	r = (Ray *)malloc(sizeof(Ray));

	return r;
}
__device__  Ray *make_ray(Vect orig, Vect dir)
{
	Ray *r;

	r = make_ray();
	VectdCopy(r->orig, orig);
	VectdCopy(r->dir, dir);

	return r;
}


__device__ Intersection *intersect_ray_sphere(Ray *ray, Sphere *S)
{
	// FILL IN CODE (line below says "no" for all spheres, so replace it)
	Intersection *interpoint;
	Vect C;
	double V, BB, delta, t;
	VectdSub(S->P, ray->orig, C);
	V = VectdDotProd(ray->dir, C);
	BB = VectdDotProd(C, C) - V * V;
	if (S->radius * S->radius - BB< 0)
		return NULL;
	else
	{
		delta = S->radius * S->radius - BB;
		t = V - sqrt(delta);
	}
	//interpoint = make_intersection();

	interpoint = new Intersection;


	interpoint->medium = NULL;

	interpoint->t = t;
	interpoint->surf = S->surf;
	VectdAddS(interpoint->t, ray->dir, ray->orig, interpoint->P);
	VectdSub(interpoint->P, S->P, interpoint->N);
	//interpoint->N[0] = interpoint->N[0] / VectDotProd(interpoint->N, interpoint->N);
	//interpoint->N[1] = interpoint->N[1] / VectDotProd(interpoint->N, interpoint->N);
	//interpoint->N[2] = interpoint->N[2] / VectDotProd(interpoint->N, interpoint->N);
	VectdUnit(interpoint->N);

	return interpoint;
	
	return NULL;
}


__device__  void update_nearest_intersection(Intersection **inter, Intersection **nearest_inter)
{
	// only do something if this was a hit

	if (*inter) {

		// this is the first object hit 

		if (!*nearest_inter)
			*nearest_inter = *inter;

		// this is closer than any previous hit

		else if ((*inter)->t < (*nearest_inter)->t) {
			delete *nearest_inter;

				//free(*nearest_inter);
				*nearest_inter = *inter;
		}

		// something else is closer--move along

		else
			delete *inter;
			//free(*inter);
	}
}

#define SMALL_NUM  0.00000001 // anything that avoids division overflow



__device__ Intersection *make_intersection()
{
	Intersection *inter;
	inter = new Intersection;
	//inter = (Intersection *)malloc(sizeof(Intersection));
	//inter->p = NULL;
	
	//cudaMalloc((void**)&inter, sizeof(Intersection));
	inter->medium = NULL;

	return inter;
}

__device__ void draw_point(int x, int y, Vect color, Image *im)
{
	//      printf("%lf\n", color[0]);

	im->data[4 * (x + im->w * y)] = color[R];
	im->data[1 + 4 * (x + im->w * y)] = color[G];
	im->data[2 + 4 * (x + im->w * y)] = color[B];
	im->data[3 + 4 * (x + im->w * y)] = color[A];
}

__device__ void set_pixel_ray_direction(double x, double y, Camera *cam, Ray *ray)
{
	// convert i, j to x frac. and y frac. (where 0, 0 is straight ahead)

	double u = x / (double)cam->im->w;
	double v = y / (double)cam->im->h;

	   ray->orig[X] = cam->eye[X];
	   ray->orig[Y] = cam->eye[Y];
	   ray->orig[Z] = cam->eye[Z];

	//ray->orig[X] = ray->orig[Y] = ray->orig[Z] = 0.0;

	ray->dir[X] = cam->clip[LEFT] + u * (cam->clip[RIGHT] - cam->clip[LEFT]);
	ray->dir[Y] = cam->clip[TOP] + v * (cam->clip[BOTTOM] - cam->clip[TOP]);
	ray->dir[Z] = -cam->clip[NEAR];

	//  printf("%lf %lf -> %lf %lf %lf\n", x, y, ray->dir[X], ray->dir[Y], ray->dir[Z]);

	VectdUnit(ray->dir);

	//  printf("unit %lf %lf -> %lf %lf %lf\n\n", x, y, ray->dir[X], ray->dir[Y], ray->dir[Z]);

}



__host__ Camera *make_camera()
{
	Camera *c;

	c = (Camera *)malloc(sizeof(Camera));

	return c;
}


__host__ void glm_transform(Transform M, GLMmodel *model)
{
	Vect V, V_prime;

	// directly iterate over vertices -- indices seems to start at 1

	for (int i = 1; i <= model->numvertices; i++) {

		V[X] = model->vertices[3 * i];
		V[Y] = model->vertices[3 * i + 1];
		V[Z] = model->vertices[3 * i + 2];
		V[W] = 1.0;

		TransformVect(M, V, V_prime);

		model->vertices[3 * i] = V_prime[X];
		model->vertices[3 * i + 1] = V_prime[Y];
		model->vertices[3 * i + 2] = V_prime[Z];

	}
}

__host__ void setup_lookat_transform(Transform M, Vect eye, Vect center, Vect up)
{
	Vect n, u, v, t;
	int i;

	// rotational components

	VectSub(eye, center, n);

	// u = up x n

	VectCross(up, n, u);

	// v = n x u

	VectCross(n, u, v);

	// normalize n, u, v

	VectUnit(n);
	VectUnit(u);
	VectUnit(v);

	// translation

	t[X] = -VectDotProd(eye, u);
	t[Y] = -VectDotProd(eye, v);
	t[Z] = -VectDotProd(eye, n);

	// put it together

	for (i = 0; i < 3; i++)
		MATRC(M, 0, i) = u[i];

	for (i = 0; i < 3; i++)
		MATRC(M, 1, i) = v[i];

	for (i = 0; i < 3; i++)
		MATRC(M, 2, i) = n[i];

	for (i = 0; i < 3; i++)
		MATRC(M, 3, i) = 0;

	for (i = 0; i < 3; i++)
		MATRC(M, i, 3) = t[i];

	MATRC(M, 3, 3) = 1;

	//   printf("lookat M = \n");
	//   TransformPrint(M);
}


__host__ void parse_scene_file(char *filename, Camera *cam)
{
	char c, obj_filename[64];
	int w, h;
	FILE *fp;
	float sx, sy, sz, rx, ry, rz, dx, dy, dz;   // scale factors, rotation angles (degrees), translation factors
	Transform Tmat, Rmat, Smat, scratchmat, Mmat;
	Vect P;
	Light *L;
	Surface *S;
	Sphere *Sph;

	// read file

	fp = fopen(filename, "r");

	// camera position

	fscanf(fp, "camera %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&cam->eye[X], &cam->eye[Y], &cam->eye[Z],
		&cam->center[X], &cam->center[Y], &cam->center[Z],
		&cam->up[X], &cam->up[Y], &cam->up[Z]);

	//added
	cam->eye[W] = 1.0f;
	cam->center[W] = 0.0f;
	cam->up[W] = 0.0f;


	setup_lookat_transform(cam->W2C, cam->eye, cam->center, cam->up);

	fscanf(fp, "clip %lf %lf %lf %lf %lf %lf\n",
		&cam->clip[LEFT], &cam->clip[RIGHT],
		&cam->clip[BOTTOM], &cam->clip[TOP],
		&cam->clip[NEAR], &cam->clip[FAR]);

	// image dimensions

	fscanf(fp, "image %i %i\n", &w, &h);
	cam->im = make_image(w, h);

	// objects and lights

	model_list.clear();
	model_surf_list.clear();
	sphere_list.clear();
	light_list.clear();

	while (1) {

		c = fgetc(fp);

		// end of file

		if (c == EOF)
			return;

		// it's a comment

		if (c == '#') {
			do { c = fgetc(fp); printf("eating %c\n", c); } while (c != '\n' && c != EOF);
			if (c == EOF)
				return;
			printf("done\n");
		}

		// it's an object
		else if (c == 'o') {

			S = (Surface *)malloc(sizeof(Surface));

			fscanf(fp, "bj %s  %f %f %f  %f %f %f  %f %f %f  %lf %lf %lf  %lf %lf %lf  %lf %lf %lf  %lf %lf  %lf %lf\n",
				obj_filename,
				&dx, &dy, &dz,
				&sx, &sy, &sz,
				&rx, &ry, &rz,
				&(S->amb[R]), &(S->amb[G]), &(S->amb[B]),
				&(S->diff[R]), &(S->diff[G]), &(S->diff[B]),
				&S->spec[R], &S->spec[G], &S->spec[B],
				&S->spec_exp, &S->ior,
				&S->reflectivity, &S->transparency);

			GLMmodel *model = glmReadOBJ(obj_filename);

			if (!model) {
				printf("no such glm model %s\n", obj_filename);
				exit(1);
			}
			// scale and center model on origin

			glmUnitize(model);

			// build model transformations (including lookat world -> camera transform)

			TransformIdentity(Tmat);
			MATRC(Tmat, 0, 3) = dx;
			MATRC(Tmat, 1, 3) = dy;
			MATRC(Tmat, 2, 3) = dz;

			TransformIdentity(Smat);
			MATRC(Smat, 0, 0) = sx;
			MATRC(Smat, 1, 1) = sy;
			MATRC(Smat, 2, 2) = sz;


			TransformProd(Tmat, Smat, scratchmat);
			TransformProd(cam->W2C, scratchmat, Mmat);
			//TransformPrint(Mmat);

			// apply transform to model

			glm_transform(Mmat, model);

			// calculate normals

			glmFacetNormals(model);
			glmVertexNormals(model, 90.0);

			model->index = model_list.size();
			model_list.push_back(model);

			model_surf_list.push_back(S);
		}
		// it's a sphere

		else if (c == 's') {

			S = (Surface *)malloc(sizeof(Surface));
			Sph = (Sphere *)malloc(sizeof(Sphere));

			fscanf(fp, "phere %f %f %f  %f  %lf %lf %lf  %lf %lf %lf  %lf %lf %lf  %lf %lf  %lf %lf\n",
				&dx, &dy, &dz,    // position
				&sx,              // size
				&(S->amb[R]), &(S->amb[G]), &(S->amb[B]),
				&(S->diff[R]), &(S->diff[G]), &(S->diff[B]),
				&S->spec[R], &S->spec[G], &S->spec[B],
				&S->spec_exp, &S->ior,
				&S->reflectivity, &S->transparency);

			Sph->P[X] = dx;
			Sph->P[Y] = dy;
			Sph->P[Z] = dz;
			Sph->P[W] = 1.0;
			TransformVect(cam->W2C, Sph->P, Sph->P);

			Sph->radius = sx;

			Sph->surf = S;

			sphere_list.push_back(Sph);
		}
		else if (c == 'l') {

			L = (Light *)malloc(sizeof(Light));

			fscanf(fp, "ight %lf %lf %lf  %lf %lf %lf  %lf %lf %lf  %lf %lf %lf\n",
				&P[X], &P[Y], &P[Z],
				&L->amb[R], &L->amb[G], &L->amb[B],
				&L->diff[R], &L->diff[G], &L->diff[B],
				&L->spec[R], &L->spec[G], &L->spec[B]);

			// move to camera coordinates
			P[W] = 1.0;
			TransformVect(cam->W2C, P, L->P);

			// add to list

			light_list.push_back(L);
		}

		else if (c == '\n' && c != ' ' && c != '\t') {
			printf("bad scene syntax %c\n", c);
			exit(1);
		}
	}


}

int my_round(double x)
{
	return (int)floor(0.5 + x);
}

void write_PPM(char *filename, Image *im)
{
	int i, j, t;
	FILE *fp;

	// write size info

	fp = fopen(filename, "wb");

	fprintf(fp, "P6\n%i %i\n255\n", im->w, im->h);

	// write binary image data
	//#pragma omp parallel for
	for (j = 0, t = 0; j < im->h; j++)
		//#pragma omp parallel for  
		for (i = 0; i < im->w; i++, t += 4)
			fprintf(fp, "%c%c%c",
			(int)my_round(255 * im->data[t]),
				(int)my_round(255 * im->data[t + 1]),
				(int)my_round(255 * im->data[t + 2]));

	// finish up

	fclose(fp);
}
