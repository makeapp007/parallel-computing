//----------------------------------------------------------------------------
// UD Ray v2.0
// copyright 2008, University of Delaware
// Christopher Rasmussen
//----------------------------------------------------------------------------
#include <omp.h>
#include "udray.h"

//#include <vector>

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------

vector < GLMmodel * > model_list;
vector < Surface * > model_surf_list;   // material properties associated with .obj models (to avoid changing glm.hh)
vector < Sphere * > sphere_list;    // material properties are linked inside each sphere struct
vector < Light * > light_list;

Camera *ray_cam;

Ray *eye_ray;
int image_i, image_j;
bool wrote_image;

int maxlevel;          // maximum depth of ray recursion
double minweight;      // minimum fractional contribution to color
double rayeps;         // round-off error tolerance     

					   //----------------------------------------------------------------------------
					   // Functions
					   //----------------------------------------------------------------------------

					   // cross product v3 = v1 x v2

void VectCross(Vect v1, Vect v2, Vect v3)
{
	v3[X] = v1[Y] * v2[Z] - v1[Z] * v2[Y];
	v3[Y] = v1[Z] * v2[X] - v1[X] * v2[Z];
	v3[Z] = v1[X] * v2[Y] - v1[Y] * v2[X];

}

//----------------------------------------------------------------------------

void VectPrint(Vect v)
{
	printf("%.2lf %.2lf %.2lf\n", v[X], v[Y], v[Z]);
}

//----------------------------------------------------------------------------

// dst = src

void VectCopy(Vect dst, Vect src)
{
	dst[X] = src[X];
	dst[Y] = src[Y];
	dst[Z] = src[Z];
}

//----------------------------------------------------------------------------

// scaled addition v3 = t * v1 + v2

void VectAddS(double t, Vect v1, Vect v2, Vect v3)
{
	v3[X] = t * v1[X] + v2[X];
	v3[Y] = t * v1[Y] + v2[Y];
	v3[Z] = t * v1[Z] + v2[Z];
}

//----------------------------------------------------------------------------

// v3 = v1 - v2

void VectSub(Vect v1, Vect v2, Vect v3)
{
	v3[X] = v1[X] - v2[X];
	v3[Y] = v1[Y] - v2[Y];
	v3[Z] = v1[Z] - v2[Z];
}

//----------------------------------------------------------------------------

// vector length

double VectMag(Vect v)
{
	return sqrt(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}

//----------------------------------------------------------------------------
void Vectmul(double k, Vect v1, Vect v2)
{
	v2[X] = v1[X] * k;
	v2[Y] = v1[Y] * k;
	v2[Z] = v1[Z] * k;
}
// make vector have unit length; return original length

double VectUnit(Vect v)
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

void VectNegate(Vect v, Vect vneg)
{
	vneg[X] = -v[X];
	vneg[Y] = -v[Y];
	vneg[Z] = -v[Z];
}

//----------------------------------------------------------------------------

void VectClamp(Vect v, double low, double high)
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

double VectDotProd(Vect v1, Vect v2)
{
	return v1[X] * v2[X] + v1[Y] * v2[Y] + v1[Z] * v2[Z];
}


//----------------------------------------------------------------------------

// multiply vector by matrix transform

void TransformVect(Transform M, Vect V, Vect V_prime)
{
	V_prime[X] = M[0] * V[X] + M[4] * V[Y] + M[8] * V[Z] + M[12] * V[W];
	V_prime[Y] = M[1] * V[X] + M[5] * V[Y] + M[9] * V[Z] + M[13] * V[W];
	V_prime[Z] = M[2] * V[X] + M[6] * V[Y] + M[10] * V[Z] + M[14] * V[W];
	V_prime[W] = M[3] * V[X] + M[7] * V[Y] + M[11] * V[Z] + M[15] * V[W];
}

//----------------------------------------------------------------------------

void TransformPrint(Transform M)
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

void TransformIdentity(Transform M)
{
	int i, r;

	for (i = 0; i < 16; i++)
		M[i] = 0.0;

	for (r = 0; r < 4; r++)
		MATRC(M, r, r) = 1.0;
}

//----------------------------------------------------------------------------

// M3 = M1 * M2

void TransformProd(Transform M1, Transform M2, Transform M3)
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

void init_raytracing()
{
	ray_cam = make_camera();

	maxlevel = 2;
	minweight = 0.01;
	rayeps = 1e-7;

	eye_ray = make_ray();

	image_i = 0;
	image_j = 0;

	wrote_image = false;
}

//----------------------------------------------------------------------------

// given a pixel location, turn it into an eye ray and trace it to get the color

void raytrace_one_pixel(int i, int j)
{
	double x, y;
	Vect eye_color;
	Ray* eye_ray = new Ray;//could be wrong
	x = 0.5 + (double)i;   // the "center" of the pixel
	y = 0.5 + (double)j;

	set_pixel_ray_direction(x, y, ray_cam, eye_ray);

	eye_color[R] = eye_color[G] = eye_color[B] = 0.0;
	trace_ray(0, 1.0, eye_ray, eye_color);
	
	draw_point(i, j, eye_color, ray_cam->im);
}

//----------------------------------------------------------------------------

// figure out the parametric 3-D line in camera coordinates that corresponds to 
// pixel coordinates (x, y), where (0, 0) is the upper-left hand corner of the image

// ray direction should be a unit vector

void set_pixel_ray_direction(double x, double y, Camera *cam, Ray *ray)
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

	VectUnit(ray->dir);

	//  printf("unit %lf %lf -> %lf %lf %lf\n\n", x, y, ray->dir[X], ray->dir[Y], ray->dir[Z]);

}

//----------------------------------------------------------------------------

// inter = current intersection (possibly NULL)
// nearest_inter = nearest intersection so far (also possibly NULL)

void update_nearest_intersection(Intersection **inter, Intersection **nearest_inter)
{
	// only do something if this was a hit

	if (*inter) {

		// this is the first object hit 

		if (!*nearest_inter)
			*nearest_inter = *inter;

		// this is closer than any previous hit

		else if ((*inter)->t < (*nearest_inter)->t) {
			free(*nearest_inter);
			*nearest_inter = *inter;
		}

		// something else is closer--move along

		else
			free(*inter);
	}
}

//----------------------------------------------------------------------------

// intersect a 3-D ray with a 3D triangle

// from http://www.softsurfer.com/Archive/algorithm_0105/algorithm_0105.htm

// Copyright 2001, softSurfer (www.softsurfer.com)
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.

#define SMALL_NUM  0.00000001 // anything that avoids division overflow

//    Input:  a ray R, and a triangle T
//    Return: intersection information (when it exists); NULL otherwise

Intersection *intersect_ray_triangle(Ray *ray, Vect V0, Vect V1, Vect V2)
{
	Vect    u, v, n;        // triangle vectors
	Vect    w0, w;          // ray vectors
	float   a, b;           // params to calc ray-plane intersect
	float t;
	Vect I;
	Intersection *inter;

	// get triangle edge vectors and plane normal

	VectSub(V1, V0, u);
	VectSub(V2, V0, v);
	VectCross(u, v, n);
	if (n[X] == 0 && n[Y] == 0 && n[Z] == 0)            // triangle is degenerate; do not deal with this case
		return NULL;

	VectSub(ray->orig, V0, w0);
	a = -VectDotProd(n, w0);
	b = VectDotProd(n, ray->dir);

	if (fabs(b) < SMALL_NUM) {     // ray is parallel to triangle plane
		if (a == 0)                  // case 1: ray lies in triangle plane
			return NULL;
		else return NULL;               // case 2: ray disjoint from plane
	}

	// get intersect point of ray with triangle plane

	t = a / b;
	if (t < rayeps)                   // triangle is behind/too close to ray => no intersect
		return NULL;                 // for a segment, also test if (t > 1.0) => no intersect

									 // intersect point of ray and plane

	VectAddS(t, ray->dir, ray->orig, I);

	// is I inside T?

	float    uu, uv, vv, wu, wv, D;
	uu = VectDotProd(u, u);
	uv = VectDotProd(u, v);
	vv = VectDotProd(v, v);
	VectSub(I, V0, w);
	wu = VectDotProd(w, u);
	wv = VectDotProd(w, v);
	D = uv * uv - uu * vv;

	// get and test parametric (i.e., barycentric) coords

	float p, q;  // were s, t in original code
	p = (uv * wv - vv * wu) / D;
	if (p < 0.0 || p > 1.0)        // I is outside T
		return NULL;
	q = (uv * wu - uu * wv) / D;
	if (q < 0.0 || (p + q) > 1.0)  // I is outside T
		return NULL;

	inter = make_intersection();
	inter->t = t;
	VectCopy(inter->P, I);
	return inter;                      // I is in T
}

//----------------------------------------------------------------------------

// multiply vertices of object by M

void glm_transform(Transform M, GLMmodel *model)
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

//----------------------------------------------------------------------------

// intersect ray with .obj model (a bunch of triangles with precomputed normals)
// if we hit something, set the color and return true

Intersection *intersect_ray_glm_object(Ray *ray, GLMmodel *model)
{
	//static GLMgroup* group;
	GLMgroup* group;
	//static GLMtriangle* triangle;
	GLMtriangle* triangle;
	Vect V0, V1, V2;

	Intersection *nearest_inter = NULL;
	Intersection *inter = NULL;

	// iterate over all groups in the model

	for (group = model->groups; group; group = group->next) {

		// iterate over all triangles in this group

		for (int i = 0; i < group->numtriangles; i++) {

			triangle = &model->triangles[group->triangles[i]];

			// get triangle vertices

			V0[X] = model->vertices[3 * triangle->vindices[0]];
			V0[Y] = model->vertices[3 * triangle->vindices[0] + 1];
			V0[Z] = model->vertices[3 * triangle->vindices[0] + 2];

			V1[X] = model->vertices[3 * triangle->vindices[1]];
			V1[Y] = model->vertices[3 * triangle->vindices[1] + 1];
			V1[Z] = model->vertices[3 * triangle->vindices[1] + 2];

			V2[X] = model->vertices[3 * triangle->vindices[2]];
			V2[Y] = model->vertices[3 * triangle->vindices[2] + 1];
			V2[Z] = model->vertices[3 * triangle->vindices[2] + 2];

			// test for intersection

			inter = intersect_ray_triangle(ray, V0, V1, V2);

			// we have a hit in front of the camera...

			if (inter) {

				// set normal

				inter->N[X] = model->facetnorms[3 * triangle->findex];
				inter->N[Y] = model->facetnorms[3 * triangle->findex + 1];
				inter->N[Z] = model->facetnorms[3 * triangle->findex + 2];

				inter->surf = model_surf_list[model->index];

				// this the first hit 

				if (!nearest_inter) {
					nearest_inter = inter;
				}

				// this is closer than any previous hit

				else if (inter->t < nearest_inter->t) {
					free(nearest_inter);
					nearest_inter = inter;
				}

				// something else is closer--move along

				else {
					free(inter);
				}
			}
		}
	}

	return nearest_inter;
}

//----------------------------------------------------------------------------

void reflection_direction(Vect incoming, Vect norm, Vect outgoing)
{
	VectAddS(-2 * VectDotProd(incoming, norm), norm, incoming, outgoing);
}

//----------------------------------------------------------------------------

// set color based on facet normal of intersection point

void shade_ray_false_color_normal(Intersection *inter, Vect color)
{
	color[R] = fabs(inter->N[X]);
	color[G] = fabs(inter->N[Y]);
	color[B] = fabs(inter->N[Z]);
}

//----------------------------------------------------------------------------

// color to draw if ray hits nothing

// could also do some sphere or cube map here (which is why ray is passed)

void shade_ray_background(Ray *ray, Vect color)
{
	color[R] = color[G] = color[B] = 0.0;
}

//----------------------------------------------------------------------------

// opposite of background--just set a constant color for any
// ray that hits something

void shade_ray_intersection_mask(Vect color)
{
	color[R] = color[G] = color[B] = 1.0;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

Camera *make_camera()
{
	Camera *c;

	c = (Camera *)malloc(sizeof(Camera));

	return c;
}

//----------------------------------------------------------------------------

Image *make_image(int w, int h)
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
			draw_point(i, j, red, im);

	return im;
}

//----------------------------------------------------------------------------

Ray *make_ray()
{
	Ray *r;

	r = (Ray *)malloc(sizeof(Ray));

	return r;
}

//----------------------------------------------------------------------------

Ray *make_ray(Vect orig, Vect dir)
{
	Ray *r;

	r = make_ray();
	VectCopy(r->orig, orig);
	VectCopy(r->dir, dir);

	return r;
}

//----------------------------------------------------------------------------

Intersection *make_intersection()
{
	Intersection *inter;

	inter = (Intersection *)malloc(sizeof(Intersection));
	///inter->p = NULL;
	inter->medium = NULL;

	return inter;
}

//----------------------------------------------------------------------------

void free_intersection(Intersection *inter)
{
	free(inter);
}

//----------------------------------------------------------------------------

// set up rigid transform of world coordinates to camera coordinates

void setup_lookat_transform(Transform M, Vect eye, Vect center, Vect up)
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

//----------------------------------------------------------------------------

void parse_scene_file(char *filename, Camera *cam)
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

	// clip planes

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

			// ROTATION WOULD GO HERE

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

			//view transform
			TransformVect(cam->W2C, Sph->P, Sph->P);

			Sph->radius = sx;

			Sph->surf = S;

			sphere_list.push_back(Sph);
		}

		// it's a light

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

		// unknown

		else if (c == '\n' && c != ' ' && c != '\t') {
			printf("bad scene syntax %c\n", c);
			exit(1);
		}
	}
}

//----------------------------------------------------------------------------

// lower-left corner of image is (0, 0)
// alpha is not used here--you need to change this for blending to work

void draw_point(int x, int y, Vect color, Image *im)
{
	//      printf("%lf\n", color[0]);

	im->data[4 * (x + im->w * y)] = color[R];
	im->data[1 + 4 * (x + im->w * y)] = color[G];
	im->data[2 + 4 * (x + im->w * y)] = color[B];
	im->data[3 + 4 * (x + im->w * y)] = color[A];
}

//----------------------------------------------------------------------------

int my_round(double x)
{
	return (int)floor(0.5 + x);
}

//----------------------------------------------------------------------------

// write binary PPM (expects data to be 4 floats per pixel)

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

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------