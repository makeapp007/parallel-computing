//----------------------------------------------------------------------------
// UD Ray v2.0
// copyright 2008, University of Delaware
// Christopher Rasmussen
//----------------------------------------------------------------------------

#ifndef RAY_DECS
#define RAY_DECS

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include <vector>

using namespace std;

#include "glut.h"

#include "glm.h"

//----------------------------------------------------------------------------
// Constants and macros
//----------------------------------------------------------------------------

#define SQUARE(x)            ((x) * (x))

// row-major
//#define MATRC(mat, r, c)     (mat[(c) + 4 * (r)])

// column-major
#define MATRC(mat, r, c)     (mat[(r) + 4 * (c)])

#define X 0
#define Y 1
#define Z 2
#define W 3

#define R 0
#define G 1
#define B 2
#define A 3

#define LEFT   0
#define RIGHT  1
#define BOTTOM 2
#define TOP    3
#define NEAR   4
#define FAR    5

//----------------------------------------------------------------------------
// Type and struct definitions
//----------------------------------------------------------------------------

typedef double Vect[4];
typedef double Transform[16];

typedef struct imagestruct 
{
	int w, h;                 // width, height of image
	GLfloat *data;            // image data

} Image;

//--------------------------------------

typedef struct camerastruct 
{ 
	double clip[6];       // clip planes (in camera coords; NEAR, FAR both distances -> positive)

	Image *im;

	// position, orientation

	Vect eye;             // camera location in world coordinates
	Vect center;          // "look-at" point in world coordinates
	Vect up;              // camera up vector in world coordinates

	// transformations

	Transform W2C;        // world to camera coordinates transformation

} Camera;

//--------------------------------------

typedef struct raystruct 
{
	Vect orig;            // origin
	Vect dir;             // direction

} Ray;

//--------------------------------------

typedef struct surfstruct 
{
	Vect amb;             // ambient reflectance 
	Vect diff;            // diffuse reflectance 
	Vect spec;            // spec reflectance 
	double spec_exp;      // specular exponent

	double ior;           // index of refraction

	double reflectivity;  // specular reflection coefficient
	double transparency;  // transmission coefficient

} Surface;

//--------------------------------------

typedef struct lightstruct 
{
	Vect P;               // position

	Vect amb;             // ambient intensity
	Vect diff;            // diffuse intensity
	Vect spec;            // specular intensity
	double spec_exp;      // specular exponent

} Light;

//--------------------------------------

typedef struct spherestruct 
{
	Vect P;                 // position
	double radius;          // radius

	Surface *surf;          // material info

} Sphere;

//--------------------------------------

typedef struct intersectionstruct 
{
	double t;               // parameter of ray at point of intersection
	Vect P;                 // location of hitpoint
	Vect N;                 // normal vector at hitpoint
	Surface *surf;          // material properties of object hit

	// need this for refraction

	bool entering;          // true or false
	Surface *medium;        // material of primitive we're in when < t

} Intersection;

//----------------------------------------------------------------------------
// Function declarations
//----------------------------------------------------------------------------

// parsing and initialization, memory handling

void parse_scene_file(char *, Camera *);

Surface *make_surface();
Camera *make_camera();
Image *make_image(int, int);
void init_raytracing();
Ray *make_ray();
Ray *make_ray(Vect, Vect);
void free_intersection(Intersection *);

//----------------------------------------------------------------------------

// transformations

void setup_lookat_transform(Transform, Vect, Vect, Vect);

//----------------------------------------------------------------------------

// ray tracing details

void raytrace_one_pixel(int, int);
void set_pixel_ray_direction(double, double, Camera *, Ray *);
void trace_ray(int, double, Ray *, Vect);
Intersection *make_intersection();
void reflection_direction(Vect, Vect, Vect);
Intersection *intersect_ray_glm_object(Ray *, GLMmodel *);
Intersection *intersect_ray_triangle(Ray *, Vect, Vect, Vect);
Intersection *intersect_ray_sphere(Ray *, Sphere *);
void update_nearest_intersection(Intersection **, Intersection **);

void shade_ray_false_color_normal(Intersection *, Vect);
void shade_ray_background(Ray *, Vect);
void shade_ray_intersection_mask(Vect);
void shade_ray_diffuse(Ray *, Intersection *, Vect);
void shade_ray_local(Ray *, Intersection *, Vect);
void shade_ray_recursive(int, double, Ray *, Intersection *, Vect);

void VectAddS(double, Vect, Vect, Vect);
double VectMag(Vect);
double VectUnit(Vect);
double VectDotProd(Vect, Vect);
void VectCopy(Vect, Vect);
void VectSub(Vect, Vect, Vect);
void VectPrint(Vect);
void VectCross(Vect, Vect, Vect);
void VectNegate(Vect, Vect);
void VectClamp(Vect, double, double);
void Vectmul(double k, Vect v1, Vect v2);
//----------------------------------------------------------------------------

// drawing, opengl

void draw_point(int, int, Vect, Image *);
void write_PPM(char *, Image *);

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
