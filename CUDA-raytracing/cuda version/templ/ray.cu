#include <stdio.h>
#include <stdlib.h>
//cuda////////////////////////////////////

//cuda////////////////////////////////////
#include <Windows.h>
//#include <omp.h>
#include "udray.cuh"
#include <iostream>



using namespace std;
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//extern int maxlevel;          // maximum depth of ray recursion 
//extern double minweight;      // minimum fractional contribution to color

extern Camera *ray_cam;       // camera info


extern int image_i, image_j;  // current pixel being shaded
extern bool wrote_image;      // has the last pixel been shaded?

							  // reflection/refraction recursion control

							  // these describe the scene

extern vector < GLMmodel * > model_list;
extern vector < Sphere * > sphere_list;
extern vector < Light * > light_list;
extern vector < Surface * > model_surf_list;






