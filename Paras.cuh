#include "device_launch_parameters.h"
#include "cuda_runtime.h"

struct Paras
{
	float dt;

	float xmin;
	float xmax;
	float ymin;
	float ymax;
	float zmin;
	float zmax;

	float mass;
	float h;
	float restDens;
	float k;
	float mu;

	int3 gridSize;
};

