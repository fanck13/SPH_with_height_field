#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <iostream>
#include <fstream>
#include <cassert>
using namespace std;

#include <gl\freeglut.h>

#define MYFILE    0

struct Node
{
	float3 loc;
	float value;
};

class Surface
{

private:
	float3 *data;
	float *density;
	unsigned int count;
	float3 lb;
	float3 rt;
	int3 dim;
	Node *surface;
	float3 dimsize;
	float *plain;
	float threshold;

	bool sfinit;
	bool plinit;
	bool isconstructed;

#if MYFILE==1 

	ofstream fout;

#endif
	

public:
	Surface(float3* _data, float *_density, unsigned int _count, float3 lb, float3 rt, int3 _dim, float _threshold);
	~Surface();

	void ConstructSurface();

	void WriteToFile(ofstream& out);
	void DrawSurface();
	void DrawParticle();
};

