/*this file */
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Paras.cuh"
#include "math_functions.h"


#define PI 3.141592657f

#define SQR(x)					((x) * (x))
#define CUBE(x)					((x) * (x) * (x))
#define POW6(x)					(CUBE(x) * CUBE(x))
#define POW9(x)					(POW6(x) * CUBE(x))


__constant__ Paras para;

inline __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ void operator+=(float3& a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __device__ float operator*(float3 a, float3 b)
{
	return (a.x*b.x + a.y*b.y + a.z*b.z);
}

inline __device__ float3 operator*(float a, float3 b)
{
	return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __device__ float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ float kernel(float3 r, float h)
{
	return 315.0f / (64.0f * PI * POW9(h)) * CUBE(SQR(h) - r*r);
}

__device__ float3 kernel_grident(float3 r, float h)
{
	return -945.0f / (32.0f * PI * POW9(h)) * SQR(SQR(h) - r*r) * r;
}


__device__ float laplacian_kernel(float3 r, float h)
{
	return 45.0f / (PI * POW6(h)) * (h - sqrtf(r.x*r.x + r.y*r.y + r.z*r.z));
}

__device__ int3 cudaCalcGridPos(float3 pos)
{
	int3 gridPos;
	gridPos.x = floor(pos.x / 1.0f);
	gridPos.y = floor(pos.y / 1.0f);
	gridPos.z = floor(pos.z / 1.0f);

	return gridPos;
}

__device__ int cudaCalcGridHash(int3 gridPos)
{
	return (gridPos.z * 50 + gridPos.y * 50*40 + gridPos.x);
}

__global__ void cudaCalHash(unsigned int* index, unsigned int* hash, float3* pos, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x; 
	if (tid >= count)
	{
		return;
	}

	int3 gridPos = cudaCalcGridPos(pos[tid]);
	int ash = cudaCalcGridHash(gridPos);

	hash[tid] = ash;
	index[tid] = tid;
}

__global__ void cudaReorderDataAndFindCellStart(unsigned int *cellstart,
	unsigned int *cellend,
												float3* spos,
												float3* svel, 
												unsigned int* hash,
												unsigned int* index,
												float3* pos, 
												float3* vel, 
												unsigned int count)
{
	extern __shared__ int sharedHash[];
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int _hash;
	if (tid < count)
	{
		_hash = hash[tid];

		sharedHash[threadIdx.x + 1] = _hash;
		if (tid > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = hash[tid - 1];
		}

	}
	__syncthreads();

	if (tid < count)
	{
		if (tid == 0 || _hash != sharedHash[threadIdx.x])
		{
			cellstart[_hash] = tid;
			if (tid > 0)
			{
				cellend[sharedHash[threadIdx.x]] = tid;
			}
		}
		if (tid == (count - 1))
		{
			cellend[_hash] = tid + 1;
		}

		int sortedIndex = index[tid];
		float3 _pos = pos[sortedIndex];
		float3 _vel = vel[sortedIndex];

		spos[tid] = _pos;
		svel[tid] = _vel;
	}
}

__device__ float length(float3 a)
{
	return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}


__device__ float cudaAddDensity(unsigned int tid, int3 gridPos, float3 pos, unsigned int* cellstart, unsigned int* cellend, float3* spos)
{

	float _dens = 0.0f;
	unsigned int _hash = cudaCalcGridHash(gridPos);
	unsigned int startIndex = cellstart[_hash];

	if (startIndex != 0xffffffff)
	{
		unsigned int endIndex = cellend[_hash];
		for (int j = startIndex; j != endIndex; j++)
		{
			float3 _pos = spos[j];
			float3 deltapos = pos - _pos;
			if (length(deltapos) <= para.h)
			{
				_dens += para.mass*kernel(deltapos, para.h);
			}
		}
	}

	return _dens;
}


__global__ void cudaCalcDensity(float* dens, unsigned int* cellstart, unsigned int* cellend, float3* spos, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= count)
	{
		return;
	}
	float3 _pos = spos[tid];

	float _dens = 0.0f;
	int3 gridPos = cudaCalcGridPos(_pos);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbour = gridPos + make_int3(x, y, z);

				if (neighbour.x >= 0 && neighbour.y >= 0 && neighbour.z >= 0)
				{
					_dens += cudaAddDensity(tid, neighbour, _pos, cellstart, cellend, spos);
				}
			}
		}
	}

	dens[tid] = _dens;
}

__global__ void cudaCalcPressure(float* press, float* dens, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= count)
	{
		return;
	}

	press[tid] = para.k*(dens[tid] - para.restDens);
}

__device__ float3 cudaAddForce(unsigned int tid, int3 gridPos, float3 pos, float3 vel, 
	float press, float* pre, unsigned int* cellstart, unsigned int* cellend, float3* spos, float3* svel, float* dens)
{
	float3 force = make_float3(0.0f, 0.0f, 0.0f);
	unsigned int _hash = cudaCalcGridHash(gridPos);
	unsigned int startIndex = cellstart[_hash];

	if (startIndex != 0xffffffff)
	{
		unsigned int endIndex = cellend[_hash];
		for (int j = startIndex; j != endIndex; j++)
		{
			float3 _pos = spos[j];
			float3 dis = pos - _pos;
			if (length(dis) <= para.h)
			{

				float3 _vel = svel[j];
				float _dens = dens[j];
				float _press = pre[j];
				float3 deltavel = _vel - vel;


				force += para.mu*para.mass*deltavel / _dens*laplacian_kernel(dis, para.h)*0.5;
				force += -para.mass*(press + _press) / (2.0f * _dens)*kernel_grident(dis, para.h)*0.2f;
			}
		}
	}

	return force;
}
__global__ void cudaCalcForce(float3* force, float3* spos, float3* svel, float3* vel, float* press,
	float* dens, unsigned int* index, unsigned int* cellstart, unsigned int* cellend, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= count)
	{
		return;
	}

	float3 _pos = spos[tid];
	float3 _vel = svel[tid];
	//float _dens = dens[tid];
	float _press = press[tid];

	int3 gridPos = cudaCalcGridPos(_pos);

	float3 _force = make_float3(0.0f, 0.0f, 0.0f);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbour = gridPos + make_int3(x, y, z);
				if (neighbour.x >= 0 && neighbour.y >= 0 && neighbour.z >= 0)
				{
					_force += cudaAddForce(tid, neighbour, _pos, _vel, _press, press, cellstart, cellend, spos, svel, dens);
				}
			}
		}
	}

	force[tid] = _force-make_float3(0.0f, 1000.0f, 0.0f);

	unsigned int originalIndex = index[tid];
	vel[originalIndex] = make_float3(_vel.x + _force.x, _vel.y + _force.y, _vel.z + _force.z);
}

__global__ void cudaUpdateVelocityAndPosition(float3* pos, float3* vel, float3* force, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= count)
	{
		return;
	}

	vel[tid] += force[tid] * para.dt;
	pos[tid] += vel[tid] * para.dt;
}

__global__ void cudaHandleBoundary(float3* pos, float3* vel, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= count)
	{
		return;
	}

	float x = pos[tid].x;
	float y = pos[tid].y;
	float z = pos[tid].z;

	if (x > para.xmax - 0.5f)
	{
		pos[tid].x =  para.xmax - 0.5f;
		vel[tid].x = -vel[tid].x;
	}

	if (x < para.xmin + 0.5f)
	{
		pos[tid].x = para.xmin + 0.5f ;
		vel[tid].x = -vel[tid].x;
	}

	if (y > para.ymax - 0.5f)
	{
		pos[tid].y = (para.ymax - 0.5f);
		vel[tid].y = -vel[tid].y;
	}

	if (y < para.ymin + 0.5f)
	{
		pos[tid].y = para.ymin + 0.5f;
		vel[tid].y = -vel[tid].y;
	}

	if (z > para.zmax - 0.5f)
	{
		pos[tid].z =  (para.zmax - 0.5f);
		vel[tid].z = -vel[tid].z;
	}

	if (z < para.zmin + 0.5f)
	{
		pos[tid].z = para.zmin + 0.5f ;
		vel[tid].z = -vel[tid].z;
	}
}

__global__ void cudaCalcR1R2H(float* r1, float* r2, float* len, float rad, float maxvel, float3* vel, int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid > count)
	{
		return;
	}
	float length = 0.0f;
	float _r1 = 0.0f;
	float _r2 = 0.0f;
	float3 velocity = vel[tid];
	float normalvel = sqrt(velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z);
	float ratio_of_vel_maxvel = normalvel / maxvel;

	length = rad*ratio_of_vel_maxvel * 3;

	float ratio_of_r1_r2 = 128.0f*CUBE(ratio_of_vel_maxvel - SQR(ratio_of_vel_maxvel)) + 1.0f;

	float index = 1.0f / 3.0f;
	float Q = 3.0f*ratio_of_vel_maxvel;
	float denominator = 2.0f*CUBE(ratio_of_r1_r2) + 2.0f + Q*ratio_of_r1_r2 + SQR(ratio_of_r1_r2)*Q + Q;

	float base = 4.0f*CUBE(rad) / denominator;
	_r2 = powf(base, index);
	_r1 = ratio_of_r1_r2*_r2;

	len[tid] = length;
	r1[tid] = _r1;
	r2[tid] = _r2;
}
