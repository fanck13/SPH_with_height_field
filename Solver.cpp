#include "Solver.h"

extern "C"
{
	void HandleError(cudaError_t status, char* message);

	void SetParas(Paras *p);

	void CalHash(unsigned int* index, unsigned int* hash, float3* pos, unsigned int count);

	void SortParticles(unsigned int *hash, unsigned int *index, unsigned int count);

	void ReorderDataAndFindCellStart(unsigned int* cellstart,
		unsigned int* cellend,
		float3* spos,
		float3* svel,
		unsigned int* hash,
		unsigned int* index,
		float3* pos,
		float3* vel,
		unsigned int count,
		unsigned int gridNum);

	void CalcDensity(float* dens, 
		             unsigned int* cellstart, 
					 unsigned int* cellend, 
					 float3 *spos, 
					 unsigned int count);

	void CalcPressure(float* press, 
		              float* dens, 
					  unsigned int count);

	//void CalcForce(float3* force, float3* spos, float3* svel, float* press, float* dens, int* cellstart, int* cellend, int count);
	void CalcForce(float3* force, 
		           float3* spos, 
				   float3* svel, 
				   float3* vel, 
				   float* press, 
				   float* dens, 
				   unsigned int* index, 
				   unsigned int* cellstart, 
				   unsigned int* cellend, 
				   unsigned int count);

	void UpdateVelocityAndPosition(float3* pos, 
		                           float3* vel, 
								   float3* force, 
								   unsigned int count);

	void HandleBoundary(float3* pos, 
		                float3* vel, 
						unsigned int count);

}

#define CHECK(ptr, message)  {if(ptr==nullptr){cerr<<message<<endl;exit(1);}}

#if SOLVER_WRITE_TO_FILE==1
ofstream out("result.txt");
ofstream out1("result1.txt");
#endif

Solver::Solver(unsigned int _count) :count(_count)
{
	size1 = count*sizeof(float);
	size3 = count*sizeof(float3);
	gridNum = 50 * 30 * 40;


	////////set parameters//////////////
	pa.mass = 1.0f;
	pa.dt = 0.001f;

	pa.xmin = 0.0f;
	pa.xmax = 50.0f;
	pa.ymin = 0.0f;
	pa.ymax = 30.0f;
	pa.zmin = 0.0f;
	pa.zmax = 40.0f;

	pa.h = 1.1f;
	pa.k = 1000.0f;
	pa.restDens = 1.2f;
	pa.mu = 0.1f;

	pa.gridSize.x = 50;
	pa.gridSize.y = 30;
	pa.gridSize.z = 40;
	////////allocate memory//////
	
	hpos=(float3*)malloc(size3);
	CHECK(hpos, "Failed to allocate memory of hpos!");

	hvel = (float3*)malloc(size3);
	CHECK(hvel, "Failed to allocate memory of hvel!");

	HandleError(cudaMalloc((void**) &dpos, size3), "Failed to allocate memory of dpos!");

	HandleError(cudaMalloc((void**) &dvel, size3), "Failed to allocate memory of dvel!");

	HandleError(cudaMalloc((void**)&dspos, size3), "Failed to allocate memory of dspos!");

	HandleError(cudaMalloc((void**)&dsvel, size3), "Failed to allocate memory of dsvel!");

	HandleError(cudaMalloc((void**) &ddens, size1), "Failed to allocate memory of ddens!");

	HandleError(cudaMalloc((void**) &dforce, size3), "Failed to allocate memory of dforce!");

	HandleError(cudaMalloc((void**) &dpress, size1), "Failed to allocate memory of dpress!");

	HandleError(cudaMalloc((void**)&dindex, count*sizeof(unsigned int)), "Failed to allocate memory of dindex");
	
	HandleError(cudaMalloc((void**)&dhash, count*sizeof(unsigned int)), "Failed to allocate memory of dhash");

	HandleError(cudaMalloc((void**)&dcellStart, gridNum*sizeof(unsigned int)), "Failed to allocate memory of dcellstart");

	HandleError(cudaMalloc((void**)&dcellEnd, gridNum*sizeof(unsigned int)), "Failed to allocate memory of dcellend");

	///_1
	HandleError(cudaMalloc((void**)&dlen, size1), "Failed to allocate memory of dlen!");
	HandleError(cudaMalloc((void**)&dr1, size1), "Failed to allocate memory of dr1!");
	HandleError(cudaMalloc((void**)&dr2, size1), "Failed to allocate memory of dr2!");

	hlen = (float*)malloc(size1);
	CHECK(hlen, "Failed to allocate memory of hlen");

	hr1 = (float*)malloc(size1);
	CHECK(hr1, "Failed to allocate memory of hr1");

	hr2 = (float*)malloc(size1);
	CHECK(hr2, "Failed to allocate memory of hr2");
	///_1

	temp = (float*)malloc(size1);
	CHECK(temp, "Failed to allocate memory of temp");

	temp3 = (float3*)malloc(size3);
	CHECK(temp3, "Failed to allocate memory of temp3");

	InitParticles();

	HandleError(cudaMemcpy(dpos, hpos, size3, cudaMemcpyHostToDevice), "Failed to copy memory of hpos!");
	HandleError(cudaMemset(dvel, 0, size3), "Failed to memset dvel!");//???//?????????????
	HandleError(cudaMemset(dsvel, 0, size3), "Failed to memset dsvel!"); 
	HandleError(cudaMemset(dspos, 0, size3), "Failed to memset dspos!"); 
	HandleError(cudaMemset(ddens, 0, size1), "Failed to memset ddens!");
	HandleError(cudaMemset(dforce, 0, size3), "Failed to memset dforce!");
	HandleError(cudaMemset(dpress, 0, size1), "Failed to memset dpress!");

	//HandleError(cudaMemset(dindex, 0, size1), "Failed to memset dindex!");
	//HandleError(cudaMemset(dhash, 0, size1), "Failed to memset dhash!");
	//HandleError(cudaMemset(dcellStart, 0, gridNum*sizeof(unsigned int)), "Failed to memset dcellstart!");
	//HandleError(cudaMemset(dcellEnd, 0, gridNum*sizeof(unsigned int)), "Failed to memset dcellend!");
}


Solver::~Solver()
{
	free(hpos);
	free(hvel);
	free(temp);
	free(temp3);

	HandleError(cudaFree(dpos), "Failed to free dpos!");
	HandleError(cudaFree(dvel), "Failed to free dvel!");
	HandleError(cudaFree(ddens), "Failed to free ddens!");
	HandleError(cudaFree(dforce), "Failed to free dforce!");
	HandleError(cudaFree(dpress), "Failed to free dpress!");
	HandleError(cudaFree(dhash), "Failed to free dhash!");
	HandleError(cudaFree(dindex), "Failed to free dindex!");
	HandleError(cudaFree(dcellStart), "Failed to free dcellStart!");
	HandleError(cudaFree(dcellEnd), "Failed to free dcellEnd!");

	HandleError(cudaFree(dspos), "Failed to free dspos!");
	HandleError(cudaFree(dsvel), "Failed to free dsvel!");

	///_1
	free(hlen);
	free(hr1);
	free(hr2);

	HandleError(cudaFree(dlen), "Failed to free dlen!");
	HandleError(cudaFree(dr1), "Failed to free dr1!");
	HandleError(cudaFree(dr2), "Failed to free dr2!");

	///_1
}


void Solver::InitParticles()
{
	int id = 0;
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				id = k * 32 * 32 + j * 32 + i;
				hpos[id].x = i*1.0f+0.5f;
				hpos[id].y = k*1.0f+0.5f;
				hpos[id].z = j*1.0f+0.5f;
			}
		}
	}

	/*for (int i = 0; i < count; i++)
	{
		out1 << hpos[i].x <<", "<< hpos[i].y <<", " <<hpos[i].z << endl;
	}*/
}




void Solver::Update()
{
	SetParas(&pa);

	CalHash(dindex, dhash, dpos, count);

	unsigned int* tmpint = (unsigned int*)malloc(sizeof(unsigned int)*gridNum);

	CHECK(tmpint, "Failed!");

	/*HandleError(cudaMemcpy(tmpint, dindex, sizeof(int)*count, cudaMemcpyDeviceToHost), "Failed to copy device to host!");

	for (int i = 0; i < count; i++)
	{
		out << tmpint[i] << endl;
	}*/

	/*HandleError(cudaMemcpy(tmpint, dhash, sizeof(unsigned int)*count, cudaMemcpyDeviceToHost), "Failed to copy device to host!");

	for (int i = 0; i < count; i++)
	{
		out << tmpint[i] << endl;
	}

	system("pause");*/

	SortParticles(dhash, dindex, count);

	/*HandleError(cudaMemcpy(tmpint, dhash, sizeof(unsigned int)*count, cudaMemcpyDeviceToHost), "Failed to copy device to host!");

	for (int i = 0; i < count; i++)
	{
		out << tmpint[i] << endl;
	}

	system("pause");*/
	
	ReorderDataAndFindCellStart(dcellStart, dcellEnd, dspos, dsvel, dhash, dindex, dpos, dvel, count, gridNum);
	
	CalcDensity(ddens, dcellStart, dcellEnd, dspos, count);

	//system("pause");

	CalcPressure(dpress, ddens, count);


	CalcForce(dforce, dspos, dsvel, dvel, dpress, ddens, dindex, dcellStart, dcellEnd, count);

	UpdateVelocityAndPosition(dpos, dvel, dforce, count);

	HandleError(cudaMemcpy(temp3, dpos, size3, cudaMemcpyDeviceToHost), "Failed to copy device to host!");

	/*out << "///////////////////////////////////////////////////////////////" << endl;
	for (int i = 0; i < count; i++)
	{
		out << temp3[i].x << ", " << temp3[i].y << ", " << temp3[i].z << endl;
	}

	system("pause");*/

	HandleBoundary(dpos, dvel, count);

	HandleError(cudaMemcpy(temp3, dpos, size3, cudaMemcpyDeviceToHost), "Failed to copy device to host in update!");

	HandleError(cudaMemcpy(temp, ddens, size1, cudaMemcpyDeviceToHost), "Failed to copy device to host in update!");
}


float3* Solver::GetPos()
{
	return temp3;
}


float* Solver::GetDensity()
{
	return temp;
}
