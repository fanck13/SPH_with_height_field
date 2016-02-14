#include "Surface.h"


Surface::Surface(float3* _data,
	             float* _density,
	             unsigned int _count,
	             float3 _lb,
	             float3 _rt,
	             int3 _dim,
	             float _threshold)
	             :
	             data(_data),
	             density(_density),
	             count(_count),
	             lb(_lb),
	             rt(_rt),
	             dim(_dim),
	             threshold(_threshold),
				 sfinit(false),
                 plinit(false),
                 isconstructed(false)
{
	surface = new Node[(dim.x + 1)*(dim.y + 1)*(dim.z + 1)];

	cout << (dim.x + 1)*(dim.y + 1)*(dim.z + 1) << endl;
	if (nullptr == surface)
	{
		cerr << "Failed to allocate memory!" << endl;
		exit(0);
	}

	plain = new float[(dim.x + 1)*(dim.z + 1)];
	if (nullptr == surface)
	{
		cerr << "Failed to allocate memory!" << endl;
		exit(0);
	}

	//rt = _rt;
	//lb = _lb;
	//data = _data;
	//density = _density;
	//count = _count;
	//threshold = _threshold;
	//dim = _dim;
	

#if MYFILE==1 
	fout.open("data.txt");
	if (!fout)
	{
		cerr << "Failed to open the file!" << endl;
		exit(1);
	}
#endif

	
	 
	dimsize = make_float3(rt.x - lb.x, rt.y - lb.y, rt.z - lb.z);


	for (int j = 0; j < dim.y + 1; j++)
	{
		for (int k = 0; k < dim.z + 1; k++)
		{
			for (int i = 0; i < dim.x + 1; i++)
			{
				int id = i + k*(dim.x + 1) + j*(dim.x + 1)*(dim.z + 1);

				assert(i < ((dim.x + 1)*(dim.y + 1)*(dim.z + 1)));

				surface[id].loc.x = lb.x + i*dimsize.x / dim.x;
				surface[id].loc.y = lb.y + j*dimsize.y / dim.y;
				surface[id].loc.z = lb.z + k*dimsize.z / dim.z;
				surface[id].value = 0.0f;

				if (0 == j)
				{
					assert((i + k*(dim.x + 1)) < ((dim.x + 1)*(dim.z + 1)));
					plain[i + k*(dim.x + 1)] = 0.0f;
				}

			}

		}
		plinit = true;
	}

	sfinit=true;
#if MYFILE==1 
	for (int i = 0; i < (dim.x + 1)*(dim.y + 1)*(dim.z + 1); i++)
	{
		fout << surface[i].loc.x << "  " << surface[i].loc.y << "  " << surface[i].loc.z << "  " << endl;
	}
#endif

}


Surface::~Surface()
{
	delete surface;
	delete plain;

#if MYFILE==1

	fout.close();

#endif
}

void Surface::ConstructSurface()
{
	assert(sfinit);
	int x = 0;
	int y = 0;
	int z = 0;

	for (int datanum = 0; datanum < count; datanum++)
	{
		//求得当前粒子最靠近的网格点
		x = static_cast<int>((data[datanum].x - lb.x) / (dimsize.x / dim.x));
		y = static_cast<int>((data[datanum].y - lb.y) / (dimsize.y / dim.y));
		z = static_cast<int>((data[datanum].z - lb.z) / (dimsize.z / dim.z));

		for (int j = -1; j <= 1; j++)
		{
			for (int k = -1; k <= 1; k++)
			{
				for (int i = -1; i <= 1; i++)
				{
					int nodeindex = (x + i) + (y + j)*(dim.x + 1)*(dim.z + 1) + (z + k)*(dim.x + 1);

					if ((x + i) < (dim.x + 1) && (x + i) >= 0 &&
						(y + j) < (dim.y + 1) && (y + j) >= 0 &&
						(z + k) < (dim.z + 1) && (z + k) >=0 )
					{

						float xp = abs(surface[nodeindex].loc.x - data[datanum].x) / dimsize.x;
						float yp = abs(surface[nodeindex].loc.y - data[datanum].y) / dimsize.y;
						float zp = abs(surface[nodeindex].loc.z - data[datanum].z) / dimsize.z;

						surface[nodeindex].value += (xp + yp + zp) / 3 * density[datanum];
					}
				}
			}
		}
	}


	assert(plinit);

	for (int k = 0; k < dim.z + 1; k++)
	{
		for (int i = 0; i < dim.x + 1; i++)
		{
			int nodeindex = i+ k*(dim.x + 1);

			for (int j = 0; j <dim.y+1 ; j++)
			{ 
				int sfindex = nodeindex + j*(dim.x + 1)*(dim.z + 1);
				int max = 0;
				if (surface[sfindex].value>threshold && j>max)
				{
					plain[nodeindex] = surface[sfindex].loc.y;
					max = j;
				}
			}
		}
	}

	isconstructed = true;
}

void Surface::WriteToFile(ofstream& out)
{
	assert(isconstructed);

	int lt = 0;
	int rt = 0;
	int lb = 0;
	int rb = 0;

	for (int k = 0; k < dim.z ; k++)
	{
		for (int i = 0; i < dim.x ; i++)
		{
			int _lt = i + k*(dim.x + 1);
			int _rt = (i+1) + k*(dim.x + 1);
			int _lb = i + (k+1)*(dim.x + 1);
			int _rb = (i+1) + (k + 1)*(dim.x + 1);

			assert(_lt < ((dim.x + 1)*dim.z - 1)
				&& _rt < ((dim.x + 1)*dim.z)
				&& _lb < ((dim.z + 1)*(dim.x + 1) - 1)
				&& _rb < ((dim.x + 1)*(dim.z + 1)));

			out << "<" << surface[_lt].loc.x << ", " << plain[_lt] << ", " << surface[_lt].loc.z << "> ";
			out << "<" << surface[_rt].loc.x << ", " << plain[_rt] << ", " << surface[_rt].loc.z << "> ";
			out << "<" << surface[_lb].loc.x << ", " << plain[_lb] << ", " << surface[_lb].loc.z << "> " << endl;

			out << "<" << surface[_rb].loc.x << ", " << plain[_rb] << ", " << surface[_rb].loc.z << "> ";
			out << "<" << surface[_rt].loc.x << ", " << plain[_rt] << ", " << surface[_rt].loc.z << "> ";
			out << "<" << surface[_lb].loc.x << ", " << plain[_lb] << ", " << surface[_lb].loc.z << "> " << endl;

		}
	}

	for (int i = 0; i < dim.x; i++)
	{
		//back
		out << "<" << surface[i].loc.x << ", " << 0.0f << ", " << surface[i].loc.z << "> ";
		out << "<" << surface[i].loc.x << ", " << plain[i] << ", " << surface[i].loc.z << "> ";
		out << "<" << surface[i + 1].loc.x << ", " << 0.0f << ", " << surface[i + 1].loc.z << "> " << endl;

		out << "<" << surface[i].loc.x << ", " << plain[i] << ", " << surface[i].loc.z << "> ";
		out << "<" << surface[i + 1].loc.x << ", " << 0.0f << ", " << surface[i + 1].loc.z << "> ";
		out << "<" << surface[i + 1].loc.x << ", " << plain[i + 1] << ", " << surface[i + 1].loc.z << "> " << endl;

		//front
		out << "<" << surface[i + dim.z*(dim.x + 1)].loc.x << ", " << 0.0f << ", " << surface[i + dim.z*(dim.x + 1)].loc.z << "> ";
		out << "<" << surface[i + dim.z*(dim.x + 1)].loc.x << ", " << plain[i + dim.z*(dim.x + 1)] << ", " << surface[i + dim.z*(dim.x + 1)].loc.z << "> ";
		out << "<" << surface[i + 1 + dim.z*(dim.x + 1)].loc.x << ", " << 0.0f << ", " << surface[i + 1 + dim.z*(dim.x + 1)].loc.z << "> " << endl;

		out << "<" << surface[i + dim.z*(dim.x + 1)].loc.x << ", " << plain[i + dim.z*(dim.x + 1)] << ", " << surface[i + dim.z*(dim.x + 1)].loc.z << "> ";
		out << "<" << surface[i + 1 + dim.z*(dim.x + 1)].loc.x << ", " << 0.0f << ", " << surface[i + 1 + dim.z*(dim.x + 1)].loc.z << "> ";
		out << "<" << surface[i + 1 + dim.z*(dim.x + 1)].loc.x << ", " << plain[i + 1 + dim.z*(dim.x + 1)] << ", " << surface[i + 1 + dim.z*(dim.x + 1)].loc.z << "> " << endl;
	}

	for (int k = 0; k < dim.z; k++)
	{
		//left
		out << "<" << surface[k*(dim.x + 1)].loc.x << ", " << 0.0f << ", " << surface[k*(dim.x + 1)].loc.z << "> ";
		out << "<" << surface[k*(dim.x + 1)].loc.x << ", " << plain[k*(dim.x + 1)] << ", " << surface[k*(dim.x + 1)].loc.z << "> ";
		out << "<" << surface[(k + 1)*(dim.x + 1)].loc.x << ", " << 0.0f << ", " << surface[(k + 1)*(dim.x + 1)].loc.z << "> " << endl;

		out << "<" << surface[k*(dim.x + 1)].loc.x << ", " << plain[k*(dim.x + 1)] << ", " << surface[k*(dim.x + 1)].loc.z << "> ";
		out << "<" << surface[(k + 1)*(dim.x + 1)].loc.x << ", " << 0.0f << ", " << surface[(k + 1)*(dim.x + 1)].loc.z << "> ";
		out << "<" << surface[(k + 1)*(dim.x + 1)].loc.x << ", " << plain[(k + 1)*(dim.x + 1)] << ", " << surface[(k + 1)*(dim.x + 1)].loc.z << "> " << endl;

		//right
		out << "<" << surface[k*(dim.x + 1) + dim.x].loc.x << ", " << 0.0f << ", " << surface[k*(dim.x + 1) + dim.x].loc.z << "> ";
		out << "<" << surface[k*(dim.x + 1) + dim.x].loc.x << ", " << plain[k*(dim.x + 1) + dim.x] << ", " << surface[k*(dim.x + 1) + dim.x].loc.z << "> ";
		out << "<" << surface[(k + 1)*(dim.x + 1) + dim.x].loc.x << ", " << 0.0f << ", " << surface[(k + 1)*(dim.x + 1) + dim.x].loc.z << "> " << endl;

		out << "<" << surface[k*(dim.x + 1) + dim.x].loc.x << ", " << plain[k*(dim.x + 1) + dim.x] << ", " << surface[k*(dim.x + 1) + dim.x].loc.z << "> ";
		out << "<" << surface[(k + 1)*(dim.x + 1) + dim.x].loc.x << ", " << 0.0f << ", " << surface[(k + 1)*(dim.x + 1) + dim.x].loc.z << "> ";
		out << "<" << surface[(k + 1)*(dim.x + 1) + dim.x].loc.x << ", " << plain[(k + 1)*(dim.x + 1) + dim.x] << ", " << surface[(k + 1)*(dim.x + 1) + dim.x].loc.z << "> " << endl;
	}


}

void Surface::DrawSurface()
{
	assert(isconstructed);

	int _lt = 0;
	int _rt = 0;
	int _lb = 0;
	int _rb = 0;

	glBegin(GL_TRIANGLES);
	{

		for (int k = 0; k < dim.z; k++)
		{
			for (int i = 0; i < dim.x; i++)
			{
				_lt = i + k*(dim.x + 1);
				_rt = (i + 1) + k*(dim.x + 1);
				_lb = i + (k + 1)*(dim.x + 1);
				_rb = (i + 1) + (k + 1)*(dim.x + 1);

				assert(_lt < ((dim.x+1)*dim.z-1)
					&& _rt < ((dim.x + 1)*dim.z)
					&& _lb < ((dim.z + 1)*(dim.x+1)-1)
					&& _rb < ((dim.x + 1)*(dim.z + 1)));


				glVertex3f(surface[_lt].loc.x, plain[_lt], surface[_lt].loc.z);
				glVertex3f(surface[_rt].loc.x, plain[_rt], surface[_rt].loc.z);
				glVertex3f(surface[_lb].loc.x, plain[_lb], surface[_lb].loc.z);

				glVertex3f(surface[_rb].loc.x, plain[_rb], surface[_rb].loc.z);
				glVertex3f(surface[_rt].loc.x, plain[_rt], surface[_rt].loc.z);
				glVertex3f(surface[_lb].loc.x, plain[_lb], surface[_lb].loc.z);

			}
		}

		int i_1 = 0;
		for (int i = 0; i < dim.x ; i++)
		{
			i_1 = i + 1;
			//back
			glVertex3f(surface[i].loc.x, 0.0f, surface[i].loc.z);
			glVertex3f(surface[i].loc.x, plain[i], surface[i].loc.z);
			glVertex3f(surface[i_1].loc.x, 0.0f, surface[i_1].loc.z);

			glVertex3f(surface[i].loc.x, plain[i], surface[i].loc.z);
			glVertex3f(surface[i_1].loc.x, 0.0f, surface[i_1].loc.z);
			glVertex3f(surface[i_1].loc.x, plain[i_1], surface[i_1].loc.z);

			//front
			glVertex3f(surface[i + dim.z*(dim.x + 1)].loc.x, 0.0f, surface[i + dim.z*(dim.x + 1)].loc.z);
			glVertex3f(surface[i + dim.z*(dim.x + 1)].loc.x, plain[i + dim.z*(dim.x + 1)], surface[i + dim.z*(dim.x + 1)].loc.z);
			glVertex3f(surface[i + 1 + dim.z*(dim.x + 1)].loc.x, 0.0f, surface[i + 1 + dim.z*(dim.x + 1)].loc.z);

			glVertex3f(surface[i + dim.z*(dim.x + 1)].loc.x, plain[i + dim.z*(dim.x + 1)], surface[i + dim.z*(dim.x + 1)].loc.z);
			glVertex3f(surface[i + 1 + dim.z*(dim.x + 1)].loc.x, 0.0f, surface[i + 1 + dim.z*(dim.x + 1)].loc.z);
			glVertex3f(surface[i + 1 + dim.z*(dim.x + 1)].loc.x, plain[i + 1 + dim.z*(dim.x + 1)], surface[i + 1 + dim.z*(dim.x + 1)].loc.z);
		}

		for (int k = 0; k < dim.z; k++)
		{
			//left
			glVertex3f(surface[k*(dim.x + 1)].loc.x, 0.0f, surface[k*(dim.x + 1)].loc.z);
			glVertex3f(surface[k*(dim.x + 1)].loc.x, plain[k*(dim.x + 1)], surface[k*(dim.x + 1)].loc.z);
			glVertex3f(surface[(k + 1)*(dim.x + 1)].loc.x, 0.0f, surface[(k+1)*(dim.x + 1)].loc.z);

			glVertex3f(surface[k*(dim.x + 1)].loc.x, plain[k*(dim.x + 1)], surface[k*(dim.x + 1)].loc.z);
			glVertex3f(surface[(k + 1)*(dim.x + 1)].loc.x, 0.0f, surface[(k + 1)*(dim.x + 1)].loc.z);
			glVertex3f(surface[(k + 1)*(dim.x + 1)].loc.x, plain[(k+1)*(dim.x + 1)], surface[(k + 1)*(dim.x + 1)].loc.z);
		
			//right
			glVertex3f(surface[k*(dim.x + 1) + dim.x].loc.x, 0.0f, surface[k*(dim.x + 1) + dim.x].loc.z);
			glVertex3f(surface[k*(dim.x + 1) + dim.x].loc.x, plain[k*(dim.x + 1) + dim.x], surface[k*(dim.x + 1) + dim.x].loc.z);
			glVertex3f(surface[(k + 1)*(dim.x + 1) + dim.x].loc.x, 0.0f, surface[(k + 1)*(dim.x + 1) + dim.x].loc.z);

			glVertex3f(surface[k*(dim.x + 1) + dim.x].loc.x, plain[k*(dim.x + 1) + dim.x], surface[k*(dim.x + 1) + dim.x].loc.z);
			glVertex3f(surface[(k + 1)*(dim.x + 1) + dim.x].loc.x, 0.0f, surface[(k + 1)*(dim.x + 1) + dim.x].loc.z);
			glVertex3f(surface[(k + 1)*(dim.x + 1) + dim.x].loc.x, plain[(k + 1)*(dim.x + 1) + dim.x], surface[(k + 1)*(dim.x + 1) + dim.x].loc.z);
		}


	}
	glEnd();
}

void Surface::DrawParticle()
{
	int x = 0;
	int z = 0;
	int plindex = 0;
	float y = 0.0f;
	float relaty = 0.0f;
	for (int i = 0; i < count; i++)
	{
		x = static_cast<int>(data[i].x / dimsize.x);
		assert(x >= 0 && x <= dim.x);
		z = static_cast<int>(data[i].z / dimsize.z);
		assert(z >= 0 && z <= dim.z);

		plindex = x + z*(dim.x + 1);
		assert(plindex >= 0 && plindex <= (dim.x*dim.z));

		y = data[i].y;
		relaty = (plain[x + z*(dim.x + 1)] + plain[(x+1) + z*(dim.x + 1)] + plain[x + (z+1)*(dim.x + 1)] + plain[(x+1) + (z+1)*(dim.x + 1)]) / 4.0f;
		if (y > relaty)
		{
			glPushMatrix();
			glTranslatef(data[i].x, data[i].y, data[i].z);
			glutSolidSphere(1.0f, 10, 10);
			glPopMatrix();
		}
	}
}