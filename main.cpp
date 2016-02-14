#include <GL\freeglut.h>
#include <fstream>
#include <iostream>
#include <ctime>
using namespace std;

#include "Solver.h"
#include "Surface.h"

#define PARTICLE     0
#define SURFACE      1
#define WRITETOFILE  0
#define DRAWSURFACE  1

#if WRITETOFILE==1
ofstream fout("fdata.txt");
#endif


struct Vec3f
{
	float x;
	float y;
	float z;
};

typedef Vec3f Translate;
typedef Vec3f Rotate;


Rotate rot;
Translate trans;

Solver solver(1024 * 16);

GLfloat sun_light_position[] = { 25.0f, 15.0f, -20.0f, 0.0f };

int i = 0;

clock_t st = 0, en = 0;

//Surface *surface;
 
void init(void)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glShadeModel(GL_SMOOTH);

	rot.x = 0.0f;
	rot.y = 0.0f;
	rot.z = 0.0f;

	trans.x = 0.0f;
	trans.y = 0.0f;
	trans.z = -90.0f;
	/*glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glLightfv(GL_LIGHT0, GL_POSITION, sun_light_position);*/
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(trans.x, trans.y, trans.z);
	glRotatef(rot.x, 1.0f, 0.0f, 0.0f);
	glRotatef(rot.y, 0.0f, 1.0f, 0.0f);
	glRotatef(rot.z, 0.0f, 0.0f, 1.0f);
	glPushMatrix();
	glScalef(50.0f, 30.0f, 40.0f);
	glutWireCube(1.0f);
	glPopMatrix();

	solver.Update();
	i++;
	cout << i << endl;

	float3* temp3 = solver.GetPos();

#if PARTICLE==1

	for (int index = 0; index < 16 * 1024; index++)
	{
		//glPushMatrix();
		//glTranslatef(temp[index].x-25.0f, temp[index].y-15.0f, temp[index].z-20.0f);
		//glutSolidSphere(0.5f, 10, 10);
		glPointSize(5);
		glBegin(GL_POINTS);
		glVertex3f(temp3[index].x - 25.0f, temp3[index].y - 15.0f, temp3[index].z - 20.0f);
		glEnd();
		//glPopMatrix();

		//fout << temp[index].x - 25.0f << ", " << temp[index].y - 15.0f << ", " << temp[index].z - 20.0f << endl;

	}
#endif

#if SURFACE==1

	float* temp = solver.GetDensity();
	Surface surface(temp3, temp, 1024 * 16, 
		          make_float3(0.0f, 0.0f, 0.0f),
		          make_float3(50.0f, 30.0f, 40.0f),
		          make_int3(50, 30, 40), 0);

	surface.ConstructSurface();
#if DRAWSURFACE==1
	glPushMatrix();
	glTranslatef(-25.0f, -15.0f, -20.0f);
	surface.DrawSurface();
	glPopMatrix();
#endif

#if WRITETOFILE==1
	surface.WriteToFile(fout);
#endif

#endif

	st = clock();

	float dura = float(st - en) / 1000.0f;

	cout << dura << endl;

	en = st;

	//char fname[12];
	//sprintf(fname, "%d.jpg", cycle);

	//SaveBMPFromOpenGl(fname);

	char text[32];
	sprintf_s(text, "The frame is %f: ", 1.0f / dura);
	glutSetWindowTitle(text);

	glutSwapBuffers(); 

}

void reshape(int width, int height)
{
	glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, static_cast<float>(width) / (height = (height == 0 ? 1 : height)), 0.01, 1000.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	default:
		break;
	}

	glutPostRedisplay();
}

void special(int key, int x, int y)
{
	switch (key)
	{
	default:
		break;
	}

	glutPostRedisplay();
}

void processMenuEvents(int option)
{
	switch (option)
	{
	default:
		break;
	}
}

void createGLUTMenus(int& menu)
{
	menu = glutCreateMenu(processMenuEvents);

	/*glutAddMenuEntry("Red",RED);
	glutAddMenuEntry("Blue",BLUE);
	glutAddMenuEntry("Green",GREEN);
	glutAddMenuEntry("White",WHITE);*/

	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void mouse(int button, int state, int x, int y)
{
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		if (GLUT_DOWN == state)
		{
		}
		else if (GLUT_UP == state)
		{
		}
		break;


	case GLUT_MIDDLE_BUTTON:
		if (GLUT_DOWN == state)
		{
		}
		else if (GLUT_UP == state)
		{
		}
		break;


	case GLUT_RIGHT_BUTTON:
		if (GLUT_DOWN == state)
		{
		}
		else if (GLUT_UP == state)
		{
		}
		break;


	default:
		break;
	}

	glutPostRedisplay();
}

void motion(int x, int y)
{
}

void passivemotion(int x, int y)
{
}


int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(512, 512);
	glutCreateWindow(argv[0]);
	en = clock();
	init();
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);

	int menu;
	createGLUTMenus(menu);

	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(passivemotion);

	glutMainLoop();

	glutDetachMenu(GLUT_RIGHT_BUTTON);
	glutDestroyMenu(menu);

	return 0;
}