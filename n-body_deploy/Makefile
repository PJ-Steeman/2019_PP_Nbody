CC = clang
CXX = clang++
CFLAGS = -Wall -std=c99 -ggdb -I../common
LIBS = -lm -lOpenCL  -lGL  $(shell sdl2-config --libs)

INCLUDES = renderer.h camera.h

all: n-body

n-body: n-body.o renderer.o camera.o
	$(CXX) n-body.o renderer.o camera.o ../common/libutils.a -o n-body $(LIBS)

n-body.o: n-body.c $(INCLUDES)
	$(CC) -c $(CFLAGS) n-body.c

renderer.o: renderer.c $(INCLUDES)
	$(CC) -c $(CFLAGS) renderer.c

camera.o: camera.cpp $(INCLUDES)
	$(CXX) -I../glm -c  camera.cpp

clean:
	rm *.o n-body
