UNAME := $(shell uname)

CC=g++

ifeq ($(UNAME), Linux)
all: flow_mod.cpp colorcode.h colorcode.cpp
	$(CC) -c -Wall -I/usr/include/python2.7/ -fpic -g colorcode.cpp -o colorcode.o
	$(CC) -c -Wall -I/usr/include/python2.7/ -fpic -g flow_mod.cpp -o flow_mod.o
	$(CC) -g colorcode.o flow_mod.o -shared -Wl,-soname,"flow_mod.so" -fpic -o flow_mod.so
else
flow_mod: flow_mod.cpp colorcode.h colorcode.cpp
	$(CC) -c -Wall  -I/usr/local/lib/python2.7/site-packages/numpy/core/include -I /usr/local/Cellar/python/2.7.6_1/Frameworks/Python.framework/Versions/2.7/include -fpic -g colorcode.cpp -o colorcode.o
	$(CC) -c -Wall  -I/usr/local/lib/python2.7/site-packages/numpy/core/include -I /usr/local/Cellar/python/2.7.6_1/Frameworks/Python.framework/Versions/2.7/include -fpic -g flow_mod.cpp -o flow_mod.o
	$(CC) -dynamiclib -Wl,-headerpad_max_install_names,-undefined,dynamic_lookup,-compatibility_version,1.0,-current_version,1.0 -o flow_mod.so colorcode.o flow_mod.o
endif

clean:
	rm -f *.o flow_mod
