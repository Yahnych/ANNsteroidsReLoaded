#!/bin/bash 
#Compiling asteroids
SRC_DIR=Asteroids/src
g++ -c -std=c++11 $SRC_DIR/GamePiece.cpp $SRC_DIR/Shape.cpp $SRC_DIR/Space.cpp
g++ $SRC_DIR/GamePiece.o $SRC_DIR/Shape.o $SRC_DIR/Space.o -framework GLUT -framework OpenGL -o Asteroids
