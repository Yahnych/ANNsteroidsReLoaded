# ANNsteroidsReLoaded

## Summary
We are building a Python controller that will attempt to learn how to play Asteroids.

## Running Instructions
Compile each of the src files independently:
'''g++ -c -std=c++11 GamePiece.cpp Shape.cpp Space.cpp'''
Then, compile the executable Asteroids:
'''g++ GamePiece.o Shape.o Space.o -framework GLUT -framework OpenGL -o Asteroids'''
Now, the executable Asteroids can be run.
