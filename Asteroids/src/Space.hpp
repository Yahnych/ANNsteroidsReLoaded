//
//  Space.hpp
//  FinalProject
//
//  Created by Adam Barson on 4/19/17.
//  Copyright © 2017 Adam Barson. All rights reserved.
//

#ifndef Space_hpp
#define Space_hpp

#define NUM_THREADS 10000

#include <stdio.h>
#include <thread>

#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif


extern "C" {
#include <pthread.h>
}
//#include <OpenAL/OpenAL.h>

//#define CONTROLLER true // the speed of the powerup

/* Game Functionality */
void spawnThrustFire();
void drawBullets();
void moveBullets();
void saveGame();
void loadGame();


/**
 * Requires: nothing
 * Modifies: the GUI display
 * Effects: add initial Asteroids to screen
 */
void drawAllAsteroids();

/**
 * Requires: nothing
 * Modifies: all the Bullets in Space
 * Effects: move all of the Bullets in Space
 */
void moveAllBullets();

/**
 * Requires: nothing
 * Modifies: all the Asteroids in Space
 * Effects: move all of the Asteroids in Space
 */
void moveAllAsteroids();

/**
 * Requires: all Asteroids of current level have been destroyed
 * Modifies: Space
 * Effects: prepare to generate Asteroids for next level
 */
void nextLevel();

/**
 * Requires: nothing
 * Modifies: game state of Space
 * Effects: pause/unpause the game
 */
void pause();

/**
 * Requires: nothing
 * Modifies: game state of Space
 * Effects: open/close meny
 */
void openMenu();

/* Game Functionality */

/* GLUT Functionality */
// Program initialization NOT OpenGL/GLUT dependent,
// as we haven't created a GLUT window yet
void init(void);



// Callback functions for GLUT */

// Draw the window - this is where all the GL actions are
void display(void);

// Called when window is resized,
// also when window is first created,
// before the first call to display().
void reshape(int w, int h);

// Refresh the display, called when system is idle
void refresh(void);

// Trap and process keyboard events
void kbd(unsigned char key, int x, int y);

// Handle "mouse cursor moved" events
void cursor(int x, int y);

// Handle "mouse moved with button pressed" events
void drag(int x, int y);

// Handle mouse button pressed and released events
void mouse(int button, int state, int x, int y);

int graphicsPlay(int argc, char *argv[]);

void create_ppm(char *prefix, int frame_id, unsigned int width, unsigned int height,
                           unsigned int color_max, unsigned int pixel_nbytes, GLubyte *pixels);

#endif /* Space_hpp */
