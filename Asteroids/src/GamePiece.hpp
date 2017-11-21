//
//  GamePiece.hpp
//  FinalProject
//
//  Created by Adam Barson on 4/19/17.
//  Copyright © 2017 Adam Barson. All rights reserved.
//

#ifndef GamePiece_hpp
#define GamePiece_hpp

#include <iostream>
#include <string>
#include <vector>
#include "Shape.hpp"
#include <thread>
#define MAX_SPEED 12 //max speed that the Ship can move in
#define BUFFER 20 //the distance from the edge of the screen the Ship or Asteroid can go before being moved to the opposite side of the screen
#define ROTATION_FORCE 5 //the amount of degrees the Ship rotates
#define FRICTION 0.95 //the force used to slow down the Ship
#define SHOT_DELAY 8 //the amount of "time" between shots
#define BULLET_SPEED 15 //the speed of the bullet
#define BULLET_SIZE 4 //the radius of the bullet
#define POWER_SPEED 5 // the speed of the powerup
using namespace std;

class GamePiece{
public:
    /**
     * Requires: nothing
     * Modifies: instance fields
     * Effects: set instance fields to default values
     */
    GamePiece();
    
    /**
     * Requires: nothing
     * Modifies: the GUI display
     * Effects: add the shape to the screen
     */
    virtual void drawShape()=0;
    

    /**
     * Requires: nothing
     * Modifies: nothing
     * Effects: get the location of the GamePiece
     */
    virtual Point2D getLocation() const=0;
    
    /**
     * Requires: nothing
     * Modifies: the location of the GamePiece
     * Effects: move the GamePiece towards a target location. If it goes out of the bounds of the screen, move it to the opposite side of the screen, for a continuous flow effect.
     */
    virtual void move() = 0;
    
    /**
     * Requires: vIn >= 1
     * Modifies: velocity
     * Effects: sets a new velocity
     */
    virtual void setVelocity(double vIn);
    
    /**
     * Requires: nothing
     * Modifies: nothing
     * Effects: get the velocity of the GamePiece
     */
    virtual double getVelocity() const;
protected:
    //fields
    Point2D target;
    Point2D origin;
private:
    //field
    double velocity;
};

class Ship: public GamePiece{
public:
    /**
     * Requires: nothing
     * Modifies: instance fields
     * Effects: set instance fields to default values
     */
    Ship();
    
    /**
     * Requires: the Ship has been destroyed
     * Modifies: the GUI display
     * Effects: respawn the ship in the middle of the screen
     */
    void regenerate();
    
    /**
     * Requires: nothing
     * Modifies: the direction of the Ship
     * Effects: rotate the Ship to the right
     */
    void rotateR();
    
    /**
     * Requires: nothing
     * Modifies: the direction of the Ship
     * Effects: rotate the Ship to the left
     */
    void rotateL();
    
    /**
     * Requires: nothing
     * Modifies: the GUI display
     * Effects: Adds shotdelay to stop spamming
     */
    void shoot();
    
    /**
     * Requires: nothing
     * Modifies: nothing
     * Effects: get the current direction of the tip of the ship
     */
    Point2D getDirection() const;
    
    /**
     * Requires: nothing
     * Modifies: velocity and position of the Ship
     * Effects: continues to move the Ship and applies friction. Put the ship back in bounds if it goes out of bounds
     */
    void update();
    
    /**
     * Requires: nothing
     * Modifies: velocity
     * Effects: reduce the velocity by the FRICTION rate
     */
    void applyFriction();
    
    /**
     * Requires: nothing
     * Modifies: direction
     * Effects: calculate the direction of the ship as a normalized 2D vector
     */
    void updateDirection();
    
    /**
     * Requires: nothing
     * Modifies: the location of the Ship
     * Effects: move the Ship using its velocity
     */
    void keepMoving();
    
    /**
     * Requires: nothing
     * Modifies: velocity
     * Effects: if the speed of the ship is too high, reduce it to fit the restraints of MAX_SPEED
     */
    void speedCap();
    
    /**
     * Requires: nothing
     * Modifies: the location of the Ship
     * Effects: if the Ship goes of screen, place it on the other side of the screen.
     */
    void checkBounds();
    
    /**
     * Requires: nothing
     * Modifies: shotDelay
     * Effects: time between shots
     */
    void gunUpdate();
    
    int getNumLives() const;
    
    void setNumLives(int nIn);
    
    /**
     * Requires: nothing
     * Modifies: shotDelay
     * Effects: set the shotDelay
     */
    void setShotDelay(int sIn);
    
    /**
     * Requires: nothing
     * Modifies: nothing
     * Effects: get the shotDelay
     */
    int getShotDelay() const;
    /**
     * Requires: nothing
     * Modifies: nothing
     * Effects: gets shape of ship
     */
    
    void setRespawning(int rIn);
    
    int getRespawning() const;
    
    void blink();
    
    void restore();
    
    Triangle_Coord getTriangle() const;
    
    virtual void drawShape() override;
    
    virtual Point2D getLocation() const override;
    
    virtual void move() override;
    
private:
    //fields
    Triangle_Coord shape;
    int numLives;
    Point2D direction;
    Point2D velocity;
    double shotDelay;
    int respawnTimer;
};

class Bullet: public GamePiece{
public:
    /**
     * Requires: Point2D for direction and location
     * Modifies: instance fields
     * Effects: set direction of Bullet and set instance fields to default values
     */
    Bullet(Point2D dIn, Point2D loc);
    /**
     * Requires: Point2D for direction and location
     * Modifies: initialization of all fields
     * Effects: bullet look, size, speed, direction
     */
    void initFields(Point2D dIn, Point2D loc);
    /**
     * Requires: nothing
     * Modifies: nothing
     * Effects: get shape of bullet
     */
    
    Circle_Coord getCircle() const;
    /**
     * Requires: nothing
     * Modifies: bullet's time on board
     * Effects: nothing
     */
    
    int getLifeTime() const;
    
    virtual Point2D getLocation() const override;
    
    virtual void drawShape() override;
    
    virtual void move() override;
    
private:
    //fields
    Point2D baseVelocity;
    Point2D bdirection;
    int lifeTime;
    Circle_Coord shape;
};

class Asteroid: public GamePiece{
public:
    /**
     * Requires: nothing
     * Modifies: instance fields
     * Effects: randomize size, start location, and target location of the Asteroid
     */
    Asteroid();
    
    /**
     * Requires: nothing
     * Modifies: initialization of all fields
     * Effects: asteroid size, speed, location, direction, color
     */
    void initFields();
    /**
     * Requires: nothing
     * Modifies: nothing
     * Effects: gets shape of asteroid
     */
    
    Circle_Coord getCircle() const;
    
    virtual void drawShape() override;
    
    virtual Point2D getLocation() const override;
    
    virtual void move() override;
    
    /**
     * Requires: bullet
     * Modifies: asteroid if bullet hits it
     * Effects: checks for collision with bullet
     */
    bool detectCollision(Bullet &bIn) const;
    
    /**
     * Requires: ship
     * Modifies: asteroid if ship hits it
     * Effects: check for collision with ship
     */
    bool detectCollision(Ship &sIn) const;
private:
    //fields
    Circle_Coord shape;
    Point2D direction;
    
};
class Powerup: public GamePiece{
public:
    /**
     * Requires: nothing
     * Modifies: instance fields
     * Effects: set instance fields to default values
     */
    Powerup(color c);
    /**
     * Requires: nothing
     * Modifies: powerup
     * Effects: changes powerup look, size, speed, direction, and location
     */
    void initFields();
    
    Circle_Coord getCircle() const;
    
    virtual void drawShape() override;
    
    virtual Point2D getLocation() const override;
    
    virtual void move() override;
    /**
     * Requires: ship
     * Modifies: powerup if ship hits it
     * Effects: check for collision with ship
     */
    bool detectCollision(Ship &sIn) const;
    
private:
    //fields
    Circle_Coord shape;
    Point2D direction;

    
};



#endif /* GamePiece_hpp */
