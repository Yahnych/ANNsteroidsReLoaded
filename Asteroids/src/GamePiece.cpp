//
//  GamePiece.cpp
//  FinalProject
//
//  Created by Adam Barson on 4/19/17.
//  Copyright © 2017 Adam Barson. All rights reserved.
//

#include "GamePiece.hpp"
#include "Space.hpp"
#include <stdlib.h>
using namespace std;
GamePiece::GamePiece(){
    velocity = 0;
    target = Point2D();
    origin = Point2D();
}

void GamePiece::setVelocity(double vIn){
    if (vIn > 0){
        velocity = vIn;
    }
    else {
        velocity = 1;
    }
}

double GamePiece::getVelocity() const{
    return velocity;
}


Ship::Ship(){
    shape = Triangle_Coord(300, 300);
    shape.set_color(1, 1, 0);
    numLives = 3;
    direction = Point2D();
    velocity = Point2D();
    shotDelay = 0;
    respawnTimer = 0;
}

void Ship::regenerate(){
    shape.set_location(300, 300);
    direction.set_x(0);
    direction.set_y(0);
    velocity.set_x(0);
    velocity.set_y(0);
}
void Ship::rotateR(){
    shape.rotate(ROTATION_FORCE);
}
void Ship::rotateL(){
    shape.rotate(-ROTATION_FORCE);
}
void Ship::shoot(){
    shotDelay++;
}

Point2D Ship::getDirection() const{
    double x_dir = shape.get_tip().get_x() - shape.get_center().get_x();
    double y_dir = shape.get_tip().get_y() - shape.get_center().get_y();
    double length = sqrt(pow((x_dir), 2) + pow((y_dir), 2));
    return Point2D(x_dir/length, y_dir/length);
}

void Ship::update(){
    keepMoving();
    applyFriction();
    checkBounds();
    gunUpdate();
}

void Ship::applyFriction(){
    if (velocity.get_x() != 0){
        velocity.set_x(velocity.get_x()*FRICTION);
        if (abs(velocity.get_x()) < 0.001){
            velocity.set_x(0);
        }
    }
    
    if (velocity.get_y() != 0){
        velocity.set_y(velocity.get_y()*FRICTION);
        if (abs(velocity.get_y()) < 0.001){
            velocity.set_y(0);
        }
    }
}

void Ship::updateDirection(){
    direction.set_x(getDirection().get_x());
    direction.set_y(getDirection().get_y());
}

void Ship::keepMoving(){
    double result_x = velocity.get_x();
    double result_y = velocity.get_y();
    
    Point2D new_tip(shape.get_tip().get_x() + result_x, shape.get_tip().get_y() + result_y);
    Point2D new_bl(shape.get_bl().get_x() + result_x, shape.get_bl().get_y() + result_y);
    Point2D new_br(shape.get_br().get_x() + result_x, shape.get_br().get_y() + result_y);
    Point2D center(shape.get_center().get_x() + result_x, shape.get_center().get_y() + result_y);
    
    shape.set_tip(new_tip);
    shape.set_bl(new_bl);
    shape.set_br(new_br);
    shape.set_center(center);
}

void Ship::speedCap(){
    double result_x = velocity.get_x();
    double result_y = velocity.get_y();
    
    double speed = sqrt(pow((result_x), 2) + pow((result_y), 2));
    if (speed > MAX_SPEED){
        result_x/=speed;
        result_y/=speed;
        result_x*=MAX_SPEED;
        result_y*=MAX_SPEED;
    }
    velocity.set_x(result_x);
    velocity.set_y(result_y);
    speed = sqrt(pow((result_x), 2) + pow((result_y), 2));
}

void Ship::checkBounds(){
    if (shape.get_center().get_x() < -BUFFER || shape.get_center().get_x() > 600 + BUFFER || shape.get_center().get_y() < -BUFFER || shape.get_center().get_y() > 600 + BUFFER){
        Point2D rel_tip = shape.get_rel_tip();
        Point2D rel_bl = shape.get_rel_bl();
        Point2D rel_br = shape.get_rel_br();
        if (shape.get_center().get_x() < 0){
            shape.set_center(Point2D(600, shape.get_center().get_y()));
        }
        else if (shape.get_center().get_x() > 600){
            shape.set_center(Point2D(0, shape.get_center().get_y()));
        }
        else if (shape.get_center().get_y() < 0){
            shape.set_center(Point2D(shape.get_center().get_x(), 600));
        }
        else if (shape.get_center().get_y() > 600){
            shape.set_center(Point2D(shape.get_center().get_x(), 0));
        }
        shape.set_tip(Point2D(shape.get_center().get_x() + rel_tip.get_x(), shape.get_center().get_y() + rel_tip.get_y()));
        shape.set_bl(Point2D(shape.get_center().get_x() + rel_bl.get_x(), shape.get_center().get_y() + rel_bl.get_y()));
        shape.set_br(Point2D(shape.get_center().get_x() + rel_br.get_x(), shape.get_center().get_y() + rel_br.get_y()));
    }
}

void Ship::gunUpdate(){
    if (shotDelay > 0){
        shotDelay++;
    }
    if (shotDelay > SHOT_DELAY){
        shotDelay = 0;
    }
}

int Ship::getNumLives() const{
    return numLives;
}

void Ship::setNumLives(int nIn){
    numLives = nIn;
}

void Ship::setShotDelay(int sIn){
    shotDelay = sIn;
}

int Ship::getShotDelay() const{
    return shotDelay;
}

void Ship::setRespawning(int rIn){
    respawnTimer = rIn;
}

int Ship::getRespawning() const{
    return respawnTimer;
}

void Ship::blink(){
    if (respawnTimer % 5 == 0){
        shape.set_color(1, 1, 0);
    } else {
        shape.set_color(0, 0, 0);
    }
}

Triangle_Coord Ship::getTriangle() const{
    return shape;
}

void Ship::drawShape() {
    shape.draw();
}


Point2D Ship::getLocation() const{
    return shape.get_tip();
}

void Ship::move() {
    updateDirection();
    
    velocity.set_x(velocity.get_x() + direction.get_x());
    velocity.set_y(velocity.get_y() + direction.get_y());
    
    speedCap();
    
    spawnThrustFire();
}


Bullet::Bullet(Point2D dIn, Point2D loc): GamePiece(){
    shape = Circle_Coord();
    bdirection = dIn;
    initFields(dIn, loc);
    lifeTime = 0;
}

void Bullet::initFields(Point2D dIn, Point2D loc){
    setVelocity(BULLET_SPEED);
    shape.set_radius(BULLET_SIZE);
    shape.set_color(1, 0, 0);
    shape.set_center(Point2D(loc.get_x() + dIn.get_x(), loc.get_y() + dIn.get_y()));
    
    
    
}

Circle_Coord Bullet::getCircle() const{
    return shape;
}

int Bullet::getLifeTime() const{
    return lifeTime;
}

Point2D Bullet::getLocation() const{
    return getCircle().get_center();
}
void Bullet::drawShape() {
    shape.draw();
}


void Bullet::move() {
    lifeTime++;
    if ((shape.get_center().get_x() > -BUFFER && shape.get_center().get_x() < 600 + BUFFER && shape.get_center().get_y() > -BUFFER && shape.get_center().get_y() < 600 + BUFFER)){
        shape.set_center(shape.get_center().get_x() + (bdirection.get_x())*BULLET_SPEED, shape.get_center().get_y() + (bdirection.get_y())*BULLET_SPEED);
    }
    else{
        if (shape.get_center().get_x() < 0){
            shape.set_center(600, shape.get_center().get_y());
        }
        else if (shape.get_center().get_x() > 600){
            shape.set_center(0, shape.get_center().get_y());
        }
        else if (shape.get_center().get_y() < 0){
            shape.set_center(shape.get_center().get_x(), 600);
        }
        else{
            shape.set_center(shape.get_center().get_x(), 0);
        }
    }
}

Asteroid::Asteroid(): GamePiece(){
    shape = Circle_Coord();
    initFields();
}

void Asteroid::initFields(){
    double r_velocity = rand() % 8 + 3;
    double r_size = rand() % 30 + 20;
    //double r_size = 10;
    setVelocity(r_velocity);
    shape.set_radius(r_size);
    
    shape.set_outside_color(100.0/255, 100.0/255, 100.0/255);
    shape.set_color(150.0/255, 150.0/255, 150.0/255);
    
    int c = rand() % 4;
    int x;
    int y;
    
    //there are four different cases for spawning the Asteroid, one for each side of the screen
    switch(c){
        case(0):
            x = 600;
            y = rand() % 300 + 200;
            break;
        case(1):
            x = rand() % 300 + 200;
            y = 600;
            break;
        case(2):
            x = 0;
            y = rand() % 300 + 200;
            break;
        case(3):
            x = rand() % 300 + 200;
            y = 0;
            break;
        default:
            x = 0;
            y = 0;
    }
    shape.set_center(x, y);
    
    //set the target to be the opposite side of the screen of the Asteroid
    target.set_x(600 - x);
    target.set_y(600 - y);
    
    double x_dir = target.get_x() - getLocation().get_x();
    double y_dir = target.get_y() - getLocation().get_y();
    double length = sqrt(pow((x_dir), 2) + pow((y_dir), 2));
    direction.set_x(x_dir/length);
    direction.set_y(y_dir/length);
    
}

Circle_Coord Asteroid::getCircle() const{
    return shape;
}

void Asteroid::drawShape(){
    shape.draw();
}

Point2D Asteroid::getLocation() const{
    return shape.get_center();
}

void Asteroid::move(){
    if ((getLocation().get_x() > -BUFFER && getLocation().get_x() < 600 + BUFFER && getLocation().get_y() > -BUFFER && getLocation().get_y() < 600 + BUFFER)){
        shape.set_center(getLocation().get_x() + (direction.get_x()) * getVelocity(), getLocation().get_y() + (direction.get_y()) * getVelocity());
    }
    else{
        if (getLocation().get_x() < 0){
            shape.set_center(600, getLocation().get_y());
        }
        else if (getLocation().get_x() > 600){
            shape.set_center(0, getLocation().get_y());
        }
        else if (getLocation().get_y() < 0){
            shape.set_center(getLocation().get_x(), 600);
        }
        else{
            shape.set_center(getLocation().get_x(), 0);
        }
    }
}

bool Asteroid::detectCollision(Bullet &bIn) const{
    if (shape.is_overlapping(bIn.getCircle())){
        return true;
    } else {
        return false;
    }
}

bool Asteroid::detectCollision(Ship &sIn) const{
    Triangle_Coord triangle = sIn.getTriangle();
    if (shape.is_overlapping(triangle.get_tip().get_x(), triangle.get_tip().get_y()) ||
            shape.is_overlapping(triangle.get_bl().get_x(), triangle.get_bl().get_y()) ||
        shape.is_overlapping(triangle.get_br().get_x(), triangle.get_br().get_y())){
        return true;
    } else {
        return false;
    }
}
Powerup::Powerup(color c): GamePiece(){
    shape = Circle_Coord();
    initFields();
    shape.set_outside_color(1, 1, 1);
    shape.set_color(c.red, c.green, c.blue);
}

void Powerup::initFields(){
    double r_velocity = 5;
    double r_size = 15;
    setVelocity(r_velocity);
    shape.set_radius(r_size);
    
    int c = rand() % 4;
    int x;
    int y;
    
    
    //there are four different cases for spawning the powerup, one for each side of the screen
    switch(c){
        case(0):
            x = 600;
            y = rand() % 300 + 200;
            break;
        case(1):
            x = rand() % 300 + 200;
            y = 600;
            break;
        case(2):
            x = 0;
            y = rand() % 300 + 200;
            break;
        case(3):
            x = rand() % 300 + 200;
            y = 0;
            break;
        default:
            x = 0;
            y = 0;
    }

    shape.set_center(x, y);
    target.set_x(600 - x);
    target.set_y(600 - y);
    
    double x_dir = target.get_x() - getLocation().get_x();
    double y_dir = target.get_y() - getLocation().get_y();
    double length = sqrt(pow((x_dir), 2) + pow((y_dir), 2));
    direction.set_x(x_dir/length);
    direction.set_y(y_dir/length);
}
void Powerup::drawShape(){
    shape.draw();
}

Point2D Powerup::getLocation() const{
    return shape.get_center();
}
Circle_Coord Powerup::getCircle() const{
    return shape;
}

void Powerup::move(){
    if ((shape.get_center().get_x() > -BUFFER && shape.get_center().get_x() < 600 + BUFFER && shape.get_center().get_y() > -BUFFER && shape.get_center().get_y() < 600 + BUFFER)){
        shape.set_center(shape.get_center().get_x() + (direction.get_x())*POWER_SPEED, shape.get_center().get_y() + (direction.get_y())*POWER_SPEED);
    }
    else{
        if (shape.get_center().get_x() < 0){
            shape.set_center(600, shape.get_center().get_y());
        }
        else if (shape.get_center().get_x() > 600){
            shape.set_center(0, shape.get_center().get_y());
        }
        else if (shape.get_center().get_y() < 0){
            shape.set_center(shape.get_center().get_x(), 600);
        }
        else{
            shape.set_center(shape.get_center().get_x(), 0);
        }
    }
}

bool Powerup::detectCollision(Ship &sIn) const{
    Triangle_Coord triangle = sIn.getTriangle();
    if (shape.is_overlapping(triangle.get_tip().get_x(), triangle.get_tip().get_y()) ||
        shape.is_overlapping(triangle.get_bl().get_x(), triangle.get_bl().get_y()) ||
        shape.is_overlapping(triangle.get_br().get_x(), triangle.get_br().get_y())){
        return true;
    } else {
        return false;
    }
}












