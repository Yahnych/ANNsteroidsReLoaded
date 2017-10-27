//
//  Shape.hpp
//  FinalProject
//
//  Created by Adam Barson on 4/19/17.
//  Copyright © 2017 Adam Barson. All rights reserved.
//

#ifndef Shape_hpp
#define Shape_hpp

#include <cmath>
#include <iostream>

#define PI acos(-1);
using namespace std;

struct color {
    double red;
    double green;
    double blue;
};

class Shape {
public:
    Shape();
    Shape(double a, double p, color f);
    Shape(color f);
    
    virtual double get_area() const;
    virtual double get_perimeter() const;
    virtual color get_fill() const;
    
    virtual void set_color(color c);
    virtual void set_color(double r, double g, double b);
    
    /**
     * Requires: nothing
     * Modifies: nothing (canvas to be drawn on)
     * Effects: draws the shape on the canvas
     */
    virtual void draw() = 0;
    
protected:
    double area;
    double perimeter;
    
    virtual void calc_area_peri() = 0;
    color fill;
};


class Circle : public Shape {
public:
    Circle();
    Circle(double r);
    Circle(color c);
    Circle(double r, color c);
    
    double get_radius() const;
    
    void set_radius(double r);
    
    // override draw method from Shape
    virtual void draw() override;
protected:
    double radius;
private:
    virtual void calc_area_peri() override;
};

class Point2D {
public:
    Point2D();
    Point2D(double x_in, double y_in);
    
    double get_x() const;
    double get_y() const;
    
    void set_x(double x_in);
    void set_y(double y_in);
    void set_point(double x_in, double y_in);
    
    void move_point(double x_offset, double y_offset);
    
    static double distance_between(double x1, double y1, double x2, double y2);
    
    friend bool operator == (Point2D &lhs,  Point2D &rhs);
private:
    double x;
    double y;
};

class Triangle_Coord : public Shape {
public:
    Triangle_Coord();
    Triangle_Coord(Point2D tIn);
    Triangle_Coord(double tip_x, double tip_y);
    Triangle_Coord(double tip_x, double tip_y, double rIn);
    Triangle_Coord(double tip_x, double tip_y, color c);
    
    Point2D get_center() const;
    void set_center(Point2D cIn);
    
    Point2D get_tip() const;
    void set_tip(Point2D tIn);
    
    Point2D get_bl() const;
    void set_bl(Point2D blIn);
    
    Point2D get_br() const;
    void set_br(Point2D brIn);
    
    double get_angle() const;
    void set_angle(double aIn);
    
    void set_location(double center_x, double center_y);
    
    void rotate(int x);
    
    Point2D get_rel_tip() const;
    Point2D get_rel_br() const;
    Point2D get_rel_bl() const;
    
    void updateMove();
    virtual void draw() override;
protected:
    Point2D center;
    Point2D tip;
    Point2D bottom_right;
    Point2D bottom_left;
    double angle;
    double size;
private:
    virtual void calc_area_peri() override;
};



class Circle_Coord : public Circle {
public:
    Circle_Coord();
    
    Circle_Coord(double rIn);
    
    Point2D get_center() const;
    int get_center_x() const;
    int get_center_y() const;
    
    void set_center(Point2D center_in);
    void set_center(double x_in, double y_in);
    
    virtual void set_outside_color(double r, double g, double b);
    
    virtual void draw() override;
    
    bool is_overlapping(double x_in, double y_in) const;
    
    bool is_overlapping(const Circle_Coord &c) const;
    
    bool is_overlapping(double x_in1, double y_in1, double x_in2, double y_in2) const;
    
private:
    Point2D center;
    color outside;
};



#endif /* Shape_hpp */
