//
//  Shape.cpp
//  FinalProject
//
//  Created by Adam Barson on 4/19/17.
//  Copyright © 2017 Adam Barson. All rights reserved.
//

#include "Shape.hpp"
#include "Space.hpp"
Shape::Shape(): area(0), perimeter(0) {
    fill.red = fill.green = fill.blue = 0;
}

Shape::Shape(double a, double p, color f): area(a), perimeter(p), fill(f) {
}

Shape::Shape(color f) : area(0), perimeter(0), fill(f) {
}

double Shape::get_area() const {
    return area;
}

double Shape::get_perimeter() const {
    return perimeter;
}

color Shape::get_fill() const {
    return fill;
}

void Shape::set_color(color c) {
    fill = c;
}

void Shape::set_color(double r, double g, double b) {
    fill = {r, g, b};
    /*
     fill.red = r;
     fill.green = g;
     fill.blue = b;
     */
}

Circle::Circle() : Shape(), radius(0) {
}

Circle::Circle(double r) : Shape() {
    set_radius(r);
}

Circle::Circle(color c) : Shape(c), radius(0) {
}

Circle::Circle(double r, color c) : Shape(c) {
    set_radius(r);
}

double Circle::get_radius() const {
    return radius;
}

void Circle::set_radius(double r) {
    radius = r;
    if (r < 0) {
        radius = 0;
    }
    calc_area_peri();
}

// This method is to be completed.
void Circle::draw() {
    cout << "Drawing a circle" << endl;
}

void Circle::calc_area_peri() {
    area = M_PI * radius * radius;
    perimeter = 2.0 * M_PI * radius;
}

Point2D::Point2D() : x(0), y(0) {
}

Point2D::Point2D(double x_in, double y_in) : x(x_in), y(y_in) {
}

double Point2D::get_x() const {
    return x;
}

double Point2D::get_y() const {
    return y;
}

void Point2D::set_x(double x_in) {
    x = x_in;
}

void Point2D::set_y(double y_in) {
    y = y_in;
}

void Point2D::set_point(double x_in, double y_in) {
    x = x_in;
    y = y_in;
}

void Point2D::move_point(double x_offset, double y_offset) {
    x += x_offset;
    y += y_offset;
}

bool operator == (Point2D &lhs, Point2D &rhs){
    double difference_x = lhs.get_x() - rhs.get_x();
    double difference_y = lhs.get_y() - rhs.get_y();
    return abs(difference_x) < 0.001 && abs(difference_y) < 0.001;
}

Triangle_Coord::Triangle_Coord(): Shape(){
    size = 10;
    set_location(0, 0);
    angle = 0;
}
Triangle_Coord::Triangle_Coord(double center_x, double center_y): Shape(){
    size = 10;
    set_location(center_x, center_y);
    angle = 0;
}

Triangle_Coord::Triangle_Coord(double center_x, double center_y, double rIn): Shape(){
    size = rIn;
    set_location(center_x, center_y);
    set_color(1, 1, 0);
    angle = 0;
}

Triangle_Coord::Triangle_Coord(double center_x, double center_y, color c) : Shape(c){
    size = 10;
    set_location(center_x, center_y);
    angle = 0;
}

Point2D Triangle_Coord::get_center() const{
    return center;
}

void Triangle_Coord::set_center(Point2D cIn){
    center = cIn;
}

Point2D Triangle_Coord::get_tip() const{
    return tip;
}

void Triangle_Coord::set_tip(Point2D tIn){
    tip = tIn;
}

Point2D Triangle_Coord::get_bl() const{
    return bottom_left;
}

void Triangle_Coord::set_bl(Point2D blIn){
    bottom_left = blIn;
}

Point2D Triangle_Coord::get_br() const{
    return bottom_right;
}

void Triangle_Coord::set_br(Point2D brIn){
    bottom_right = brIn;
}

double Triangle_Coord::get_angle() const{
    return angle;
}

void Triangle_Coord::set_angle(double aIn){
    angle = aIn;
}
void Triangle_Coord::calc_area_peri(){
    //irrelevant
}

void Triangle_Coord::set_location(double center_x, double center_y){
    set_center(Point2D(center_x, center_y));
    tip = Point2D(center_x, center_y - size);
    bottom_left = Point2D(center_x - size, center_y + size + 10);
    bottom_right = Point2D(center_x + size, center_y + size + 10);
    angle = 0;
}

void Triangle_Coord::rotate(int x){
    double rad = x / 180.0 * PI;
    
    double rel_tip_x = tip.get_x() - get_center().get_x();
    double rel_tip_y = tip.get_y() - get_center().get_y();
    double tip_x = (cos(rad)*(rel_tip_x) - sin(rad)*(rel_tip_y));
    double tip_y = (sin(rad)*(rel_tip_x) + cos(rad)*(rel_tip_y));
    tip = Point2D(center.get_x() + tip_x, center.get_y() + tip_y);
    
    double rel_bl_x = bottom_left.get_x() - get_center().get_x();
    double rel_bl_y = bottom_left.get_y() - get_center().get_y();
    double bl_x = (cos(rad)*(rel_bl_x) - sin(rad)*(rel_bl_y));
    double bl_y = (sin(rad)*(rel_bl_x) + cos(rad)*(rel_bl_y));
    bottom_left = Point2D(center.get_x() + bl_x, center.get_y() + bl_y);
    
    double rel_br_x = bottom_right.get_x() - get_center().get_x();
    double rel_br_y = bottom_right.get_y() - get_center().get_y();
    double br_x = (cos(rad)*(rel_br_x) - sin(rad)*(rel_br_y));
    double br_y = (sin(rad)*(rel_br_x) + cos(rad)*(rel_br_y));
    bottom_right = Point2D(center.get_x() + br_x, center.get_y() + br_y);
}

Point2D Triangle_Coord::get_rel_tip() const{
    double rel_tip_x = tip.get_x() - get_center().get_x();
    double rel_tip_y = tip.get_y() - get_center().get_y();
    return Point2D(rel_tip_x, rel_tip_y);
}

Point2D Triangle_Coord::get_rel_bl() const{
    double rel_bl_x = bottom_left.get_x() - get_center().get_x();
    double rel_bl_y = bottom_left.get_y() - get_center().get_y();
    return Point2D(rel_bl_x, rel_bl_y);
}

Point2D Triangle_Coord::get_rel_br() const{
    double rel_br_x = bottom_right.get_x() - get_center().get_x();
    double rel_br_y = bottom_right.get_y() - get_center().get_y();
    return Point2D(rel_br_x, rel_br_y);
}

void Triangle_Coord::draw() {
    glBegin(GL_TRIANGLES);
    //set fill color
    glColor3f(fill.red, fill.green, fill.blue);
    //draw center point
    
    glVertex2f(get_tip().get_x(), get_tip().get_y());
    glVertex2f(bottom_left.get_x(), bottom_left.get_y());
    glVertex2f(bottom_right.get_x(), bottom_right.get_y());
    glEnd(); //End triangle coordinates
}



Circle_Coord::Circle_Coord() {
    center = Point2D();
}

Circle_Coord::Circle_Coord(double rIn): Circle(rIn) {
    center = Point2D();
}


Point2D Circle_Coord::get_center() const {
    return center;
}

int Circle_Coord::get_center_x() const {
    return center.get_x();
}

int Circle_Coord::get_center_y() const {
    return center.get_y();
}

void Circle_Coord::set_center(Point2D center_in) {
    center = center_in;
}

void Circle_Coord::set_center(double x_in, double y_in) {
    center = Point2D(x_in, y_in);
}

void Circle_Coord::set_outside_color(double r, double g, double b) {
    outside = {r, g, b};
}

double Point2D::distance_between(double x1, double y1, double x2, double y2){
    return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}

bool Circle_Coord::is_overlapping(const Circle_Coord &c) const{
    return (Point2D::distance_between(get_center_x(), get_center_y(), c.get_center_x(), c.get_center_y()) <= radius + c.get_radius());
}

bool Circle_Coord::is_overlapping(double x_in, double y_in) const{
    Circle_Coord temp;
    temp.set_center(x_in, y_in);
    return is_overlapping(temp);
}

bool Circle_Coord::is_overlapping(double x_in1, double y_in1, double x_in2, double y_in2) const{
    // the center y is in between y_in1 and y_in2
    if (x_in1 == x_in2 && ((get_center_y() > y_in2) == (get_center_y() < y_in1) &&
                           (x_in1 >= get_center_x() - radius) && x_in1 <= get_center_x() + radius)){
        return true;
        //horizontal line segment
    } else if (y_in1 == y_in2 && ((get_center_x() > x_in2) == (get_center_x() < x_in1) &&
                                  (y_in1 >= get_center_y() - radius) && y_in1 <= get_center_y() + radius)){
        return true;
    }
    return false;
}

void Circle_Coord::draw() {
    glBegin(GL_TRIANGLE_FAN);
    // set fill color
    glColor3f(fill.red, fill.green, fill.blue);
    // draw center point
    glVertex2f(get_center_x(), get_center_y());
    // set outside color
    glColor3f(outside.red, outside.green, outside.blue);
    
    // draw outside points
    for (int i = 0; i <= 360; ++i) {
        glVertex2f(get_center_x() + radius * cos(i * M_PI / 180.0), get_center_y() + radius * sin(i * M_PI / 180.0));
    }
    glEnd();
}
