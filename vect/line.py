import os

import sys
import cairo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mobjects.mobjects as mbj
import numpy as np

class Line(mbj.VMobject):
    """"
    a class for dealing with lines""""
    def __init__(self,start_pt,end_pt):
        super().__init__()
        self.start_pt = start_pt
        self.end_pt = end_pt
    def tangent(self):
        #returning the slope of the line
        dx = self.end_pt[0] - self.start_pt[0]
        dy = self.end_pt[1] - self.start_pt[1]
        if dx:
            return dy/dx
        else:
            return np.inf
    def extend(self,x_lim_left,y_lim_down,x_lim_right,y_lim_up):
        if self.tangent == np.inf:#vertical line
            self.start_pt = np.array([self.start_pt[0],y_lim_down])
            self.end_pt = np.array([self.end_pt[0],y_lim_up])
            return
        a= self.tangent()
        b = self.start_pt[1] - a*self.start_pt[0]
        if a >0 :
            #line equation goes y = tangent*x + (y0 - tangent*x0)
            if a*x_lim_right + b > y_lim_up:
                #line will pass by top
                self.end_pt = np.array([(y_lim_up-b)/a,y_lim_up])
            else:
               self.end_pt = np.array([x_lim_right,a*x_lim_right + b]) 
            if a*x_lim_left + b < y_lim_down:
                #line will pass by top
                self.start_pt = np.array([(y_lim_down-b)/a,y_lim_down])
            else:
               self.start_pt = np.array([x_lim_left,a*x_lim_left + b]) 
        elif a <0 :
        #line equation goes y = tangent*x + (y0 - tangent*x0)
            if a*x_lim_right + b< y_lim_down:
                #line will pass by top
                self.end_pt = np.array([(y_lim_down-b)/a,y_lim_down])
            else:
                self.end_pt = np.array([x_lim_right,a*x_lim_right + b]) 
            if a*x_lim_left + b > y_lim_up:
                #line will pass by top
                self.start_pt = np.array([(y_lim_up-b)/a,y_lim_up])
            else:
               self.start_pt = np.array([x_lim_left,a*x_lim_left + b])
        else : 
            self.start_pt = np.array([x_lim_left,b])
            self.end_pt = np.array([x_lim_right,b])
    def is_valid_line(self):
        if self.start_pt == self.end_pt:
            return False
        for point in self.points:
            if not self.start_pt[0] <= point[0] <= self.end_pt[0]:
                #if we want a segment
                return False
            a = self.tangent()
            if a != np.inf:#the other case is checked
                dx = point[0] - self.start_pt[0]
                dy = point[1] - self.start_pt[1]
                if dy/dx != a:
                    return False
        return True
class Vector2D(mbj.Mobject):
    pass


           

