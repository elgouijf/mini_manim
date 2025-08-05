import numpy as np
from ..config import *
from math import cos, sin, tan, pi
from ..utilities.bezier import *

class Mobject:
    def __init__(self, **settings):
        # start with no poinst (i.e 0 rows)
        self.points = np.zeros((0,2))

        self.stroke_color = settings.get("stroke_color", DEFAULT_STROKE_COLOR)
        self.stroke_width = settings.get("stroke_width", DEFAULT_STROKE_WIDTH)
        self.fill_color = settings.get("fill_color", DEFAULT_FILL_COLOR)
        self.fill_opacity = settings.get("fill_opacity", DEFAULT_FILL_OPACITY)
        self.transform_matrix = np.identity(3) # we'll be using homogenous coordinates

        self.submobjects = [] # object starts with no children

        self.name = self.__class__.__name__ # object gets the name of the runtime class of the instance (Mobject or a subclass) 

    def get_center(self):
        return np.mean(self.points, axis = 0) # axis = 0 => mean over columns
    
    def get_translation_matrix(self, dx, dy):
        return np.array([
                [1, 0, dx],
                [0, 1, dy],
                [0, 0, 1]])
    
    def move_to(self, x, y):

        center = self.get_center() #get the center of the object
        dx = x - center[0]
        dy = y - center[1]

        translation_matrix = self.get_translation_matrix(dx, dy)
        """# Convert points to homogenous coordinates
        n_rows = self.points.shape[0] # points is of shape (N,2)

        homogenous_surplus = np.ones((n_rows,1))
        points_homogenous = np.hstack([self.points, homogenous_surplus]) # points_homogenous is of shape (N,3)

        transleted_points_homogenous = (translation_matrix @ points_homogenous.T).T # transleted points is of shape (N,3)
        # eliminate homogenous coodinates
        self.points = transleted_points_homogenous[:, :2]  """

        # update tranform_matrix
        self.transform_matrix = translation_matrix @ self.transform_matrix


    def scale(self, s):
        center = self.get_center()

        T1 = self.get_translation_matrix(*(-center)) # This grants us a shift-back to the origin before scaling
        S = np.array(              [[s, 0, 0],
                                   [0, s, 0],
                                   [0, 0, 1]])
        
        T2 = self.get_translation_matrix(*(center)) # with this we translate our object so it is again centered around center
        
        scaling_matrix = T2 @ S @ T1

        """ 
        n_rows = self.points.shape[0] # points is of shape (N,2)
        homogenous_surplus = np.ones((n_rows,1))
        shifted_points = self.points - center
        points_homogenous = np.hstack([shifted_points, homogenous_surplus]) # points_homogenous is of shape (N,3)

        scaled_points_homogenous = (scaling_matrix @ points_homogenous.T).T
        self.points = scaled_points_homogenous[:, :2] + center """

        
        self.transform_matrix = scaling_matrix @ self.transform_matrix

    def rotate(self, theta):

        center = self.get_center()
        T1 = self.get_translation_matrix(*(-center)) # This grants us a shift-back to the origin before scaling

        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0             ,0            , 1]])
        T2 = self.get_translation_matrix(*(center)) # with this we translate our object so it is again centered around center

        rotation_matrix = T2 @ R@ T1
        self.transform_matrix = rotation_matrix @ self.transform_matrix


    def apply_transform(self):
        n_rows = self.points.shape[0] # points is of shape (N,2)
        homogenous_surplus = np.ones((n_rows,1))
        points_homogenous = np.hstack([self.points, homogenous_surplus]) # points_homogenous is of shape (N,3)
        tarsnformed_points_homogenous = (self.transform_matrix @ points_homogenous.T).T
        self.points = tarsnformed_points_homogenous[:, :2]
        # reset transform_matrix
        self.transform_matrix = np.identity(3)
    

class Group(Mobject):
    def __init__(self, *mobjects):
        super().__init__() # Group being a subclass of Mobject it'll have the same attributes intialzation
        self.submobjects = list(mobjects)

    def add_objs(self, *mobjects):
        self.submobjects.extend(mobjects)

    def remove(self, mobject):
        try:
            self.submobjects.remove(mobject)
        except ValueError:
            print(f"Warning: {mobject} is not a part of {self.name}")

    def get_center(self):
        # get the group's center
        points = np.vstack([mobj.points for mobj in self.submobjects])
        return np.mean(points, axis=0)

    def apply_transform(self):
        for mobj in self.submobjects:
            mobj.transform_matrix = self.transform_matrix @ mobj.transform_matrix
            mobj.apply_transofrm()
        # reset transform matrix
        self.transform_matrix = np.identity(3)


class VMobject(Mobject):
    def __init__(self, **settings):
        super().__init__(**settings)
        self.closed = False # This tells the renderer if the shape is closed (A->B->C->A) or open (A->B->C)
        # self.subpaths = {path : closed_bool} won't work since numpy array are not hachable (because they're mutable)
        #same goes with named tuples so we'll just se parallel lists
        self.subpaths = []
        self.closed_subpaths = []

    def set_corners(self, corners):
        self.points = np.array(corners) # just to be safe we'll apply an np.array on the corners iterable

    def add_line(self, point):
        # this will connect the last point of our VMobject to anew one creating a new line
        self.points = np.vstack([self.points, point])

    def close(self):
        self.closed = True

    def add_subpaths(self, points, closed = False):
        self.subpaths.append(np.array(points))
        self.closed_subpaths.append(closed)

    def interpolate(self, vomb1, vomb2, t):
        "nterpolation will help us cover whatever segment in a space to create an animating effect"
        self.points = (1 - t)*vomb1.points + t*vomb2.points

    
    def set_fill_color(self, color):
        self.fill_color = color


    def set_stroke_color(self, color):
        self.stroke_color = color

class Line(VMobject):
    """
    a class for dealing with lines
    """

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

class Circle(VMobject):

    def __init__(self, n_segments = 4, center = np.array[0,0], radius = 1.0, n_bezier_points = 59, **settings):
        super().__init__(**settings) # this will take care of all settings
        self.radius = radius
        self.close()
        if n_segments % 4 != 0:
            raise ValueError("n_segment must be a multiple of 4 (4 included)")
        elif n_segments:
            self.n_segments = n_segments #setting the number of points the circle is made of
        self.generate_circle(center, n_bezier_points)

    
        
    def generate_circle(self, center, n_bezier_points):
        radius = self.radius
        n_segments = self.n_segments
        theta = (2*pi)/n_segments

        if n_segments == 4:
            kappa = 0.5522847498 # this is (4*radius*tan((pi/2)/4))/3
        elif theta < pi/12 :
            # for a small theta we can go to the first order of tan and still maintain a good quality
            kappa = (radius*theta) / 3
        else:
            # we use the midpoint method/ cross product to determine a good estimation for kappa
            kappa = (4*radius*tan(theta/4))/3

        
        circle = []
        for i in range(n_segments):
            start_angle = theta*i
            end_angle = theta*(i + 1)

            # Compute enpoints of the arc
            p0 = np.array([radius*cos(start_angle), radius*sin(start_angle)])
            p3 = np.array([radius*cos(end_angle), radius*sin(end_angle)])

            # Compute Tangents(Direction vectors)
            T0 = np.array([-radius*sin(start_angle), radius*cos(start_angle)])
            T1 = np.array([-radius*sin(end_angle), -radius*cos(end_angle)])

            # Compute control points
            p1 = p0 + kappa*T0/np.linalg.norm(T0) #we're sure that the direction will never be 0 because it's a circle
            p2 = p3 - kappa*T1/np.linalg.norm(T1)

            for t in np.linspace(0, 1, n_bezier_points):
                point = bezier_cubic(t, p0, p1, p2, p3)
                circle.append(point + center)

        self.add_subpaths(circle, closed=True)


class Square(VMobject):
    def __init__(self,center, side_len, **settings):
        super().__init__(**settings)
        self.close()
        self.generate_square(center, side_len)
    
    def generate_square(self, center, side_len = 2.0):
        half = side_len/2
        # Define corners
        corners = [
        np.array([-half, half]) + center,
        np.array([half, half]) + center,
        np.array([half, -half]) + center,
        np.array([-half, -half]) + center
        ]
        self.set_corners(corners)


class Polygon(VMobject):
    def __init__(self, center=np.array([0,0]), radius = 2.0, n = 8.0 , **settings):
        super().__init__(**settings)
        self.close()
        self.generate_polygon(center, n, radius)
        
    def generate_polygon(self, center, n, radius):
        corners = [np.array([0, radius])]
        theta = 2*pi/n
        Rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)],])
        
        for i in range(1, n):
            corners.append = (Rotation @ corners[i - 1].T).T
        
        for i in range(len(corners)):
            corners[i] += center
            
        self.set_corners(corners)
