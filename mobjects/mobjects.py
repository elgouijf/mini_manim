import numpy as np
from config import *
from math import cos, sin, tan, pi
from utilities.bezier import *
import inspect
import cairo
from copy import deepcopy
from utilities.color import *
def resize_preserving_order(points, new_length):
    """
    Given an array of points, return a new array with exactly new_length points,
    interpolated evenly along the original order.
    """
    old_length = len(points)
    if old_length == 0:
        return np.zeros((new_length, 2))
    elif old_length == new_length:
        return points.copy()

    # Compute fractional indices to sample along the original points
    idxs = np.linspace(0, old_length - 1, new_length)
    new_points = []

    for idx in idxs:
        low = int(np.floor(idx))
        high = int(np.ceil(idx))
        t = idx - low
        if high >= old_length:
            high = old_length - 1
        pt = (1 - t) * points[low] + t * points[high]
        new_points.append(pt)
    
    return np.array(new_points)


class Mobject:
    def __init__(self, **settings):
        # start with no poinst (i.e 0 rows)
        self.points = np.zeros((0,2))

        self.stroke_color = (settings.get("stroke_color", DEFAULT_STROKE_COLOR)).get_rgb()  # Convert color to RGB tuple
        self.stroke_width = settings.get("stroke_width", DEFAULT_STROKE_WIDTH)
        self.fill_color = (settings.get("fill_color", DEFAULT_FILL_COLOR)).get_rgb()  # Convert color to RGB tuple
        self.fill_opacity = settings.get("fill_opacity", DEFAULT_FILL_OPACITY)
        self.stroke_opacity = settings.get("stroke_opacity", DEFAULT_STROKE_OPACITY)
        self.stroke_width = settings.get("stroke_width", DEFAULT_STROKE_WIDTH)
        self.transform_matrix = np.identity(3) # we'll be using homogenous coordinates

        self.submobjects = [] # object starts with no children

        self.name = self.__class__.__name__ # object gets the name of the runtime class of the instance (Mobject or a subclass)

    def interpolate(self, mob1, mob2, t):
        "nterpolation will help us cover whatever segment in a space to create an animating effect"
        self.points = (1 - t)*mob1.points + t*mob2.points 

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
        T1 = self.get_translation_matrix(*(-center)) # This grants us a shift-back to the origin before rotating

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

    def set_fill_opacity(self, opacity):
        self.fill_opacity = opacity

    def set_stroke_opacity(self, opacity):
        self.stroke_opacity = opacity
        
        
    def set_fill_color(self, color):
        self.fill_color =  ensure_rgb(color) # Convert color to RGB tuple

    def set_stroke_color(self, color):
        self.stroke_color = ensure_rgb(color)  # Convert color to RGB tuple


    def set_stroke(self, color=None, width=None, opacity=None):
        """Set the stroke (outline) style of the shape."""
        if color is not None:
            self.set_stroke_color(color)
        if width is not None:
            self.stroke_width = width
        if opacity is not None:
            self.stroke_opacity = opacity
        return self
    
    def set_points(self, points):
        self.points = deepcopy(points)  # Use deepcopy to ensure the points are not modified outside this class

    def add_updater(self, update_func): #functions are objects too (in C it would've been a callback function but it is simpler in python)
        self.updaters.append(update_func)

    def remove_updater(self, update_func):
        if update_func in self.updaters:
            self.updaters.remove(update_func)

    def run_updates(self, **kwargs):
        """ Runs all the updaters in the order they were added
        """        
        for updater in self.updaters:
            signature = inspect.signature(updater)
            bound_args = {}
            # Always pass self as the first argument
            bound_args['mobj'] = self
            for name, param in signature.parameters.items():
                if name == 'mobj':
                    continue
                if name in kwargs:
                    bound_args[name] = kwargs[name]
                elif param.default is not param.empty:
                    bound_args[name] = param.default
                # else: leave missing, will raise error if required
            updater(**bound_args)
        self.apply_transform() # apply the transformation after all updaters have been run

    def resize_points(self, new_length, resize_func=resize_preserving_order):
        """
        Resize the points array to have exactly new_length points,
        interpolated evenly along the original order.
        """
        self.points = resize_func(self.points, new_length)
        
    def align_points(self, mobject):
        n1 = len(self.points)
        n2 = len(mobject.points)
        max_pts = max(n1, n2)
        for mobj in (self, mobject):
            mobj.resize_points(max_pts, resize_func=resize_preserving_order)
        



class Group(Mobject):
    def __init__(self, *mobjects, **settings):
        super().__init__(**settings) # Group being a subclass of Mobject it'll have the same attributes intialzation
        self.submobjects = list(mobjects)
        self.refresh_points()  # Refresh points after adding mobjects

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
    
    def refresh_points(self):
        """ Refreshes the points of the group by combining the points of all submobjects """
        if self.is_empty():
            self.points = np.zeros((0, 2))
        else:
            self.points = np.vstack([mobj.points for mobj in self.submobjects])

    def move_to(self, x, y):
        for mobj in self.submobjects:
            mobj.move_to(x, y)
        

    def scale(self, s):
        for mobj in self.submobjects:
            mobj.scale(s)


    def rotate(self, theta):
        for mobj in self.submobjects:
            mobj.rotate(theta)


    def set_opacity(self, opacity):
        for mobj in self.submobjects:
            mobj.set_opacity(opacity)

    def set_fill_color(self, color):
        for mobj in self.submobjects:
            mobj.set_fill_color(color)
    
    def set_stroke_color(self, color):
        for mobj in self.submobjects:
            mobj.set_stroke_color(color)

    def add_updater(self, update_func):
        """ Adds an updater to all submobjects in the group """
        for mobj in self.submobjects:
            mobj.add_updater(update_func)
    
    def remove_updater(self, update_func):
        """ Removes an updater from all submobjects in the group """
        for mobj in self.submobjects:
            if update_func in mobj.updaters:
                mobj.remove_updater(update_func)

    def run_updates(self, *args, **kwargs):
        """ Runs all the updaters in the order they were added for all submobjects """
        for mobj in self.submobjects:
            mobj.run_updates(*args, **kwargs)
        self.refresh_points()

    def is_empty(self):
        """ Checks if the group has no submobjects """
        return len(self.submobjects) == 0
        
    def apply_transform(self):
        for mobj in self.submobjects:
            mobj.transform_matrix = self.transform_matrix @ mobj.transform_matrix
            mobj.apply_transofrm()
        # Refresh points after applying transformations
        self.refresh_points()  
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

    def open(self):
        self.closed = False

    def add_subpaths(self, points, closed=False):
        points = np.array(points)
        self.subpaths.append(points)
        self.closed_subpaths.append(closed)

        # Flatten all subpaths into self.points for transforms/animations
        if len(self.subpaths) > 0:
            self.points = np.vstack(self.subpaths)
        else:
            self.points = np.zeros((0, 2))

    def get_subpaths(self):
        """ Returns the subpaths of the VMobject """
        return self.subpaths, self.closed_subpaths 
    

    def apply_transform(self):
        """ Applies the transformation matrix to the points of the VMobject """
        # Convert points to homogenous coordinates
        n_rows = self.points.shape[0] # points is of shape (N,2)
        homogenous_surplus = np.ones((n_rows,1))
        points_homogenous = np.hstack([self.points, homogenous_surplus]) # points_homogenous is of shape (N,3)
        tarsnformed_points_homogenous = (self.transform_matrix @ points_homogenous.T).T
        self.points = tarsnformed_points_homogenous[:, :2]

        # Update subpaths
        for i in range(len(self.subpaths)):
            subpath = self.subpaths[i]
            n_rows = subpath.shape[0]
            homogenous_surplus = np.ones((n_rows, 1))
            subpath_homogenous = np.hstack([subpath, homogenous_surplus])
            transformed_subpath_homogenous = (self.transform_matrix @ subpath_homogenous.T).T
            self.subpaths[i] = transformed_subpath_homogenous[:, :2]
        # reset transform_matrix
        self.transform_matrix = np.identity(3)



class VGroup(Group, VMobject): # Group must come first for methods to follow MRO (Method Resolution Order)
    def __init__(self, *vmobjects, **settings):
        Group.__init__(self, *vmobjects, **settings) # initialize the group part
        VMobject.__init__(self, **settings) # initialize the VMobject part
        self.refresh_points()  # Refresh points after adding mobjects
    
    def close(self):
        self.closed = True

    def open(self):
        self.closed = False
    
    def refresh_subpaths(self):
        """ Refreshes the subpaths of the VGroup by combining the subpaths of all submobjects """
        self.subpaths = []
        self.closed_subpaths = []
        for mobj in self.submobjects:
            subpaths, closed_subpaths = mobj.get_subpaths()
            self.subpaths.extend(subpaths)
            self.closed_subpaths.extend(closed_subpaths)

    def run_updates(self, *args, **kwargs):
        """ Runs all the updaters in the order they were added for all submobjects """
        for vmobj in self.submobjects:
            vmobj.run_updates(*args, **kwargs)
        self.refresh_points()
        self.refresh_subpaths()  # Refresh subpaths after running updates

    def apply_transform(self):
        """ Applies the transformation matrix to the points of the VGroup """
        for vmobject in self.submobjects:
            vmobject.transform_matrix = self.transform_matrix @ vmobject.transform_matrix
            vmobject.apply_transform()  # Apply the transformation to the VMobject part
        # reset group's transform matrix after all submobjects are updated
        self.transform_matrix = np.identity(3)
        self.refresh_points()
        self.refresh_subpaths()



class Point(Mobject):
    def __init__(self, artificial_width = 1e-5, artificial_height = 1e-5, position = np.array([0,0]), **settings):
        super().__init__(**settings)
        self.artificial_width = artificial_width
        self.artificial_height = artificial_height
        self.set_position(position)

    def set_position(self, position):
        """ Sets the position of the point """
        self.set_points(np.array([position]))


class Dot(Point):
    def __init__(self, radius = 1e-5, artificial_width = 1e-5, artificial_height = 1e-5, position = np.array([0,0]), **settings):
        super().__init__(artificial_width, artificial_height, position, **settings)
        self.radius = radius
        self.generate_dot()

    def generate_dot(self):
        """ Generates the dot by creating a circle with the given radius """
        circle = Circle(radius=self.radius, center=self.points[0]) # n_segments will be at default since a Dot is small enough for it to work
        self.set_points(circle.points)  # Set the points of the dot to the points of the circle
        
class GlowingDot(Dot, VGroup):
    def __init__(self, radius = 1e-5, position = np.array([0,0]), glow_radius = 0.2, **settings):
        super().__init__(radius=radius, position=position, **settings)
        self.glow_radius = glow_radius
        self.generate_glowing_dot()

    def generate_glowing_dot(self):
        """ Generates the glowing dot by creating a circle with the given glow radius """
        glow_extend = self.glow_radius - self.radius
        glow_layers = max(1, int(glow_extend/2))  # Number of layers to create a glow effect
        print(self.fill_color)
        fill_color = getattr(self, "fill_color", (1, 1, 1))
        fill_opacity = getattr(self, "fill_opacity", 1.0)
        fill_opacity = fill_opacity*(1 - 0.4)
        
        for i in reversed(range(glow_layers)): # Create a series of circles to simulate a glow effect
            circle = Circle(radius=self.radius + (i/glow_layers)*glow_extend, center=self.points[0])
            circle.set_fill_color(fill_color)
            circle.set_stroke(width= 0, opacity= 0)
            circle.set_fill_opacity(fill_opacity * (1 - i / glow_layers)**4)  # Fade out the glow effect

            """ print("Circle color",circle.fill_color)
            print("Circle stroke width",circle.stroke_width) """
            
            if circle.fill_opacity > 0.01:
                print("Circle opacity", circle.fill_opacity)
                for i in range(2):
                    self.add_objs(circle)

            else: 
                fixed_circle = Circle(radius=self.radius + (i/glow_layers)*glow_extend, center=self.points[0])
                fixed_circle.set_fill_color(fill_color)
                fixed_circle.set_stroke(width= 0, opacity= 0)
                fixed_circle.set_fill_opacity(0.01) 
                print("Circle opacity", 0.01)

                for i in range(2):
                    self.add_objs(fixed_circle)
               

class FunctionGraph(VMobject):
    """
    a class for dealing with function graphs
    """

    def __init__(self, func, x_lim_left, x_lim_right, closed = False, n_points = 100, n_bezier_points = 59, error = 0.01, **settings):
        super().__init__(**settings)
        self.func = func
        self.x_lim_left = x_lim_left
        self.x_lim_right = x_lim_right
        self.n_points = n_points
        self.error = error
        self.closed = closed
        self.generate_graph(n_bezier_points)

                
    

    def generate_graph(self, n_bezier_points):

        curve = []
        x_values = np.linspace(self.x_lim_left, self.x_lim_right, self.n_points)
        y_values = self.func(x_values)
        points = np.column_stack((x_values, y_values))
        n = len(x_values)
        for i in range(n - 1):
            p0 = points[i]
            p3 = points[(i + 1)]
            p1, p2 = control_points(self.func, p0, p3, self.error)

            for t in np.linspace(0, 1, n_bezier_points):
                point = bezier_cubic(t, p0, p1, p2, p3)
                curve.append(point)

        self.add_subpaths(curve, self.closed)
                


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
        if self.tangent() == np.inf:#vertical line
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
                self.end_pt = np.array([(y_lim_down-b) / a,y_lim_down])

            else:
                self.end_pt = np.array([x_lim_right,a * x_lim_right + b]) 

            if a*x_lim_left + b > y_lim_up:
                #line will pass by top
                self.start_pt = np.array([(y_lim_up-b) / a,y_lim_up])

            else:
               self.start_pt = np.array([x_lim_left,a * x_lim_left + b])

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
    
    def grid(WIDTH, HEIGHT,n_lines_x,n_lines_y):
        """ Returns a grid of lines with the given width and height """
        lines = []
        for x in np.linspace(0, WIDTH, n_lines_x):
            lines.append(Line(np.array([x, 0]), np.array([x, HEIGHT])))
        for y in np.linspace(0, HEIGHT, n_lines_y):
            lines.append(Line(np.array([0, y]), np.array([WIDTH, y])))
        return lines
    
    
class Vector2D(VMobject):
    def __init__(self,tip,offset = np.array([0,0])):
        #tip is coordinates o the vector with origin (0,0)
        super().__init__()
        self.tip = tip
        self.offset = offset
    @property#getter ,don't assign
    def x(self):
        return self.tip[0]
    @property
    def y(self):
        return self.tip[1]
    def norm2(self):
        return np.sqrt(self.x**2 + self.y**2)
    def norm1(self):
        return abs(self.x) + abs(self.y)
    def norm3(self):
        return max(self.x,self.y)
    def angle(self):#between -pi (included) and pi
        if self.x>0:
            return np.atan(self.y/self.x)
        elif self.x<0 :
            if self.y>0:
                return np.atan(self.y/self.x) + np.pi
            else:
                return np.atan(self.y/self.x) - np.pi
        else :#null cosine
            if self.y>0:
                return np.pi/2
            else:
                return -np.pi/2
    
 
    def dot_prod(self,other):
        return self.x*other.x  + self.y*other.y
    def __mul__(self,other):
        return  self.x*other.y  - self.y*other.x
    def __repr__(self):
        return f"({self.x},{self.y})"
    
class Arrow2d(Vector2D):
    @classmethod
    def x_axis(x_lim_left,x_lim_right,y_mid):
        return Vector2D(tip=np.array((x_lim_right - x_lim_left,y_mid)),offset=np.array((x_lim_left,0)))
    @classmethod
    def y_axis(y_up,y_down,x_mid):
        return Vector2D(tip=np.array((0,y_up - y_down)),offset=np.array((x_mid,y_down)))
    def geometric_prop(self):
        line = Line(self.points[0],self.points[-1])
        return line.tangent(),line.offset
    


    

class Circle(VMobject):
    def __init__(self, n_segments=4, center=np.array([0.0, 0.0]), radius=1.0, n_bezier_points=60, **settings):
        super().__init__(**settings)
        self.radius = float(radius)
        self.center = np.array(center, dtype=np.float64)
        
        if n_segments % 4 != 0:
            raise ValueError("n_segments must be a multiple of 4")
        self.n_segments = n_segments
        self.generate_circle(n_bezier_points)

    def generate_circle(self, n_bezier_points):
        radius = self.radius
        n_segments = self.n_segments
        theta = 2 * np.pi / n_segments
        # kappa formula for circular arc approximation
        kappa = (4/3) * np.tan(theta / 4)

        points = []
        for i in range(n_segments):
            start_angle = i * theta
            end_angle = (i + 1) * theta

            # endpoints of the cubic
            p0 = np.array([radius * np.cos(start_angle), radius * np.sin(start_angle)], dtype=np.float64)
            p3 = np.array([radius * np.cos(end_angle), radius * np.sin(end_angle)], dtype=np.float64)

            # tangents (directions)
            T0 = np.array([-np.sin(start_angle), np.cos(start_angle)], dtype=np.float64)
            T1 = np.array([-np.sin(end_angle), np.cos(end_angle)], dtype=np.float64)

            # control points
            p1 = p0 + kappa * T0 * radius
            p2 = p3 - kappa * T1 * radius

            # sample points along the cubic
            t_values = np.linspace(0, 1, n_bezier_points, endpoint=False) if i < n_segments - 1 else np.linspace(0, 1, n_bezier_points)
            for t in t_values:
                points.append(bezier_cubic(t, p0, p1, p2, p3) + self.center)

        self.add_subpaths(points, closed=True)


class Square(VMobject):
    def __init__(self, side_len, center= np.array([0,0]), n_points = None, **settings):
        super().__init__(**settings)
        self.close()
        self.side_len = side_len
        self.generate_square(center, self.side_len, n_points)
    
    def generate_square(self, center, side_len = 2.0,n_points = None):
        half = side_len/2
        # Define corners
        corners = [
        np.array([-half, half]) + center,
        np.array([half, half]) + center,
        np.array([half, -half]) + center,
        np.array([-half, -half]) + center
        ]
        
        if n_points is not None:
            points_per_edge = n_points // 4
            all_points = []
            for i in range(4):
                start = corners[i]
                end = corners[(i + 1) % 4]
                for t in np.linspace(0, 1, points_per_edge, endpoint=(i==3)):
                    all_points.append((1-t)*start + t*end)
            all_points.append(all_points[0])  # close path
            self.set_corners(np.array(all_points))
        else:
            corners.append(corners[0])  # close path
            self.set_corners(corners)


class Polygon(VMobject):
    def __init__(self, center=np.array([0,0]), radius = 2.0, n = 8 , **settings):
        super().__init__(**settings)
        self.close()
        self.generate_polygon(center, n, radius)
        
    def generate_polygon(self, center, n, radius):
        corners = [np.array([0.0, float(radius)])]
        theta = 2*pi/n
        Rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)],])
        
        for i in range(1, n):
            corners.append(Rotation @ corners[i - 1])
        
        for i in range(len(corners)):
            corners[i] += center

        self.set_corners(corners)


class Text(Mobject) :
    """" Text Class """
    def __init__(self, text, position, font_size, color) :
        # call the __init__ method of the Mobject class (superclass)
        super().__init__()      
        self.text = text            # a string : the text we want to show
        self.position = position    # the coordinates of the top left corner of the text
        self.font = "Sans"  # default font       
        self.font_size = font_size
        self.text_color = color     # a triplet (r,g,b)
        self.opacity = 1.0             # opacity of the text

    def generate_points(self):
        """ Generates the points of the text using the context's text_path method """
        ctx = self.ctx
        ctx.select_font_face(self.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(self.font_size)
        ctx.set_source_rgba(self.text_color[0], self.text_color[1], self.text_color[2], self.opacity)
        x, y = self.position 
        extents = ctx.font_extents()
        y = y + extents.ascent  # Adjust y position for baseline
        ctx.move_to(x, y)
        ctx.text_path(self.text)#create a path for the text
        path_data = ctx.copy_path()  # Get the path of the text
        points = []
        sub_paths = []
        curr_subpath = []
        prev_point = None

        for type,coords in path_data:
            if type == cairo.PATH_MOVE_TO :#if there is an existingb subpath
                if curr_subpath:
                    sub_paths.append(np.array(curr_subpath))
                    
                    curr_subpath = []
                prev_point = np.array(coords)
                curr_subpath.append(np.array(coords))

            elif type == cairo.PATH_LINE_TO:
                prev_point = np.array(coords)
                curr_subpath.append(np.array(coords))

            elif type == cairo.PATH_CURVE_TO:
                # Bezier curve points are in groups of 3
                p0 = prev_point
                p1 = np.array(coords[:2])
                p2 = np.array(coords[2:4])
                p3 = np.array(coords[4:6])
                bezier_points = bezier_cubic(np.linspace(0, 1, 10), p0, p1, p2, p3)
                curr_subpath.extend(bezier_points)
                prev_point = p3

            elif type == cairo.PATH_CLOSE_PATH:
                #close adn save subpath
                if curr_subpath:
                    sub_paths.append(np.array(curr_subpath))
                    curr_subpath = []
                prev_point = None
                
        if curr_subpath:
            sub_paths.append(np.array(curr_subpath))
        self.subpaths = sub_paths
        # Combine all subpaths into a single array of points
        self.set_points(np.vstack(sub_paths))  # Set the points of the text
