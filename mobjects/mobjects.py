import numpy as np
from collections import namedtuple
class Mobject:
    def __init__(self):
        # start with no poinst (i.e 0 rows)
        self.points = np.zeros((0,2))

        self.fill_color = None
        self.stroke_color = None

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
    def __init__(self):
        super().__init__()
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
        self.close = True

    def add_subpath(self, point, closed = False):
        self.subpaths.append(np.array(point))
        self.closed_subpaths.append(closed)
