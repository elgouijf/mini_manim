import numpy as np
class Mobjects:
    def __init__(self):
        # start with no poinst (i.e 0 rows)
        self.ponints = np.zeros((0,2))

        self.fill_color = None
        self.stroke_color = None

        self.transform_matrix = np.identity(3) # we'll be using homogenous coordinates

        self.submobjects = [] # object starts with no children

        self.name = self.__class__.__name__ # object gets Mobjetc as a name before it gets specified

    def get_center(self):
        return np.mean(self.points, axis = 0) # axis = 0 => mean over columns
    
    def move_to(self, x, y):

        center = self.get_center() #get the center of the object
        tx = x - center[0]
        ty = y - center[1]

        translation_matrix = np.array([[1, 0, tx],
                                        [0, 1, ty],
                                        [0, 0, 1]]
                                       )
        # Convert points to homogenous coordinates
        n_rows = self.points.shape[0] # points is of shape (N,2)

        homogenous_surplus = np.ones((n_rows,1))
        points_homogenous = np.hstack([self.points, homogenous_surplus]) # points_homogenous is of shape (N,3)

        transleted_points_homogenous = (translation_matrix @ points_homogenous.T).T # transleted points is of shape (N,3)
        # eliminate homogenous coodinates
        self.points = transleted_points_homogenous[:, :2] 


        # update tranform_matrix
        self.transform_matrix = translation_matrix @ self.transform_matrix

    def scale(self, s):
        center = self.get_center()
        self.points = (self.points - center)*s + center
