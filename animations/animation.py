import mobjects.mobjects as mbj
from copy import deepcopy
from utilities.rate_functions import *

class Animation :
    def __init__(self, mobject, start_time, duration, rate_function = smooth):
        mobject.apply_transform()  # clear out any pending transforms
        self.mobject = mobject
        self.rate = rate_function
        self.start_time = start_time
        self.end_time = start_time + duration
        self.duration = duration
        self.finished = False
        self.starting_mobject = deepcopy(mobject) # make a copy to let the mobject move freely as it is used in here
        
        

class Transform(Animation):
    def __init__(self, starting_mobject, target_mobject, start_time = 0, rate_time=1, rate_function=smooth):
        super().__init__(starting_mobject, start_time, rate_time, rate_function)
        # make a copy to let the target_mobject move freely as it is used in here
        # otherwise any transformation it submits to will affect the animation
        self.target_mobject = deepcopy(target_mobject)
        self.mobject = self.starting_mobject  # mobject is the one being animated, not the target
        # Align points so interpolation works
        print("Before alignment:")
        print("Circle points:", len(self.starting_mobject.points))
        print("Square points:", len(self.target_mobject.points))
        self.starting_mobject.align_points(self.target_mobject)
        
        print("After alignment:")
        print("Circle points:", len(self.starting_mobject.points))
        print("Square points:", len(self.target_mobject.points))


    def interpolate(self, t):
        f_t = self.rate(t)
        self.mobject.interpolate(self.starting_mobject, self.target_mobject, f_t)

    def finish(self):
        self.mobject.interpolate(self.starting_mobject, self.target_mobject, 1) 



class Move(Animation):
    def __init__(self, mobject, target_point,start_time = 0, rate_time=1, rate_function=smooth):
        super().__init__(mobject, start_time, rate_time, rate_function)
        self.starting_position = self.starting_mobject.get_center()
        self.target_point = target_point

    def interpolate(self, t):
        f_t = self.rate(t)
        inter_x = (1 -f_t)*self.starting_position[0] +f_t*self.target_point[0] 
        inter_y = (1 - f_t)*self.starting_position[1] + f_t*self.target_point[1]
   
        self.mobject.move_to(inter_x, inter_y)

    def finish(self):
        x = self.target_point[0]
        y = self.target_point[1]
        self.mobject.move_to(x, y)


class Scale(Animation):
    def __init__(self, mobject, scale, start_time = 0, rate_time=1, rate_function=smooth):
        super().__init__(mobject,start_time, rate_time, rate_function)
        self.scale = scale

    def interpolate(self, t):
        f_t = self.rate(t)
        s = 1 + (self.scale - 1)*f_t #scaling starts at 1
        self.mobject.scale(s)

    def finish(self):
        s = self.scale
        self.mobject.set_points(self.starting_mobject.points)
        self.mobject.scale(s)


class Rotate(Animation):
    def __init__(self, mobject, theta, start_time= 0, rate_time=1, rate_function=smooth):
        super().__init__(mobject, start_time, rate_time, rate_function)
        self.angle = theta
        self.prev_angle = 0

    def interpolate(self, t):
        f_t = self.rate(t)
        current_angle = self.angle * f_t
        delta = current_angle - self.prev_angle
        print("Frame", t, "mobject points:", self.mobject.points.shape)

        self.mobject.set_points(self.starting_mobject.points)
        self.mobject.rotate(delta)
        self.prev_angle = current_angle

    def finish(self):
        theta = self.angle
        self.mobject.rotate(theta)

    
class Fade(Animation):
    def __init__(self, mobject, target_opacity, start_time=0, rate_time=1, rate_func=smooth):
        super().__init__(mobject, start_time, rate_time, rate_func)
        self.starting_opacity = self.mobject.opacity  # float, no copy()
        self.target_opacity = target_opacity

    def interpolate(self, t):
        f_t= self.rate(t)
        self.mobject.set_opacity((1 - f_t)*self.starting_opacity +  f_t* self.target_opacity)

    def finish(self):
        self.mobject.set_opacity(self.target_opacity)

class FadeIn(Fade):
    def __init__(self, mobject, start_time = 0, rate_time=1, rate_func=smooth):
        super().__init__(mobject, 1, start_time, rate_time, rate_func=rate_func)
        self.starting_opacity = 0  # fade in from invisible

class FadeOut(Fade):
    def __init__(self, mobject, start_time,rate_time=1, rate_func=smooth):
        super().__init__(mobject, 0, start_time, rate_time, rate_func=rate_func)
        self.starting_opacity = 1  # fade out from visible

