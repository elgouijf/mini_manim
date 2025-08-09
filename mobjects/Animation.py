import mobjects.mobjects as mbj
from ..utilities.rate_functions import *

class Animation :
    def __init__(self, mobject, rate_time = 1.0, rate_function = smooth):
        mobject.apply_transform()  # clear out any pending transforms
        self.mobject = mobject
        self.time = rate_time
        self.rate = rate_function
        self.starting_mobject = self.mobject.copy()
        

class Transform(Animation):
    def __init__(self, starting_mobject, target_mobject, rate_time=1, rate_function=smooth):
        super().__init__(starting_mobject, rate_time, rate_function)
        # make a copy to let the target_mobject move freely as it is used in here
        # otherwise any transformation it submits to will affect the animation
        self.target_mobject = target_mobject.copy()

    def interpolate(self, t):
        f_t = self.rate(t)
        self.mobject.interpolate(self.starting_mobject, self.target_mobject, f_t)
 
    def finish(self):
        self.mobject.interpolate(self.starting_mobject, self.target_mobject, 1)


class Move(Animation):
    def __init__(self, mobject, target_point, rate_time=1, rate_function=smooth):
        super().__init__(mobject, rate_time, rate_function)
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
    def __init__(self, mobject, scale, rate_time=1, rate_function=smooth):
        super().__init__(mobject, rate_time, rate_function)
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
    def __init__(self, mobject, theta, rate_time=1, rate_function=smooth):
        super().__init__(mobject, rate_time, rate_function)
        self.angle = theta

    def interpolate(self, t):
        f_t = self.rate(t)
        phi = self.angle*f_t
        self.mobject.set_points(self.starting_mobject.points)
        self.mobject.rotate(phi)

    def finish(self):
        theta = self.angle
        self.mobject.rotate(theta)

    
class Fade(Animation):
    def __init__(self, mobject, target_opacity, duration=1, rate_func=smooth):
        super().__init__(mobject, duration, rate_func)
        self.starting_opacity = self.mobject.opacity  # float, no copy()
        self.target_opacity = target_opacity

    def interpolate(self, t):
        f_t= self.rate(t)
        self.mobject.set_opacity((1 - f_t)*self.starting_opacity +  f_t* self.target_opacity)

    def finish(self):
        self.mobject.set_opacity(self.target_opacity)

class FadeIn(Fade):
    def __init__(self, mobject, duration=1, rate_func=smooth):
        super().__init__(mobject, target_opacity=1, duration=duration, rate_func=rate_func)
        self.starting_opacity = 0  # fade in from invisible

class FadeOut(Fade):
    def __init__(self, mobject, duration=1, rate_func=smooth):
        super().__init__(mobject, target_opacity=0, duration=duration, rate_func=rate_func)
        self.starting_opacity = 1  # fade out from visible

