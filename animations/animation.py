import mobjects.mobjects as mbj
from copy import deepcopy
from utilities.rate_functions import *
from utilities.color import *

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
        
         # Align points so interpolation works
        #print("Before alignment:")
        #print("Circle points:", len(self.starting_mobject.points))
        #print("Square points:", len(self.target_mobject.points))
        self.starting_mobject.align_points(self.target_mobject)
        
        #print("After alignment:")
        #print("Circle points:", len(self.starting_mobject.points))
        #print("Square points:", len(self.target_mobject.points))

        if isinstance(self.mobject, mbj.VMobject):
            self.mobject.subpaths = []
            self.mobject.closed_subpaths = []

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
        inter_x = (1 -f_t)*self.starting_position[0] +f_t*(self.target_point[0] )
        inter_y = (1 - f_t)*self.starting_position[1] + f_t*(self.target_point[1])
   
        self.mobject.move_to(inter_x, inter_y)

    def finish(self):
        x = self.target_point[0]
        y = self.target_point[1]
        self.mobject.move_to(x, y)


class Scale(Animation):
    def __init__(self, mobject, scale, start_time=0, duration=1, rate_function=smooth):
        super().__init__(mobject, start_time, duration, rate_function)
        self.target_scale = scale

    def interpolate(self, t):
        f_t = self.rate(t)
        s = 1 + (self.target_scale - 1) * f_t
        # reset mobject to starting state at the beginning of animation
        self.mobject.set_points(self.starting_mobject.points)
        self.mobject.scale(s)

    def finish(self):
        self.interpolate(1)


class Rotate(Animation):
    def __init__(self, mobject, theta, center=None, start_time=0, rate_time=1, rate_function=smooth):
        super().__init__(mobject, start_time, rate_time, rate_function)
        self.angle = theta
        self.prev_angle = 0
        # If no center given, mark that we should use the mobject’s center dynamically
        self.use_mobject_center = (center is None)
        # Store the fixed center if provided (None if dynamic)
        self.center = center

    def interpolate(self, t):
        # Compute incremental rotation angle
        f_t = self.rate(t)
        current_angle = self.angle * f_t
        delta = current_angle - self.prev_angle

        # Determine rotation center: dynamic or fixed
        if self.use_mobject_center:
            center = self.mobject.get_center()
        else:
            center = self.center

        # Rotate the mobject by the incremental angle about the chosen center
        self.mobject.rotate(delta, center)
        self.prev_angle = current_angle

    def finish(self):
        self.interpolate(1)


    
class Fade(Animation):
    def __init__(self, mobject, target_fill_opacity, target_stroke_opacity, start_time=0, rate_time=1, rate_func=smooth):
        super().__init__(mobject, start_time, rate_time, rate_func)
        self.starting_fill_opacity = self.mobject.fill_opacity  # float, no copy()
        self.starting_stroke_opacity = self.mobject.stroke_opacity
        self.target_fill_opacity = target_fill_opacity
        self.target_stroke_opacity = target_stroke_opacity


    def interpolate(self, t):
        f_t= self.rate(t)
        self.mobject.set_fill_opacity((1 - f_t)*self.starting_fill_opacity +  f_t* self.target_fill_opacity)
        self.mobject.set_stroke_opacity((1 - f_t)*self.starting_stroke_opacity +  f_t* self.target_stroke_opacity)

    def finish(self):
        self.interpolate(1)


class FadeIn(Fade):
    def __init__(self, mobject, start_time = 0, rate_time=1, rate_func=smooth):
        super().__init__(mobject, 1, 1, start_time, rate_time, rate_func=rate_func)
        self.starting_fill_opacity = self.mobject.fill_opacity  # fade in from invisible
        self.starting_stroke_opacity = self.mobject.stroke_opacity


class FadeOut(Fade):
    def __init__(self, mobject, start_time = 0,rate_time=1, rate_func=smooth):
        super().__init__(mobject, 0, 0, start_time, rate_time, rate_func=rate_func)
        self.starting_fill_opacity = self.mobject.fill_opacity  # fade out from visible
        self.starting_stroke_opacity = self.mobject.stroke_opacity


class ColorChange(Animation):
    def __new__(cls, mobject, *args, **kwargs):
        if hasattr(mobject, "is_group") and mobject.is_group:
            return ColorChange_group(mobject, *args, **kwargs)
        else:
            # Since Animation has no __new__ method object.__new
            return super().__new__(cls)
        
    def __init__(self, mobject, target_fill_color = PURPLE, target_stroke_color= PURPLE, start_time=0, rate_time=1, rate_function=smooth):
        super().__init__(mobject, start_time, rate_time, rate_function)
        self.starting_fill_color = self.mobject.fill_color
        self.target_fill_color = target_fill_color
        self.starting_stroke_color = self.mobject.stroke_color
        self.target_stroke_color = target_stroke_color
        # maintain the same opacity as before
    
        

    def interpolate(self, t):
        f_t = self.rate(t)
        new_fill_color = interpolate_colors(self.starting_fill_color, self.target_fill_color, f_t)
        self.mobject.set_fill_color(new_fill_color)
        print(self.mobject.fill_color)

        new_stroke_color = interpolate_colors(self.starting_stroke_color, self.target_stroke_color, f_t)
        self.mobject.set_stroke_color(new_stroke_color)
        """ print(self.mobject.stroke_opacity) """

    def finish(self):
        self.interpolate(1)


class ColorChange_group(Animation):
    def __init__(self, mobject, target_fill_color=PURPLE, target_stroke_color=PURPLE, start_time=0, rate_time=1, rate_function=smooth):
        super().__init__(mobject, start_time, rate_time, rate_function)
        self.sub_animations = [
            ColorChange(sub, target_fill_color, target_stroke_color, start_time, rate_time, rate_function)
            for sub in mobject.submobjects
        ]

    def interpolate(self, t):
        for anim in self.sub_animations:
            anim.interpolate(t)

    def finish(self):
        for anim in self.sub_animations:
            anim.finish()
