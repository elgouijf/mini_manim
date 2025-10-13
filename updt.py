import numpy as np
import cairo
from manim_bg.renderer import Scene, Renderer
from mobjects.mobjects import Circle, Square, Polygon, Dot, GlowingDot, Point, Group, VGroup
from animations.animation import Transform, Move, Rotate, Scale, ColorChange, FadeIn, FadeOut, Fade
from math import pi
from utilities.color import *
from utilities.rate_functions import *
from animations.updaters import *

WIDTH, HEIGHT, FPS = 4096, 2160, 60
# Create Cairo surface and context
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)
renderer = Renderer(surface, ctx, FPS, WIDTH, HEIGHT, file_name="updt.mp4")

class non_slip_bearing(Scene):
    def construct(self):
        
        square = Square(center = np.array([WIDTH/2, HEIGHT/2]),side_len= 100.0, fill_color = BLUE, stroke_color= BLUE)
        square.add_updater(move_continuous)
        square.add_updater(rotate_continuous)
        square.add_updater(scale_continuous)
        self.add(square)
        self.wait(1)
        for i in range(100):
            square.run_updates(dt = 4/100, ang_speed = (np.pi)/2, lin_speed = 200, direction = right, scale_speed = 2/4, iteration = i)
            self.render_frame("output", True)

        self.wait(1)



# Run the scene
scene_instance = non_slip_bearing(renderer, fps=FPS)
scene_instance.construct()
renderer.close_video()
        
        