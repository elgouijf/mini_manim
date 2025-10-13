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
renderer = Renderer(surface, ctx, FPS, WIDTH, HEIGHT, file_name="dot.mp4")

print(Color("yellow").get_rgb())
print(Color("Yellow").get_rgb())

class dot_scene(Scene):
    def construct(self):
        dot = GlowingDot(glow_radius= 50.0, position=np.array([WIDTH/2, HEIGHT/2]), fill_color=YELLOW)
        circle = Circle(center = np.array([WIDTH/2, HEIGHT/2]), radius=50.0)
        
        circle.fill_opacity = 0.0
        circle.set_stroke_color(RED)
        print("dot.fill_color =", dot.fill_color)
        print("resolved color =", ensure_color(dot.fill_color).get_rgb())
        print("fill_opacity =", dot.fill_opacity)

        """ mobjects = [dot, circle]
        group = VGroup(*mobjects) """
        """ self.add(circle) """
        print("Dot color", dot.fill_color)

        dot.add_updater(scale_continuous)

        self.add(dot)
        self.wait(1)
        self.play(ColorChange(dot, target_fill_color= BLUE))
        print("Dot color", dot.fill_color)
        print("Dot circle color", dot.submobjects[0].fill_color)
        self.wait(1)
        self.play(ColorChange(dot, target_fill_color= RED))
        print("Dot color", dot.fill_color)
        self.wait(1) 
        
        self.play(Move(dot, np.array([3*WIDTH/4, HEIGHT/2])))
        self.play(Move(dot, np.array([WIDTH/2, HEIGHT/2]))) 

        self.wait(1)
        self.play(Scale(dot, 3))
        self.play(Scale(dot, 1.5))

        for i in range(100):
            dot.run_updates(dt = 4/100, scale_speed = 2/4, iteration = i)
            self.render_frame("output", True)

        self.wait(1)



        self.wait(1)


# Run the scene
scene_instance = dot_scene(renderer, fps=FPS)
scene_instance.construct()
renderer.close_video()