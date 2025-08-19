import numpy as np
import cairo
from manim_bg.renderer import Scene, Renderer
from mobjects.mobjects import Circle, Square, Polygon, Dot, GlowingDot, Point
from animations.animation import Transform, Move, Rotate, Scale, ColorChange, FadeIn, FadeOut, Fade
from math import pi
from utilities.color import *

WIDTH, HEIGHT, FPS = 4096, 2160, 60
# Create Cairo surface and context
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)
renderer = Renderer(surface, ctx, FPS, WIDTH, HEIGHT, file_name="dot.mp4")

print(Color("yellow").get_rgb())
print(Color("Yellow").get_rgb())  
class dot_scene(Scene):
    def construct(self):
        
        dot = GlowingDot(glow_radius= 20.0, position=np.array([WIDTH/2, HEIGHT/2]), fill_color=YELLOW)
        circle = Circle(center = np.array([WIDTH/2, HEIGHT/2]), radius=200.0)
        dot.fill_color = YELLOW
        dot.stroke_color = YELLOW
        circle.fill_opacity = 0.0
        circle.set_stroke_color(RED)
        print("dot.fill_color =", dot.fill_color)
        print("resolved color =", ensure_color(dot.fill_color).get_rgb())
        print("fill_opacity =", dot.fill_opacity)

        self.add(circle)
        self.add(dot)

        self.wait(1)



# Run the scene
scene_instance = dot_scene(renderer, fps=FPS)
scene_instance.construct()
renderer.close_video()
