import numpy as np
import cairo
from manim_bg.renderer import Scene, Renderer
from mobjects.mobjects import Circle, Square, Polygon, Dot, GlowingDot, Point, Group, VGroup
from animations.animation import Transform, Move, Rotate, Scale, ColorChange, FadeIn, FadeOut, Fade
from math import pi
from utilities.color import *
from animations.updaters import *

WIDTH, HEIGHT, FPS = 4096, 2160, 60
# Create Cairo surface and context
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)
renderer = Renderer(surface, ctx, FPS, WIDTH, HEIGHT, file_name="double_star.mp4")


class Group_scene(Scene):
    def construct(self):
        """dot1 = GlowingDot(glow_radius= 100.0, position=np.array([WIDTH/2 + 100, HEIGHT/2]), fill_color=YELLOW)
        dot2 = GlowingDot(glow_radius= 100.0, position=np.array([WIDTH/2 - 100, HEIGHT/2]), fill_color=WHITE) """
        square1 = Square(50, center = [WIDTH/2 + 50, HEIGHT/2], fill_color = ORANGE, stroke_width = 0)
        square2 = Square(50, center = [WIDTH/2 - 50, HEIGHT/2], fill_color = ORANGE, stroke_width = 0)

        """ Double_star = VGroup(dot1, dot2) """
        Double_star = VGroup(square1, square2)

        Double_star.add_updater(rotate_continuous)
        Double_star.add_updater(move_continuous)
        Double_star.add_updater(scale_continuous)
        self.add(Double_star)
        """  print(Double_star.updaters)
        center = Double_star.get_center()

        

        self.wait(1)
        self.play(ColorChange(Double_star, target_fill_color= BLUE))
    
        self.wait(1)
        self.play(Rotate(Double_star, 2*pi, rate_time= 4))
        self.play(Scale(dot1, 1.5))"""
        for i in range(100): 
            Double_star.run_updates(dt = 4/100, ang_speed = (np.pi)/2, lin_speed = 200, direction = right, scale_speed =1/100, iteration = i, n_iterations = 99)
           
            self.render_frame("output", True)
        self.wait(1)
        self.wait(1)

# Run the scene
scene_instance = Group_scene(renderer, fps=FPS)
scene_instance.construct()
renderer.close_video()