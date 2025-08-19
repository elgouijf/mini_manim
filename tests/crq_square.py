import numpy as np
import cairo
from manim_bg.renderer import Scene, Renderer
from mobjects.mobjects import Circle, Square, Polygon
from animations.animation import Transform, Move, Rotate, Scale, ColorChange, FadeIn, FadeOut, Fade
from math import pi
from utilities.color import *

WIDTH, HEIGHT, FPS = 4096, 2160, 60

# Create Cairo surface and context
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)

# Create renderer
renderer = Renderer(surface, ctx, FPS, WIDTH, HEIGHT, file_name="circle_to_square_fixed.mp4")

# Patch: Clear background each frame (optional if Renderer or Scene does it)
class ClearScreenScene(Scene):
    def render_frame(self, out="output", main_save=True):
        # Clear to black background each frame
        self.renderer.ctx.set_source_rgb(0, 0, 0)
        self.renderer.ctx.paint()
        super().render_frame(out, main_save=main_save)

class CircleToSquare(ClearScreenScene):
    def construct(self):
        # Create a larger circle and square centered on screen
        circle = Circle(radius=100.0, center=np.array([WIDTH/2, HEIGHT/2]))
        polygone = Polygon(radius=100, center=np.array([WIDTH/2, HEIGHT/2]), n =8)
        square = Square(side_len=200.0, center=np.array([WIDTH/2, HEIGHT/2]))
        triangle = Polygon(radius=100, center=np.array([WIDTH/2, HEIGHT/2]), n=3)

        polygones = []
        for i in range(1,9):
            poly = Polygon(radius=200, center=np.array([WIDTH/2, HEIGHT/2]), n=i+3)
            polygones.append(poly)

        # Explicitly set colors
        circle.set_fill_color((0, 0, 0))   # blue fill
        circle.stroke_color = (0, 0, 1)    # blue stroke
        polygone.set_fill_color((0, 1, 0))   # red fill
        polygone.stroke_color = (0, 1, 0)    # red stroke
        square.set_fill_color((0, 0, 0))   # red fill
        square.set_fill_opacity(0.0)
        square.stroke_color = (1, 0, 0)    # red stroke

        """ print("Circle points before animation:", circle.points.shape)
        print("First 5 points:", circle.points[:5])
        self.add(circle) """
    
        
        
        """ print(square.fill_opacity) """
        """ self.play(Rotate(circle, pi/4)) """
        self.add(square)
        self.wait(0.05)
        self.play(Rotate(square, pi/4))
        self.wait(0.05)
        self.wait(0.05)
        self.play(Rotate(square, -pi/4))
        self.play(Transform(square, circle))
        self.wait(0.05)
        self.play(Transform(square, polygone))
        self.wait(0.05)
        self.play(Transform(square, triangle))
        self.wait(0.05)
        self.play(Scale(square, 2))
        self.wait(0.05)
        self.play(Rotate(square, pi))
        self.wait(0.05)
        self.play(ColorChange(square, WHITE, GREEN))
        self.wait(0.05)
        self.play(FadeOut(square))
        self.wait(0.05)
        self.play(Fade(square, target_fill_opacity=0, target_stroke_opacity=1))
        side = square.side_len
        self.play(Move(square, square.get_center() + np.array([WIDTH/2 - side, 0.0])))
        self.play(Move(square, np.array([side, HEIGHT/2])))
        self.play(Move(square, np.array([WIDTH/2, HEIGHT/2])))

        for i in range(0, len(polygones)):
            self.play(Transform(square, polygones[i]))
            self.wait(0.05)

        self.play(Transform(square, Circle(radius=200.0, center=np.array([WIDTH/2, HEIGHT/2]))))
        self.wait(0.05)
        self.play(Transform(square, Square(side_len=200.0, center=np.array([WIDTH/2, HEIGHT/2]))))


# Run the scene
scene_instance = CircleToSquare(renderer, fps=FPS)
scene_instance.construct()
renderer.close_video()
