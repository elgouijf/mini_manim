import numpy as np
import cairo
from manim_bg.renderer import Scene, Renderer
from mobjects.mobjects import Circle, Square, Polygon
from animations.animation import Transform, Move, Rotate
from math import pi

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
        polygon = Polygon(radius=100, center=np.array([WIDTH/2, HEIGHT/2]), n =32)
        square = Square(side_len=200.0, center=np.array([WIDTH/2, HEIGHT/2]))

        # Explicitly set colors
        circle.set_fill_color((0, 0, 0))   # blue fill
        circle.stroke_color = (0, 0, 1)    # blue stroke
        polygon.set_fill_color((0, 1, 0))   # red fill
        polygon.stroke_color = (0, 1, 0)    # red stroke
        square.set_fill_color((0, 0, 0))   # red fill
        square.set_opacity(0.0)
        square.stroke_color = (0, 0, 1)    # red stroke

        """ print("Circle points before animation:", circle.points.shape)
        print("First 5 points:", circle.points[:5])
        self.add(circle) """
        self.add(circle)
        
        self.wait(1)
        """ self.play(Rotate(circle, pi/4)) """
        self.play(Transform(circle, square))

        self.wait(1)

# Run the scene
scene_instance = CircleToSquare(renderer, fps=FPS)
scene_instance.construct()
renderer.close_video()
