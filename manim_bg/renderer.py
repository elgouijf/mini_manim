import cairo
from mobjects.mobjects import Mobjects
WIDTH = 480
HEIGHT = 360
im_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,WIDTH,HEIGHT)
ctx = cairo.Context(im_surface)

class Renderer:
    #object to draw
    def render(self,mobject):
        ctx.set_source_rgb(mobject.fill_color)#fill_color
        ctx.fill(mobject.stroke_color)

        

    
class Scene:
    """"
    collection of objects to draw
    """
    def __init__(self,renderer):
        self.renderer = renderer
        self.mobjects = []
    def add(self,mobject):
        #storing the list pf visuals
        self.mobjects.append(mobject)
    def remove(self,mobject):

        for i in range(len(self.mobjects)):
            if mobject == self.mobjects[i]:
                self.mobjects.pop(i)
                break
    def render(self):
        for mobject in self.mobjects:
            self.renderer.render(mobject)
sc = Scene(Renderer)
sc.add("circle")
sc.add("square")
sc.render()
        