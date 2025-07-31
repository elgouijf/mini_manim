import os

import sys
import cairo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mobjects.mobjects as mbj
import numpy as np
WIDTH = 480
HEIGHT = 360

class Renderer:
    #object to draw
    def __init__(self,im_surface,ctx):
        self.im_surface = im_surface
        self.ctx = ctx
    def render_polygone(self,mobject):
        if mobject.points.shape[0] == 0:
            return#no points to draw
        #drawing succesion of points
        points = mobject.points
        for i in range(len(points)):
            if not i:
                self.ctx.move_to(*points[0])
            else:
                self.ctx.line_to(*points[i])
        self.ctx.line_to(*points[0])
        #fill options
        self.ctx.set_source_rgb(mobject.fill_color[0],mobject.fill_color[1],
                           mobject.fill_color[2])#fill_color
        self.ctx.fill_preserve()#fill but keep the path
        self.ctx.set_source_rgb(mobject.stroke_color[0],mobject.stroke_color[1],
                           mobject.stroke_color[2])
        self.ctx.set_line_width(2)
        self.ctx.stroke()
    #if it is  a vmobject
    def render_vm(self,vmobject : mbj.VMobject):
        if not vmobject.submobjects:
            if not vmobject.subpaths:
                if vmobject.points.shape[0] == 0:
                    return#no points to draw

                points = vmobject.points
                for i in range(len(points)):
                    if not i:
                        self.ctx.move_to(*points[0])
                    else:
                        self.ctx.line_to(*points[i])
                if vmobject.close:
                    self.ctx.line_to(*points[0])
                
                    #fill options
                    self.ctx.set_source_rgb(vmobject.fill_color[0],vmobject.fill_color[1],
                                    vmobject.fill_color[2])#fill_color
                    self.ctx.fill_preserve()#fill but keep the path
                    self.ctx.set_source_rgb(vmobject.stroke_color[0],vmobject.stroke_color[1],
                                    vmobject.stroke_color[2])
                    self.ctx.set_line_width(2)
                    self.ctx.stroke()
            else:
                for i,subpath in enumerate(vmobject.subpaths):
                    if subpath.shape[0] == 0:
                        continue#no points to draw

                    points = subpath
                    for j in range(len(points)):
                        if not j:
                            self.ctx.move_to(*points[0][:2])
                        else:
                            self.ctx.line_to(*points[j][:2])
                    if vmobject.closed_subpaths[i]:
                        self.ctx.line_to(*points[0])
                    
                        #fill options
                        self.ctx.set_source_rgb(vmobject.fill_color[0],vmobject.fill_color[1],
                                        vmobject.fill_color[2])#fill_color
                        self.ctx.fill_preserve()#fill but keep the path
                        self.ctx.set_source_rgb(vmobject.stroke_color[0],vmobject.stroke_color[1],
                                        vmobject.stroke_color[2])
                        self.ctx.set_line_width(2)
                        self.ctx.stroke()
        else:
            for submobject in vmobject.submobjects:
                if isinstance(submobject,mbj.VMobject):
                    self.render_vm(submobject)
                else:
                    continue

            
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
            self.renderer.render_polygon(mobject)


def test_mobject_basic_transform():
    mobj = mbj.Mobject()
    mobj.points = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    
    # Set visual style
    mobj.fill_color = (0.2, 0.6, 0.8)     # light blue
    mobj.stroke_color = (0, 0, 0)         # black
    
    # Apply transformations
    mobj.move_to(200, 150)
    mobj.apply_transform()
    mobj.scale(80)
    mobj.apply_transform()

    # Cairo setup
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    renderer = Renderer(surface, ctx)
    scene = Scene(renderer)

    scene.add(mobj)
    scene.render()

    surface.write_to_png("mobject_output.png")
    print("Saved to mobject_output.png")
#test_mobject_basic_transform()
def test_vmobject_open_and_closed():
 

    # Cairo setup
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)

    # adjust import if needed
    renderer = Renderer(surface, ctx)

    # Create a closed shape
    closed_vm = mbj.VMobject()
    closed_vm.points = np.array([
        [100, 100],
        [150, 100],
        [150, 150],
        [100, 150]
    ])
    closed_vm.fill_color = (0.3, 0.6, 0.3)
    closed_vm.stroke_color = (0, 0, 0)
    closed_vm.close = True

    # Create an open shape
    open_vm = mbj.VMobject()
    open_vm.points = np.array([
        [200, 200],
        [250, 200],
        [250, 250]
    ])
    open_vm.fill_color = (0.8, 0.2, 0.2)
    open_vm.stroke_color = (0, 0, 0)
    open_vm.close = False

    # Render both
    renderer.render_vm(closed_vm)
    renderer.render_vm(open_vm)

    surface.write_to_png("vmobject_test_output.png")
    print("Saved to vmobject_test_output.png")

test_vmobject_open_and_closed()