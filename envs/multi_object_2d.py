"""
This file defines the 2d environment with moving objects.

At first, the env can contain up to 5 objects of different colors.
"""
import numpy as np

import time
import pygame

from PIL import Image

COLOR_NAMES = ['red', 'blue', 'green', 'yellow', 'purple']
COLORS = {
    'red': (1., 0.1, 0.1),
    'blue': (0.1, 0.1, 1.),
    'green': (0.1, 1., 0.1),
    'yellow': (1., 1., 0.1),
    'purple': (1., 0.1, 1.)
}
PI = np.pi

### Shape definitions

class Shape():
    """
    Abstract implementation of an object and its methods.
    """

    def __init__(self, size, color, pos, ori):
        """
        An abstract shape representation. A shape has the following attributes:

            - size (float) : the radius of the shape. The radius is defined as
                the distance between the center of the shape and the point
                farthest from the shape. This is used to define a 
                bounding-circle centered on the position of the object.

            - color (RGB array) : the color of the shape.

            - pos (array of size 2) : the absolute poition of the object. 

            - ori (float) : the orientation of the object, in radians.

        The class also defines movement. Each shape has a min and max x and y
        position and a frquency on x and on y.
        """

        # object attributes
        self.size = size
        self.color = color
        self.pos = np.array(pos)
        self.ori = ori

        # movement attributes
        self.moves = False
        self.x_amp = 0.
        self.y_amp = 0.
        self.x_freq = 1.
        self.y_freq = 1.
        self.x_phase = 0.
        self.y_phase = PI / 2

        # concrete atributes
        self.cond = NotImplemented
        self.shape_index = NotImplemented

    def to_pixels(self, gridsize):
        """
        Returns a two dimensional array of 4D vectors (RGBa), of size the 
        object size times the grid size, in which the shape is encoded into
        pixels.

        Arguments : 
            - gridsize (int) : the number of pixels in a unit.
        """
        size = int(self.size * gridsize)
        x, y = np.meshgrid(np.arange(2*size), np.arange(2*size))
        x = (x - size) / size
        y = (y - size) / size
        void = np.zeros(4)
        color = np.concatenate((self.color, [1.]))
        x = np.expand_dims(x, -1)
        y = np.expand_dims(y, -1)
        bbox = np.where(self.cond(x, y), color, void)
        return bbox

class Square(Shape):

    def __init__(self, size, color, pos, ori):
        super().__init__(
            size,
            color,
            pos,
            ori)
        self.shape_index = 0
        self.cond = self.cond_fn

    def cond_fn(self, x, y):
        theta = self.ori
        x_ = x * np.cos(theta) - y * np.sin(theta)
        y_ = x * np.sin(theta) + y * np.cos(theta)
        c =  np.less_equal(
            np.maximum(abs(x_), abs(y_)),
            1/np.sqrt(2))
        return c

    def copy(self):
        """
        Returns a Square with the same attributes as the current one.
        """
        size = self.size
        color = np.array(self.color)
        pos = np.array(self.pos)
        ori = self.ori
        return Square(size, color, pos, ori)

class Circle(Shape):

    def __init__(self, size, color, pos, ori):
        super().__init__(
            size,
            color,
            pos,
            ori)
        self.shape_index = 1
        self.cond = lambda x, y : np.less_equal(x**2 + y**2, 1)

    def copy(self):
        """
        Returns a Circle with the same attributes as the current one.
        """
        size = self.size
        color = np.array(self.color)
        pos = np.array(self.pos)
        ori = self.ori
        return Circle(size, color, pos, ori)

class Triangle(Shape):

    def __init__(self, size, color, pos, ori):
        super().__init__(
            size,
            color,
            pos,
            ori)
        self.shape_index = 2
        self.cond = self.cond_fn

    def cond_fn(self, x, y):
        theta = self.ori
        x_ = x * np.cos(theta) - y * np.sin(theta)
        y_ = x * np.sin(theta) + y * np.cos(theta)
        a = np.sqrt(3)
        b = 1.
        c = np.greater_equal(y_, -1/2) * \
            np.less_equal(y_, a*x_ + b) * \
            np.less_equal(y_, (- a)*x_ + b)
        return c

    def copy(self):
        """
        Returns a Triangle with the same attributes as the current one.
        """
        size = self.size
        color = np.array(self.color)
        pos = np.array(self.pos)
        ori = self.ori
        return Triangle(size, color, pos, ori)

def overlay(mat1, mat2):
    """
    Overalays mat2 (last channel of last dimension is considered alpha channel)
    over mat1.
    Retruns the resulting matrix, with no alpha channel.
    """
    alphas = np.expand_dims(mat2[..., -1], -1)
    return mat1 * (1 - alphas) \
        + mat2[..., :-1] * alphas

### Envs

class Env():
    """
    Base class for the different Environment instances.
    """
    def __init__(self, gridsize, envsize, objects=[]):

        self.gridsize = gridsize
        self.envsize = envsize
        self.L = int(envsize * gridsize)
        self.objects = objects

        self.num_shapes = len(self.objects)

        self.time = 0.

    def render(self):
        """
        Renders the environment, returns a rasterized image as a numpy array.
        """
        L = self.L
        mat = np.zeros((L, L, 3))
        l = self.L

        for obj in self.objects:
            obj_mat = obj.to_pixels(self.gridsize)
            s = len(obj_mat) # size of object in pixel space

            # base x and y pos
            ox, oy = ((self.gridsize * obj.pos) - int(s/2)).astype(int)
            # compute x and y deviation due to movement
            xm = obj.x_amp * np.sin(obj.x_freq * self.time + obj.x_phase)
            ym = obj.y_amp * np.sin(obj.y_freq * self.time + obj.y_phase)

            ox += xm
            oy += ym

            obj_mat = obj_mat[..., :] * np.expand_dims(obj_mat[..., 3], -1)
            
            # indices
            xmin = max(ox, 0)
            xmax = max(ox + s, 0)
            ymin = max(oy, 0)
            ymax = max(oy + s, 0)
            xminobj = max(-ox, 0)
            xmaxobj = max(L - ox, 0)
            yminobj = max(-oy, 0)
            ymaxobj = max(L - oy, 0)
            
            mat[xmin:xmax, ymin:ymax] = overlay(
                mat[xmin:xmax, ymin:ymax],
                obj_mat[xminobj:xmaxobj, yminobj:ymaxobj])
        
        mat = np.flip(mat, axis=0)

        return mat

    def save_frame(self, name):
        mat = self.render()
        img = Image.from_numpy(np.uint8(mat))
        img.save(name)

class OneSphereEnv(Env):
    """
    Environment with just one moving red sphere.

    The sphere is initialized at the center.
    It doesn't move out of the FoV.
    """
    def __init__(self):
        
        super().__init__(16, 20)

        c = Circle(2, COLORS['red'], (8, 8), 0.)
        c.x_amp = 6.
        c.y_amp = 6.
        c.x_freq = 2 * PI / 5.
        c.y_freq = 2 * PI / 10.
        c.y_phase = PI / 2

        self.objects.append(c)

### Testing environments

if __name__ == '__main__':

    env = OneSphereEnv()
    pygame.init()
    done = False

    X = env.L
    Y = env.L

    framename = 'frame.jpg'
    env.save_frame(framename)
    display = pygame.display.set_mode((X, Y))
    pygame.display.set_caption('Movement test')

    while not done:
        display.fill((0, 0, 0))
        display.blit(pygame.image.load(framename), (0, 0))
        pygame.display.update()

        events = pygame.event.get()
        for event in events:
            if event.key in [pygame.K_ESCAPE, pygame.Q]:
                    done = True

        env.save_frame(framename)
    pygame.quit()