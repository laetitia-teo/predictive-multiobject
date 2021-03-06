"""
This file defines the 2d environment with moving objects.
"""
import h5py
import os.path as op
import json
import time
import pygame
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image

from skimage.transform import resize

COLOR_NAMES = ['red', 'blue', 'green', 'yellow', 'purple']
COLORS = {
    'red': (1., 0.1, 0.1),
    'blue': (0.1, 0.1, 1.),
    'green': (0.1, 1., 0.1),
    'yellow': (1., 1., 0.1),
    'purple': (1., 0.1, 1.),
    'black': (0., 0., 0.),
    'white': (1., 1., 1.)
}

PI = np.pi
TAU = 2 * PI

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
        self.center = 0.

        # concrete atributes
        self.cond = NotImplemented
        self.shape_index = NotImplemented

    def freeze(self):
        """
        Make the object motionless. Keeps the phase.
        """
        self.x_freq = 0.
        self.x_amp = 0.
        self.y_freq = 0.
        self.y_amp = 0.

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

        self.env_params = {}

    def get_metadata(self):
        return self.env_params

    def render(self, aa=True):
        """
        Renders the environment, returns a rasterized image as a numpy array.
        """
        if aa:
            scale = 4
            L = self.L * scale
        else:
            scale = 1
            L = self.L

        gridsize = self.gridsize * scale
        mat = np.zeros((L, L, 3))
        l = self.L

        for obj in self.objects:
            obj_mat = obj.to_pixels(gridsize)
            s = len(obj_mat) # size of object in pixel space

            # base x and y pos
            ox, oy = ((gridsize * obj.pos)).astype(int)

            # compute x and y deviation due to movement
            xm = obj.x_amp * \
                (np.sin(obj.x_freq * self.time + obj.x_phase) + 1) / 2
            ym = obj.y_amp * \
                (np.sin(obj.y_freq * self.time + obj.y_phase) + 1) / 2

            ox += xm * gridsize
            oy += ym * gridsize

            # apply alpha channel
            obj_mat = obj_mat[..., :] * np.expand_dims(obj_mat[..., 3], -1)
            
            # indices
            xmin = int(max(ox, 0))
            xmax = int(max(ox + s, 0))
            ymin = int(max(oy, 0))
            ymax = int(max(oy + s, 0))
            xminobj = int(max(-ox, 0))
            xmaxobj = int(max(L - ox, 0))
            yminobj = int(max(-oy, 0))
            ymaxobj = int(max(L - oy, 0))
            
            mat[xmin:xmax, ymin:ymax] = overlay(
                mat[xmin:xmax, ymin:ymax],
                obj_mat[xminobj:xmaxobj, yminobj:ymaxobj])
        
        mat = np.flip(mat, axis=0)
        if aa:
            mat = resize(mat, (L//scale, L//scale, 3), anti_aliasing=True)

        return mat

    def reset_time(self):
        self.time = 0.

    def save_frame(self, name):
        mat = self.render() * 255
        img = Image.fromarray(np.uint8(mat))
        img.save(name)

    def make_dataset(self, dt, N, path, prefix=''):
        # test ds ? We just train for now ?
        Path(path).mkdir(parents=True, exist_ok=True)
        t = 0.

        for i in range(N):
            t = t + dt
            self.time = t
            self.save_frame(op.join(path, f"{prefix}frame{i}.png"))

        envdict = dict(self.env_params)
        envdict["dt"] = dt
        envdict["N"] = N

        # return env params as json string
        return json.dumps(envdict)

    def make_sequence(self, dt, N):
        # create a sequence with two frames per obs
        mat = np.zeros((N, self.L, self.L, 6))
        t = 0.
        self.time = t
        for n in range(N):
            mat[n, ..., :3] = self.render()
            self.time += dt
            mat[n, ..., 3:] = self.render()
            self.time += dt

        return mat

    def make_sequence_two(self, dt, N):
        # create two interleaved sequences with two frames per obs
        mat = np.zeros((N, self.L, self.L, 6))
        matnext = np.zeros((N, self.L, self.L, 6))
        t = 0.
        self.time = t
        for n in range(N):
            # obs
            mat[n, ..., :3] = self.render()
            self.time += dt
            mat[n, ..., 3:] = self.render()
            self.time += dt
            
            # next obs
            matnext[n, ..., :3] = self.render()
            self.time += dt
            matnext[n, ..., 3:] = self.render()
            self.time += dt

        return mat, matnext

    def make_dataset_symbolic(self, dt, N, path, prefix=''):
        # Same as make_dataset, but we only output symbolic positions of the
        # objects
        Path(path).mkdir(parents=True, exist_ok=True)
        t = 0.

        data = []

        for i in range(N):
            t = t + dt
            self.time = t
            data.append(np.stack((
                self.objects[0].pos,
                self.objects[1].pos
            ), 0))

        data = np.stack(data, 0)
        np.save(op.join(path, f"{prefix}.npy"))

        envdict = dict(self.env_params)
        envdict["dt"] = dt
        envdict["N"] = N

        # return env params as json string
        return json.dumps(envdict)        


    def get_frame_at_time(self, t):
        """
        Returns the array image of the env at time t.
        """
        mem = self.time
        self.time = t
        mat = self.render()
        self.time = mem
        return mat


# environment instances

class OneSphereEnv(Env):
    """
    Environment with just one moving red sphere.

    The sphere is initialized at the center.
    It doesn't move out of the FoV.
    """
    def __init__(self):
        
        super().__init__(5, 20)

        c = Circle(2, COLORS['red'], (8, 8), 0.)
        c.x_amp = 6. * self.gridsize
        c.y_amp = 6. * self.gridsize
        c.x_freq = TAU / 5.
        c.y_freq = TAU / 13.
        c.y_phase = PI / 2

        self.objects.append(c)

    def reset_params(self):
        pass

class TwoSphereEnv(Env):
    """
    Environment with just two moving spheres: one big and red, the other small
    and blue.

    The spheres don't move out of the FoV. The small one may cover the big one.
    """
    def __init__(self, gridsize=3, envsize=10, occluder=False):

        super().__init__(gridsize, envsize)

        self.reset_params(occluder=occluder)

    def reset_params(self, occluder=False):
        # sample parameters for the environment
        x_freq = np.random.random(2) * TAU
        y_freq = np.random.random(2) * TAU
        x_phase = np.random.random(2) * TAU
        y_phase = np.random.random(2) * TAU

        # register params
        self.env_params['type'] = "TwoSphereEnv"
        self.env_params['x_freq'] = list(x_freq)
        self.env_params['y_freq'] = list(y_freq)
        self.env_params['x_phase'] = list(x_phase)
        self.env_params['y_phase'] = list(y_phase)
        # self.env_params['x_amp'] = x_amp
        # self.env_params['y_amp'] = y_amp

        # reset objects
        self.objects = []

        c1 = Circle(2., COLORS['red'], (0., 0.), 0.)
        c1.x_amp = self.envsize - 2 * c1.size
        c1.y_amp = self.envsize - 2 * c1.size
        c1.x_freq = x_freq[0]
        c1.y_freq = y_freq[0]
        c1.x_phase = x_phase[0]
        c1.y_phase = y_phase[0]
        self.objects.append(c1)

        c2 = Circle(1., COLORS['blue'], (0., 0.), 0.)
        c2.x_amp = self.envsize - 2 * c2.size
        c2.y_amp = self.envsize - 2 * c2.size
        c2.x_freq = x_freq[1]
        c2.y_freq = y_freq[1]
        c2.x_phase = x_phase[1]
        c2.y_phase = y_phase[1]
        self.objects.append(c2)

        if occluder:
            # ocradius = (np.random.rand() * 6 + 6) / self.gridsize
            ocradius = 10 / self.gridsize
            ocpos = np.ones(2) * self.envsize / 2 - ocradius
            occ = Circle(ocradius, COLORS['white'], ocpos, 0.)
            occ.x_phase = 0.
            occ.y_phase = 0.
            self.objects.append(occ)

class TwoSphereScreenEnv(TwoSphereEnv):
    """
    An env with two moving spheres and a randomly placed black occluding
    screen.
    """
    def __init__(self):
        # check this
        super(TwoSphereEnv, self).__init__(3, 10)

        self.reset_params()

    def reset_params(self):

        super().reset_params()

        # add black screen
        # for now in center, motionless
        screen = Square(8., COLORS['black'], (2., 2.), 0.)
        self.objects.append(screen)


### Generating datasets

def generate_hdf5(env_type, N_samples, T, dt, fname):
    """
    Generates N_samples data points and saves them as an hdf5 file, with a 
    subgroup per sequence.
    """

    paramlist = []
    
    # sequences = {}
    f = h5py.File(fname, 'w')
    for n in range(N_samples):
        # random parameters for environment
        env = env_type()
        mat, matnext = env.make_sequence(dt, T)
        f.create_dataset(op.join(str(n), "obs"), data=mat)
        f.create_dataset(op.join(str(n), "next_obs"), data=matnext)


def load_hdf5(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    data = []

    with h5py.File(fname, 'r') as f:
        for i, grp in enumerate(f.keys()):
            data.append({})
            for key in f[grp].keys():
                data[i][key] = f[grp][key][:]
    return data


def generate(env_type, N_samples, T, dt, path):
    
    paramlist = []

    env = env_type()
    
    for n in tqdm(range(N_samples)):
        paramlist.append(env.make_dataset(dt, T, path, prefix=f"sample{n}"))
        env.reset_params()

    with open(op.join(path, "envparams.txt"), 'w') as f:
        f.write("\n".join(paramlist))


def generate_grid_two_spheres(side, object_id, path):
    """
    Generates a set of images where the object identified by object-id
    is placed sucessively at all possible points of a regular grid of length
    side. The other object stays motionless.
    """
    env = TwoSphereEnv()
    objsize = env.objects[object_id].size

    lin = np.linspace(0, env.envsize - 2*objsize, side)
    grid = np.stack(np.meshgrid(lin, lin), -1)

    for i in range(side):
        for j in range(side):
            env.objects[object_id].pos = grid[i, j]
            env.objects[object_id].x_amp = 0.
            env.objects[object_id].y_amp = 0.

            mat = env.render()
            env.save_frame(op.join(
                path,
                f"grid{object_id}frame{i*side+j}.png")
            )


def generate_two_spheres_no_movement(T, N_samples, dt, path):
    """
    Generate a dataset where only one of the spheres moves at a time.
    The even-numbered samples have the 0th object not move, the odd-numbered
    ones have the 1st object not move.
    """
    # T = 200
    # N_samples is multiplied by 2
    # dt = 0.3 ?
    paramlist = []
    env = TwoSphereEnv()

    for n in tqdm(range(2*N_samples)):
        idx = n % 2    
        env.objects[idx].x_amp = 0.
        env.objects[idx].y_amp = 0.
        paramlist.append(env.make_dataset(dt, T, path, prefix=f"sample{n}"))
        env.reset_params()

    with open(op.join(path, "envparams.txt"), 'w') as f:
        f.write("\n".join(paramlist))

def generate_two_sphere_dataset(dest,
                                train_set_size,
                                seq_len,
                                mode='simple',
                                img_size=None,
                                dt=0.3,
                                seed=0,
                                occluder=False,
                                **kwargs):

    # possible modes are 'simple', 'indep', 'indep_partial'
    np.random.seed(seed)

    env_type = TwoSphereEnv

    # generate dataset
    f = h5py.File(dest, 'w')
    for n in range(train_set_size):
        if n % 100 == 0:
            print(f"Generated {n} sequences out of {train_set_size}")
        # random parameters for environment
        env = env_type(occluder=occluder)
        if mode == "indep":
            if np.random.random() < 0.5:
                env.objects[0].freeze()
            else:
                env.objects[1].freeze()
        if mode == "indep_partial":
            r = np.random.random()
            if r < 1/3:
                env.objects[0].freeze()
            elif r < 2/3:
                env.objects[1].freeze()

        mat = env.make_sequence(dt, seq_len)
        f.create_dataset(str(n), data=mat)
        # record metadata
        for key, value in env.get_metadata().items():
            f[str(n)].attrs[key] = value

def generate_two_sphere_grid(dest,
                             side,
                             occluder=False):

    f = h5py.File(dest, 'w')

    for object_id in range(2):
        env = TwoSphereEnv(occluder=occluder)
        objsize = env.objects[object_id].size

        lin = np.linspace(0, env.envsize - 2*objsize, side)
        grid = np.stack(np.meshgrid(lin, lin), -1)

        N = side**2
        mat = np.zeros((N, env.L, env.L, 6))
        t = 0.
        env.time = t

        for i in range(side):
            for j in range(side):
                env.objects[object_id].pos = grid[i, j]
                env.objects[object_id].x_amp = 0.
                env.objects[object_id].y_amp = 0.

                n = side*i + j
                mat[n, ..., :3] = env.render()
                mat[n, ..., 3:] = env.render()

        f.create_dataset(f'object_{object_id}_grid', data=mat)

### Testing environments

envdict = {
    "two sphere": TwoSphereEnv,
    "two sphere screen": TwoSphereScreenEnv
}

# TESTED_ENV = envdict["two sphere"]
TESTED_ENV = envdict["two sphere"]

if __name__ == '__main__':

    # env = OneSphereEnv()
    env = TESTED_ENV()
    pygame.init()
    done = False

    X = env.L
    Y = env.L
    # dt = 1 / 30 # fps
    dt = 0.15

    framename = 'frame.jpg'
    env.save_frame(framename)
    display = pygame.display.set_mode((X, Y))
    pygame.display.set_caption('Movement test')

    t0 = time.time()

    while not done:
        display.fill((0, 0, 0))
        display.blit(pygame.image.load(framename), (0, 0))
        pygame.display.update()

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                        done = True

        # update env
        t = time.time() - t0
        env.time = t
        env.save_frame(framename)

        # cap the time interval
        t1 = time.time() - t0
        if (t1 - t) < dt:
            time.sleep(dt - (t1 - t))
    pygame.quit()