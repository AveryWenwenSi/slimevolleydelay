"""
Port of Neural Slime Volleyball to Python Gym Environment

David Ha (2020)

Original version:

https://otoro.net/slimevolley
https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
https://github.com/hardmaru/neuralslimevolley

No dependencies apart from Numpy and Gym
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import cv2 # installed with gym anyways
from collections import deque

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

import pdb

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True

REF_W = 24*2
REF_H = REF_W
REF_U = 1.5 # ground height
REF_WALL_WIDTH = 1.0 # wall width
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 10*1.75
PLAYER_SPEED_Y = 10*1.35
MAX_BALL_SPEED = 15*1.5
TIMESTEP = 1/30.
NUDGE = 0.1
FRICTION = 1.0 # 1 means no FRICTION, less means FRICTION
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8*2*1.5

MAXLIVES = 5 # game ends when one agent loses this many games

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 500

FACTOR = WINDOW_WIDTH / REF_W

# if set to true, renders using cv2 directly on numpy array
# (otherwise uses pyglet / opengl -> much smoother for human player)
PIXEL_MODE = False 
PIXEL_SCALE = 4 # first render at multiple of Pixel Obs resolution, then downscale. Looks better.

PIXEL_WIDTH = 84*2*1
PIXEL_HEIGHT = 84*1

PRETRAINED_MODEL_PATH = "zoo/ppo_sp/history_00000144.zip"

def setNightColors():
  ### night time color:
  global BALL_COLOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR
  global PIXEL_AGENT_LEFT_COLOR, PIXEL_AGENT_RIGHT_COLOR
  global BACKGROUND_COLOR, FENCE_COLOR, COIN_COLOR, GROUND_COLOR
  BALL_COLOR = (217, 79, 0)
  AGENT_LEFT_COLOR = (35, 93, 188)
  AGENT_RIGHT_COLOR = (255, 236, 0)
  PIXEL_AGENT_LEFT_COLOR = (255, 191, 0) # AMBER
  PIXEL_AGENT_RIGHT_COLOR = (255, 191, 0) # AMBER
  
  BACKGROUND_COLOR = (11, 16, 19)
  FENCE_COLOR = (102, 56, 35)
  COIN_COLOR = FENCE_COLOR
  GROUND_COLOR = (116, 114, 117)

def setDayColors():
  ### day time color:
  ### note: do not use day time colors for pixel-obs training.
  global BALL_COLOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR
  global PIXEL_AGENT_LEFT_COLOR, PIXEL_AGENT_RIGHT_COLOR
  global BACKGROUND_COLOR, FENCE_COLOR, COIN_COLOR, GROUND_COLOR
  global PIXEL_SCALE, PIXEL_WIDTH, PIXEL_HEIGHT
  PIXEL_SCALE = int(4*1.0)
  PIXEL_WIDTH = int(84*2*1.0)
  PIXEL_HEIGHT = int(84*1.0)
  BALL_COLOR = (255, 200, 20)
  AGENT_LEFT_COLOR = (240, 75, 0)
  AGENT_RIGHT_COLOR = (0, 150, 255)
  PIXEL_AGENT_LEFT_COLOR = (240, 75, 0)
  PIXEL_AGENT_RIGHT_COLOR = (0, 150, 255)
  
  BACKGROUND_COLOR = (255, 255, 255)
  FENCE_COLOR = (240, 210, 130)
  COIN_COLOR = FENCE_COLOR
  GROUND_COLOR = (128, 227, 153)

setNightColors()

# by default, don't load rendering (since we want to use it in headless cloud machines)
rendering = None
def checkRendering():
  global rendering
  if rendering is None:
    from gym.envs.classic_control import rendering as rendering

def setPixelObsMode():
  """
  used for experimental pixel-observation mode
  note: new dim's chosen to be PIXEL_SCALE (2x) as Pixel Obs dims (will be downsampled)

  also, both agent colors are identical, to potentially facilitate multiagent
  """
  global WINDOW_WIDTH, WINDOW_HEIGHT, FACTOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR, PIXEL_MODE
  PIXEL_MODE = True
  WINDOW_WIDTH = PIXEL_WIDTH * PIXEL_SCALE
  WINDOW_HEIGHT = PIXEL_HEIGHT * PIXEL_SCALE
  FACTOR = WINDOW_WIDTH / REF_W
  AGENT_LEFT_COLOR = PIXEL_AGENT_LEFT_COLOR
  AGENT_RIGHT_COLOR = PIXEL_AGENT_RIGHT_COLOR

def upsize_image(img):
  return cv2.resize(img, (PIXEL_WIDTH * PIXEL_SCALE, PIXEL_HEIGHT * PIXEL_SCALE), interpolation=cv2.INTER_NEAREST)
def downsize_image(img):
  return cv2.resize(img, (PIXEL_WIDTH, PIXEL_HEIGHT), interpolation=cv2.INTER_AREA)

# conversion from space to pixels (allows us to render to diff resolutions)
def toX(x):
  return (x+REF_W/2)*FACTOR
def toP(x):
  return (x)*FACTOR
def toY(y):
  return y*FACTOR

class DelayScreen:
  """ initially the ball is held still for INIT_DELAY_FRAMES(30) frames """
  def __init__(self, life=INIT_DELAY_FRAMES):
    self.life = 0
    self.reset(life)
  def reset(self, life=INIT_DELAY_FRAMES):
    self.life = life
  def status(self):
    if (self.life == 0):
      return True
    self.life -= 1
    return False

def make_half_circle(radius=10, res=20, filled=True):
  """ helper function for pyglet renderer"""
  points = []
  for i in range(res+1):
    ang = math.pi-math.pi*i / res
    points.append((math.cos(ang)*radius, math.sin(ang)*radius))
  if filled:
    return rendering.FilledPolygon(points)
  else:
    return rendering.PolyLine(points, True)

def _add_attrs(geom, color):
  """ help scale the colors from 0-255 to 0.0-1.0 (pyglet renderer) """
  r = color[0]
  g = color[1]
  b = color[2]
  geom.set_color(r/255., g/255., b/255.)

def create_canvas(canvas, c):
  if PIXEL_MODE:
    result = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    for channel in range(3):
      result[:, :, channel] *= c[channel]
    return result
  else:
    rect(canvas, 0, 0, WINDOW_WIDTH, -WINDOW_HEIGHT, color=BACKGROUND_COLOR)
    return canvas

def rect(canvas, x, y, width, height, color):
  """ Processing style function to make it easy to port p5.js program to python """
  if PIXEL_MODE:
    canvas = cv2.rectangle(canvas, (round(x), round(WINDOW_HEIGHT-y)),
      (round(x+width), round(WINDOW_HEIGHT-y+height)),
      color, thickness=-1, lineType=cv2.LINE_AA)
    return canvas
  else:
    box = rendering.make_polygon([(0,0), (0,-height), (width, -height), (width,0)])
    trans = rendering.Transform()
    trans.set_translation(x, y)
    _add_attrs(box, color)
    box.add_attr(trans)
    canvas.add_onetime(box)
    return canvas

def half_circle(canvas, x, y, r, color):
  """ Processing style function to make it easy to port p5.js program to python """
  if PIXEL_MODE:
    return cv2.ellipse(canvas, (round(x), WINDOW_HEIGHT-round(y)),
      (round(r), round(r)), 0, 0, -180, color, thickness=-1, lineType=cv2.LINE_AA)
  else:
    geom = make_half_circle(r)
    trans = rendering.Transform()
    trans.set_translation(x, y)
    _add_attrs(geom, color)
    geom.add_attr(trans)
    canvas.add_onetime(geom)
    return canvas

def circle(canvas, x, y, r, color):
  """ Processing style function to make it easy to port p5.js program to python """
  if PIXEL_MODE:
    return cv2.circle(canvas, (round(x), round(WINDOW_HEIGHT-y)), round(r),
      color, thickness=-1, lineType=cv2.LINE_AA)
  else:
    geom = rendering.make_circle(r, res=40)
    trans = rendering.Transform()
    trans.set_translation(x, y)
    _add_attrs(geom, color)
    geom.add_attr(trans)
    canvas.add_onetime(geom)
    return canvas

class Particle:
  """ used for the ball, and also for the round stub above the fence """
  def __init__(self, x, y, vx, vy, r, c):
    self.x = x
    self.y = y
    self.prev_x = self.x
    self.prev_y = self.y
    self.vx = vx
    self.vy = vy
    self.r = r
    self.c = c
  def display(self, canvas):
    return circle(canvas, toX(self.x), toY(self.y), toP(self.r), color=self.c)
  def move(self):
    self.prev_x = self.x
    self.prev_y = self.y
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
  def applyAcceleration(self, ax, ay):
    self.vx += ax * TIMESTEP
    self.vy += ay * TIMESTEP
  def checkEdges(self):
    if (self.x<=(self.r-REF_W/2)):
      self.vx *= -FRICTION
      self.x = self.r-REF_W/2+NUDGE*TIMESTEP

    if (self.x >= (REF_W/2-self.r)):
      self.vx *= -FRICTION;
      self.x = REF_W/2-self.r-NUDGE*TIMESTEP

    if (self.y<=(self.r+REF_U)):
      self.vy *= -FRICTION
      self.y = self.r+REF_U+NUDGE*TIMESTEP
      if (self.x <= 0):
        return -1
      else:
        return 1
    if (self.y >= (REF_H-self.r)):
      self.vy *= -FRICTION
      self.y = REF_H-self.r-NUDGE*TIMESTEP
    # fence:
    if ((self.x <= (REF_WALL_WIDTH/2+self.r)) and (self.prev_x > (REF_WALL_WIDTH/2+self.r)) and (self.y <= REF_WALL_HEIGHT)):
      self.vx *= -FRICTION
      self.x = REF_WALL_WIDTH/2+self.r+NUDGE*TIMESTEP

    if ((self.x >= (-REF_WALL_WIDTH/2-self.r)) and (self.prev_x < (-REF_WALL_WIDTH/2-self.r)) and (self.y <= REF_WALL_HEIGHT)):
      self.vx *= -FRICTION
      self.x = -REF_WALL_WIDTH/2-self.r-NUDGE*TIMESTEP
    return 0;
  def getDist2(self, p): # returns distance squared from p
    dy = p.y - self.y
    dx = p.x - self.x
    return (dx*dx+dy*dy)
  def isColliding(self, p): # returns true if it is colliding w/ p
    r = self.r+p.r
    return (r*r > self.getDist2(p)) # if distance is less than total radius, then colliding.
  def bounce(self, p): # bounce two balls that have collided (this and that)
    abx = self.x-p.x
    aby = self.y-p.y
    abd = math.sqrt(abx*abx+aby*aby)
    abx /= abd # normalize
    aby /= abd
    nx = abx # reuse calculation
    ny = aby
    abx *= NUDGE
    aby *= NUDGE
    while(self.isColliding(p)):
      self.x += abx
      self.y += aby
    ux = self.vx - p.vx
    uy = self.vy - p.vy
    un = ux*nx + uy*ny
    unx = nx*(un*2.) # added factor of 2
    uny = ny*(un*2.) # added factor of 2
    ux -= unx
    uy -= uny
    self.vx = ux + p.vx
    self.vy = uy + p.vy
  def limitSpeed(self, minSpeed, maxSpeed):
    mag2 = self.vx*self.vx+self.vy*self.vy;
    if (mag2 > (maxSpeed*maxSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vx *= maxSpeed
      self.vy *= maxSpeed

    if (mag2 < (minSpeed*minSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vx *= minSpeed
      self.vy *= minSpeed

class Wall:
  """ used for the fence, and also the ground """
  def __init__(self, x, y, w, h, c):
    self.x = x;
    self.y = y;
    self.w = w;
    self.h = h;
    self.c = c
  def display(self, canvas):
    return rect(canvas, toX(self.x-self.w/2), toY(self.y+self.h/2), toP(self.w), toP(self.h), color=self.c)

class RelativeState:
  """
  keeps track of the obs.
  Note: the observation is from the perspective of the agent.
  an agent playing either side of the fence must see obs the same way
  """
  def __init__(self):
    # agent
    self.x = 0
    self.y = 0
    self.vx = 0
    self.vy = 0
    # ball
    self.bx = 0
    self.by = 0
    self.bvx = 0
    self.bvy = 0
    # opponent
    self.ox = 0
    self.oy = 0
    self.ovx = 0
    self.ovy = 0
  def getObservation(self):
    result = [self.x, self.y, self.vx, self.vy,
              self.bx, self.by, self.bvx, self.bvy,
              self.ox, self.oy, self.ovx, self.ovy]
    scaleFactor = 10.0  # scale inputs to be in the order of magnitude of 10 for neural network.
    result = np.array(result) / scaleFactor
    return result

class Agent:
  """ keeps track of the agent in the game. note this is not the policy network """
  def __init__(self, dir, x, y, c):
    self.dir = dir # -1 means left, 1 means right player for symmetry.
    self.x = x
    self.y = y
    self.r = 1.5
    self.c = c
    self.vx = 0
    self.vy = 0
    self.desired_vx = 0
    self.desired_vy = 0
    self.state = RelativeState()
    self.emotion = "happy"; # hehe...
    self.life = MAXLIVES
  def lives(self):
    return self.life
  def setAction(self, action):
    forward = False
    backward = False
    jump = False
    if action[0] > 0:
      forward = True
    if action[1] > 0:
      backward = True
    if action[2] > 0:
      jump = True
    self.desired_vx = 0
    self.desired_vy = 0
    if (forward and (not backward)):
      self.desired_vx = -PLAYER_SPEED_X
    if (backward and (not forward)):
      self.desired_vx = PLAYER_SPEED_X
    if jump:
      self.desired_vy = PLAYER_SPEED_Y
  def move(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
  def step(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
  def update(self):
    self.vy += GRAVITY * TIMESTEP

    if (self.y <= REF_U + NUDGE*TIMESTEP):
      self.vy = self.desired_vy

    self.vx = self.desired_vx*self.dir

    self.move()

    if (self.y <= REF_U):
      self.y = REF_U;
      self.vy = 0;

    # stay in their own half:
    if (self.x*self.dir <= (REF_WALL_WIDTH/2+self.r) ):
      self.vx = 0;
      self.x = self.dir*(REF_WALL_WIDTH/2+self.r)

    if (self.x*self.dir >= (REF_W/2-self.r) ):
      self.vx = 0;
      self.x = self.dir*(REF_W/2-self.r)

  def updateState(self, ball, opponent):
    """ normalized to side, appears different for each agent's perspective"""
    # agent's self
    self.state.x = self.x*self.dir
    self.state.y = self.y
    self.state.vx = self.vx*self.dir
    self.state.vy = self.vy
    # ball
    self.state.bx = ball.x*self.dir
    self.state.by = ball.y
    self.state.bvx = ball.vx*self.dir
    self.state.bvy = ball.vy
    # opponent
    self.state.ox = opponent.x*(-self.dir)
    self.state.oy = opponent.y
    self.state.ovx = opponent.vx*(-self.dir)
    self.state.ovy = opponent.vy

  def getObservation(self):
    return self.state.getObservation()

  def display(self, canvas, bx, by):
    x = self.x
    y = self.y
    r = self.r

    angle = math.pi * 60 / 180
    if self.dir == 1:
      angle = math.pi * 120 / 180
    eyeX = 0
    eyeY = 0

    canvas = half_circle(canvas, toX(x), toY(y), toP(r), color=self.c)

    # track ball with eyes (replace with observed info later):
    c = math.cos(angle)
    s = math.sin(angle)
    ballX = bx-(x+(0.6)*r*c);
    ballY = by-(y+(0.6)*r*s);

    if (self.emotion == "sad"):
      ballX = -self.dir
      ballY = -3

    dist = math.sqrt(ballX*ballX+ballY*ballY)
    eyeX = ballX/dist
    eyeY = ballY/dist

    canvas = circle(canvas, toX(x+(0.6)*r*c), toY(y+(0.6)*r*s), toP(r)*0.3, color=(255, 255, 255))
    canvas = circle(canvas, toX(x+(0.6)*r*c+eyeX*0.15*r), toY(y+(0.6)*r*s+eyeY*0.15*r), toP(r)*0.1, color=(0, 0, 0))

    # draw coins (lives) left
    for i in range(1, self.life):
      canvas = circle(canvas, toX(self.dir*(REF_W/2+0.5-i*2.)), WINDOW_HEIGHT-toY(1.5), toP(0.5), color=COIN_COLOR)

    return canvas

class Game:
  """
  the main slime volley game.
  can be used in various settings, such as ai vs ai, ai vs human, human vs human
  """
  def __init__(self, np_random=np.random):
    self.ball = None
    self.ground = None
    self.fence = None
    self.fenceStub = None
    self.agent_left = None
    self.agent_right = None
    self.delayScreen = None
    self.np_random = np_random
    self.reset()
  def reset(self):
    self.ground = Wall(0, 0.75, REF_W, REF_U, c=GROUND_COLOR)
    self.fence = Wall(0, 0.75 + REF_WALL_HEIGHT/2, REF_WALL_WIDTH, (REF_WALL_HEIGHT-1.5), c=FENCE_COLOR)
    self.fenceStub = Particle(0, REF_WALL_HEIGHT, 0, 0, REF_WALL_WIDTH/2, c=FENCE_COLOR);
    ball_vx = self.np_random.uniform(low=-20, high=20)
    ball_vy = self.np_random.uniform(low=10, high=25)
    self.ball = Particle(0, REF_W/4, ball_vx, ball_vy, 0.5, c=BALL_COLOR);
    self.agent_left = Agent(-1, -REF_W/4, 1.5, c=AGENT_LEFT_COLOR)
    self.agent_right = Agent(1, REF_W/4, 1.5, c=AGENT_RIGHT_COLOR)
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)
    self.delayScreen = DelayScreen()
  def newMatch(self):
    ball_vx = self.np_random.uniform(low=-20, high=20)
    ball_vy = self.np_random.uniform(low=10, high=25)
    self.ball = Particle(0, REF_W/4, ball_vx, ball_vy, 0.5, c=BALL_COLOR);
    self.delayScreen.reset()
  def step(self):
    """ main game loop """

    self.betweenGameControl()
    self.agent_left.update()
    self.agent_right.update()

    if self.delayScreen.status():
      self.ball.applyAcceleration(0, GRAVITY)
      self.ball.limitSpeed(0, MAX_BALL_SPEED)
      self.ball.move()

    if (self.ball.isColliding(self.agent_left)):
      self.ball.bounce(self.agent_left)
    if (self.ball.isColliding(self.agent_right)):
      self.ball.bounce(self.agent_right)
    if (self.ball.isColliding(self.fenceStub)):
      self.ball.bounce(self.fenceStub)

    # negated, since we want reward to be from the persepctive of right agent being trained.
    result = -self.ball.checkEdges()

    if (result != 0):
      self.newMatch() # not reset, but after a point is scored
      if result < 0: # baseline agent won
        self.agent_left.emotion = "happy"
        self.agent_right.emotion = "sad"
        self.agent_right.life -= 1
      else:
        self.agent_left.emotion = "sad"
        self.agent_right.emotion = "happy"
        self.agent_left.life -= 1
      return result

    # update internal states (the last thing to do)
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)

    return result
  def display(self, canvas):
    # background color
    # if PIXEL_MODE is True, canvas is an RGB array.
    # if PIXEL_MODE is False, canvas is viewer object
    canvas = create_canvas(canvas, c=BACKGROUND_COLOR)
    canvas = self.fence.display(canvas)
    canvas = self.fenceStub.display(canvas)
    canvas = self.agent_left.display(canvas, self.ball.x, self.ball.y)
    canvas = self.agent_right.display(canvas, self.ball.x, self.ball.y)
    canvas = self.ball.display(canvas)
    canvas = self.ground.display(canvas)
    return canvas
  def betweenGameControl(self):
    agent = [self.agent_left, self.agent_right]
    if (self.delayScreen.life > 0):
      pass
      '''
      for i in range(2):
        if (agent[i].emotion == "sad"):
          agent[i].setAction([0, 0, 0]) # nothing
      '''
    else:
      agent[0].emotion = "happy"
      agent[1].emotion = "happy"

class SlimeVolleyAdvEnv(gym.Env):
  """
  Gym wrapper for Slime Volley game.

  By default, the agent you are training controls the right agent
  on the right. The agent on the left is controlled by the baseline
  RNN policy.

  Game ends when an agent loses 5 matches (or at t=3000 timesteps).

  Note: Optional mode for MARL experiments, like self-play which
  deviates from Gym env. Can be enabled via supplying optional action
  to override the default baseline agent's policy:

  obs1, reward, done, info = env.step(action1, action2)

  the next obs for the right agent is returned in the optional
  fourth item from the step() method.

  reward is in the perspective of the right agent so the reward
  for the left agent is the negative of this number.
  """
  metadata = {
    'render.modes': ['human', 'rgb_array', 'state'],
    'video.frames_per_second' : 50
  }

  # for compatibility with typical atari wrappers
  atari_action_meaning = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
  }
  atari_action_set = {
    0, # NOOP
    4, # LEFT
    7, # UPLEFT
    2, # UP
    6, # UPRIGHT
    3, # RIGHT
  }

  action_table = [[0, 0, 0], # NOOP
                  [1, 0, 0], # LEFT (forward)
                  [1, 0, 1], # UPLEFT (forward jump)
                  [0, 0, 1], # UP (jump)
                  [0, 1, 1], # UPRIGHT (backward jump)
                  [0, 1, 0]] # RIGHT (backward)

  from_pixels = False
  atari_mode = False
  survival_bonus = False # Depreciated: augment reward, easier to train
  multiagent = True # optional args anyways

  def __init__(self):
    """
    Reward modes:

    net score = right agent wins minus left agent wins

    0: returns net score (basic reward)
    1: returns 0.01 x number of timesteps (max 3000) (survival reward)
    2: sum of basic reward and survival reward

    0 is suitable for evaluation, while 1 and 2 may be good for training

    Setting multiagent to True puts in info (4th thing returned in stop)
    the otherObs, the observation for the other agent. See multiagent.py

    Setting self.from_pixels to True makes the observation with multiples
    of 84, since usual atari wrappers downsample to 84x84
    """

    self.t = 0
    self.t_limit = 3000

    #self.action_space = spaces.Box(0, 1.0, shape=(3,))
    if self.atari_mode:
      self.action_space = spaces.Discrete(6)
    else:
      self.action_space = spaces.MultiBinary(3)

    if self.from_pixels:
      setPixelObsMode()
      self.observation_space = spaces.Box(low=0, high=255,
        shape=(PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
    else:
      high = np.array([np.finfo(np.float32).max] * 12)
      self.observation_space = spaces.Box(-high, high)
    self.canvas = None
    self.previous_rgbarray = None

    self.game = Game()
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function

    self.policy = PPO1.load(PRETRAINED_MODEL_PATH) # the “bad guy”

    self.viewer = None

    # another avenue to override the built-in AI's action, going past many env wraps:
    self.otherAction = None

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.game = Game(np_random=self.np_random)
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function
    return [seed]

  def getObs(self):
    if self.from_pixels:
      obs = self.render(mode='state')
      self.canvas = obs
    else:
      obs = self.game.agent_right.getObservation()
    return obs

  def discreteToBox(self, n):
    # convert discrete action n into the actual triplet action
    if isinstance(n, (list, tuple, np.ndarray)): # original input for some reason, just leave it:
      if len(n) == 3:
        return n
    assert (int(n) == n) and (n >= 0) and (n < 6)
    return self.action_table[n]

  def step(self, action, otherAction=None):
    """
    baseAction is only used if multiagent mode is True
    note: although the action space is multi-binary, float vectors
    are fine (refer to setAction() to see how they get interpreted)
    """
    done = False
    self.t += 1

    if self.otherAction is not None:
      otherAction = self.otherAction
    
    # pdb.set_trace()
    if otherAction is None: # override baseline policy
      obs = self.game.agent_left.getObservation()
      otherAction = self.policy.predict(obs)

    otherAction = otherAction[0]

    if self.atari_mode:
      action = self.discreteToBox(action)
      otherAction = self.discreteToBox(otherAction)

    self.game.agent_left.setAction(otherAction)
    self.game.agent_right.setAction(action) # external agent is agent_right

    reward = self.game.step()

    obs = self.getObs()

    if self.t >= self.t_limit:
      done = True

    if self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
      done = True

    otherObs = None
    if self.multiagent:
      if self.from_pixels:
        otherObs = cv2.flip(obs, 1) # horizontal flip
      else:
        otherObs = self.game.agent_left.getObservation()

    info = {
      'ale.lives': self.game.agent_right.lives(),
      'ale.otherLives': self.game.agent_left.lives(),
      'otherObs': otherObs,
      'state': self.game.agent_right.getObservation(),
      'otherState': self.game.agent_left.getObservation(),
    }

    if self.survival_bonus:
      return obs, reward+0.01, done, info
    return obs, reward, done, info

  def init_game_state(self):
    self.t = 0
    self.game.reset()

  def reset(self):
    self.init_game_state()
    return self.getObs()

  def checkViewer(self):
    # for opengl viewer
    if self.viewer is None:
      checkRendering()
      self.viewer = rendering.SimpleImageViewer(maxwidth=2160) # macbook pro resolution

  def render(self, mode='human', close=False):

    if PIXEL_MODE:
      if self.canvas is not None: # already rendered
        rgb_array = self.canvas
        self.canvas = None
        if mode == 'rgb_array' or mode == 'human':
          self.checkViewer()
          larger_canvas = upsize_image(rgb_array)
          self.viewer.imshow(larger_canvas)
          if (mode=='rgb_array'):
            return larger_canvas
          else:
            return

      self.canvas = self.game.display(self.canvas)
      # scale down to original res (looks better than rendering directly to lower res)
      self.canvas = downsize_image(self.canvas)

      if mode=='state':
        return np.copy(self.canvas)

      # upsampling w/ nearest interp method gives a retro "pixel" effect look
      larger_canvas = upsize_image(self.canvas)
      self.checkViewer()
      self.viewer.imshow(larger_canvas)
      if (mode=='rgb_array'):
        return larger_canvas

    else: # pyglet renderer
      if self.viewer is None:
        checkRendering()
        self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)

      self.game.display(self.viewer)
      return self.viewer.render(return_rgb_array = mode=='rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
    
  def get_action_meanings(self):
    return [self.atari_action_meaning[i] for i in self.atari_action_set]

class SurvivalRewardEnv(gym.RewardWrapper):
  def __init__(self, env):
    """
    adds 0.01 to the reward for every timestep agent survives

    :param env: (Gym Environment) the environment
    """
    gym.RewardWrapper.__init__(self, env)

  def reward(self, reward):
    """
    adds that extra survival bonus for living a bit longer!

    :param reward: (float)
    """
    return reward + 0.01

class FrameStack(gym.Wrapper):
  def __init__(self, env, n_frames):
    """Stack n_frames last frames.

    (don't use lazy frames)
    modified from:
    stable_baselines.common.atari_wrappers

    :param env: (Gym Environment) the environment
    :param n_frames: (int) the number of frames to stack
    """
    gym.Wrapper.__init__(self, env)
    self.n_frames = n_frames
    self.frames = deque([], maxlen=n_frames)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                        dtype=env.observation_space.dtype)

  def reset(self):
    obs = self.env.reset()
    for _ in range(self.n_frames):
        self.frames.append(obs)
    return self._get_ob()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.frames.append(obs)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.n_frames
    return np.concatenate(list(self.frames), axis=2)

#####################
# helper functions: #
#####################

def multiagent_rollout(env, policy_right, policy_left, render_mode=False):
  """
  play one agent vs the other in modified gym-style loop.
  important: returns the score from perspective of policy_right.
  """
  obs_right = env.reset()
  obs_left = obs_right # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  t = 0

  while not done:

    action_right = policy_right.predict(obs_right)
    action_left = policy_left.predict(obs_left)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs_right, reward, done, info = env.step(action_right, action_left)
    obs_left = info['otherObs']

    total_reward += reward
    t += 1

    if render_mode:
      env.render()

  return total_reward, t

def render_atari(obs):
  """
  Helper function that takes in a processed obs (84,84,4)
  Useful for visualizing what an Atari agent actually *sees*
  Outputs in Atari visual format (Top: resized to orig dimensions, buttom: 4 frames)
  """
  tempObs = []
  obs = np.copy(obs)
  for i in range(4):
    if i == 3:
      latest = np.copy(obs[:, :, i])
    if i > 0: # insert vertical lines
      obs[:, 0, i] = 141
    tempObs.append(obs[:, :, i])
  latest = np.expand_dims(latest, axis=2)
  latest = np.concatenate([latest*255.0] * 3, axis=2).astype(np.uint8)
  latest = cv2.resize(latest, (84 * 8, 84 * 4), interpolation=cv2.INTER_NEAREST)
  tempObs = np.concatenate(tempObs, axis=1)
  tempObs = np.expand_dims(tempObs, axis=2)
  tempObs = np.concatenate([tempObs*255.0] * 3, axis=2).astype(np.uint8)
  tempObs = cv2.resize(tempObs, (84 * 8, 84 * 2), interpolation=cv2.INTER_NEAREST)
  return np.concatenate([latest, tempObs], axis=0)

####################
# Reg envs for gym #
####################

register(
    id='SlimeVolleyAdv-v0',
    entry_point='slimevolleygym.slimevolley_adversarial:SlimeVolleyAdvEnv'
)