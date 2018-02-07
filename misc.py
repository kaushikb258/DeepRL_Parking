import numpy as np
import random
import sys


#--------------------------------------------------------------

# limit angle to [-pi,pi] range

def limit_angle(ang):

  while (ang > np.pi):
   ang -= 2.0*np.pi
  while (ang < -np.pi):
   ang += 2.0*np.pi 

  if (ang < -np.pi or ang > np.pi):
    print("angle error ", ang)
    sys.exit()

  return ang

#--------------------------------------------------------------

def calc_init_state(episode, xdim, ydim, carl, carw, train_indicator):

  marginx = 0.55*carl
  marginy = 0.55*carl

  if (train_indicator == 1):

    if (episode <= 10000):
      xl = -xdim/4.0 
      xr = 0.0 
      yt = carw
      yb = -carw  
    elif (episode > 10000 and episode <= 20000):
      xl = -xdim/4.0 - marginx
      xr = -marginx
      yt = carw 
      yb = -ydim/8.0
    elif (episode > 20000 and episode <= 40000):
      xl = -xdim/2.0 - marginx
      xr = -marginx
      yt = -marginy
      yb = -ydim/4.0
    elif (episode > 40000 and episode <= 60000):
      xl = -3.0*xdim/4.0 - marginx
      xr = -marginx
      yt = -marginy 
      yb = -3.0*ydim/8.0
    else:
      xl = -xdim + marginx
      xr = -marginx
      yt = -marginy
      yb = -ydim/2.0 + marginy

  else:
    xl = -xdim + marginx
    xr = -marginx
    yt = -marginy
    yb = -ydim/2.0 + marginy


  # x location (randomly chosen between xl and xr)
  r = np.random.uniform()
  x = r*xl + (1.0-r)*xr

  # y location (randomly chosen between yt and yb)
  r = np.random.uniform()
  y = r*yt + (1.0-r)*yb


  # heading angle
  # for early episodes, make the car point towards goal, with a small perturbation added (for training mode only!)
  # this may speed up learning
  # goal location: (1.5*carl/2,0)
  xgoal = 1.5*carl/2.0
  ygoal = 0.0

  if (episode <= 40000 and train_indicator == 1):
    zi = np.arctan2((ygoal-y),(xgoal-x))  + np.random.randn()*np.pi/18.0
  else: 
    # randomly choose a heading angle between [45,135] deg if bottom half of domain
    # randomly choose a heading angle between [-135,-45] deg if top half of domain
    r  = np.random.uniform()
    if (y < 0.0):
     zi = np.pi/4.0 + np.pi/2.0*r
    else:
     zi = -(np.pi/4.0 + np.pi/2.0*r) 
  
  zi = limit_angle(zi)


  # steering angle
  delta = 0.0

  # velocity
  v = 0.0

  dv_dt = 0.0
  ddelta_dt = 0.0
  

  return x, y, v, zi, delta, dv_dt, ddelta_dt

#--------------------------------------------------------------


def move(dt, v_max, delta_max, L_wheel_2_wheel, x, y, zi, delta, v, dv_dt, ddelta_dt):

  # velocity
  v_new = v + dv_dt*dt
  v_new = min(max(v_new,-v_max),v_max)

  # steering angle
  delta_new = delta + ddelta_dt*dt
  delta_new = min(max(delta_new,-delta_max),delta_max)
  

  # heading angle
  zi_new = zi + dt*v/L_wheel_2_wheel*np.tan(delta)
  zi_new = limit_angle(zi_new)


  #x
  x_new = x + dt*v*np.cos(zi)

  # y
  y_new = y + dt*v*np.sin(zi)    


  x = x_new
  y = y_new
  v = v_new
  zi = zi_new
  delta  = delta_new  

  return x, y, v, zi, delta


#--------------------------------------------------------------

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

#--------------------------------------------------------------

reward_normal = -1.0
reward_crash = -500.0
reward_destination = 5000.0
reward_jerky = 2.0
reward_parking_area = 250.0
reward_gaussian = 0.0


def compute_reward(xdim, ydim, carw, carl, x, y, v, zi, u1, u1_old, u2, u2_old, print_screen):

  rectDiag  = np.sqrt((carl/2.0)*(carl/2.0) + (carw/2.0)*(carw/2.0))
  rectAngle = np.arctan2(carw/2.0, carl/2.0)
  rectAngle = limit_angle(rectAngle)
  # atan2 returns angle in [-pi, pi] range

  #1
  x1 = x + -rectDiag * np.cos(-rectAngle + zi)
  y1 = y + -rectDiag * np.sin(-rectAngle + zi)
   
  #2
  x2 = x + rectDiag * np.cos(rectAngle + zi)
  y2 = y + rectDiag * np.sin(rectAngle + zi)
   
  #3
  x3 = x + rectDiag * np.cos(-rectAngle + zi)
  y3 = y + rectDiag * np.sin(-rectAngle + zi)
   
  #4
  x4 = x + -rectDiag * np.cos(rectAngle + zi)
  y4 = y + -rectDiag * np.sin(rectAngle + zi)

  xright = max(x1,x2,x3,x4)
  xleft = min(x1,x2,x3,x4)
  ytop = max(y1,y2,y3,y4)
  ybottom = min(y1,y2,y3,y4)
  
  A = np.array([x1,y1])
  B = np.array([x2,y2])
  C = np.array([x3,y3])
  D = np.array([x4,y4])

  reward = reward_normal
  done  = False


#-------------------------

  # (new - old) reward

  if (np.abs(u1-u1_old) < 0.2 and np.abs(u2-u2_old) < 0.2):
    reward += reward_jerky
  else:
    reward -= reward_jerky
   
#-------------------------

  # gaussian reward 
  # reward for being close to (xc,yc)
  # this may make the learning faster

  xc = 0.0
  yc = 0.0
  sigma = 2.5

  dist = np.sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc))
  exp_factor = np.exp(-dist*dist/(2.0*sigma*sigma))
  pre_exp_factor = 1.0/np.sqrt(2.0*np.pi*sigma*sigma)
 
  gauss  = pre_exp_factor*exp_factor

  reward += gauss*reward_gaussian

#-------------------------

  # reward for being in the parking area

  if (ytop <= carw and ybottom >= -carw and x >= 0.0):
    reward += reward_parking_area
    done = False

  
  # reward for entering the parking area with an angle close to zero

  if (ytop <= carw and ybottom >= -carw and np.abs(zi) <= np.pi/6.0):
    reward += reward_parking_area
    done = False

#-------------------------

  # check for crash

  # left 
  if (xleft <= -xdim):
    reward += reward_crash
    if (print_screen):
     print("left crash")
    done = True
    return reward, done

  # top  
  elif (ytop >= ydim/2.0):
    reward += reward_crash
    if (print_screen):
     print("top crash")
    done = True
    return reward, done

  # bottom  
  elif (ybottom <= -ydim/2.0):
    reward += reward_crash
    if (print_screen):
     print("bottom crash")
    done = True
    return reward, done


  # right top  
  P1 = np.array([0.0, carw])
  P2 = np.array([0.0, ydim/2.0])
  if (intersect(A,B,P1,P2) or intersect(B,C,P1,P2) or intersect(C,D,P1,P2) or intersect(D,A,P1,P2)):
    reward += reward_crash
    if (print_screen):
     print("top right crash")
    done = True
    return reward, done 

  # right bottom
  P1 = np.array([0.0, -carw])
  P2 = np.array([0.0, -ydim/2.0])
  if (intersect(A,B,P1,P2) or intersect(B,C,P1,P2) or intersect(C,D,P1,P2) or intersect(D,A,P1,P2)):
    reward += reward_crash
    if (print_screen):
     print("bottom right crash")
    done = True
    return reward, done 

  # parking zone top
  P1 = np.array([0.0, carw])
  P2 = np.array([1.5*carl, carw])
  if (intersect(A,B,P1,P2) or intersect(B,C,P1,P2) or intersect(C,D,P1,P2) or intersect(D,A,P1,P2)):
    reward += reward_crash
    if (print_screen):
     print("top parking crash")
    done = True
    return reward, done   

  # parking zone bottom
  P1 = np.array([0.0, -carw])
  P2 = np.array([1.5*carl, -carw])
  if (intersect(A,B,P1,P2) or intersect(B,C,P1,P2) or intersect(C,D,P1,P2) or intersect(D,A,P1,P2)):
    reward += reward_crash
    if (print_screen):
     print("bottom parking crash")
    done = True
    return reward, done

  # parking zone right
  P1 = np.array([1.5*carl, carw])
  P2 = np.array([1.5*carl, -carw])
  if (intersect(A,B,P1,P2) or intersect(B,C,P1,P2) or intersect(C,D,P1,P2) or intersect(D,A,P1,P2)):
    reward += reward_crash
    if (print_screen):
     print("right parking crash")
    done = True
    return reward, done

  
#-------------

  # reached destination
  if (ytop <= carw and ybottom >= -carw and x >= 0.5*1.5*carl):
    print("reached destination!")
    if (abs(zi) <= np.pi/12.0 and np.abs(v) <= 0.5):
      reward += reward_destination      
      done = True
    else:
      # partial bonus
      reward += reward_destination/4.0
      done = True
  return reward, done    


#--------------------------------------------------------------

def compute_statevars(xdim,ydim,carl,x,y):

   # distance to left wall
   d2left = np.abs(-xdim - x)

   # distance to top wall
   d2top = np.abs(ydim/2.0 - y)

   # distance to bottom wall
   d2bottom = np.abs(-ydim/2.0 - y)

   # distance to right wall
   d2right = np.abs(0 - x)

   # goal location: (1.5*carl/2,0)
   xgoal = 1.5*carl/2.0
   ygoal = 0.0

   # distance to goal
   d2goal = np.sqrt((xgoal-x)**2.0 + (ygoal-y)**2.0)

   # angle to goal 
   ang2goal = np.arctan2((ygoal-y),(xgoal-x)) 
   ang2goal = limit_angle(ang2goal)
   

   return d2left, d2top, d2bottom, d2right, d2goal, ang2goal

#--------------------------------------------------------------

