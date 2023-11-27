import os

import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from scipy.spatial.transform import Rotation as R

xml_path = 'drone.xml'  # xml file (assumes this is in the same folder as this file)
simend = 200  # simulation time
print_camera_config = 0  # set to 1 to print camera config
# this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model, data):
    # initialize the controller here. This function is called once, in the beginning
    pass


def controller(model, data):
    # put the controller here. This function is called inside the simulation.
    pass


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)



def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height,
                      dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)


# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data
cam = mj.MjvCamera()  # Abstract camera
opt = mj.MjvOption()  # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# cam.azimuth = -90.68741727466428;
# cam.elevation = -2.8073894766455036;
cam.distance = 10
cam.lookat = np.array([0.0, 0.0, 3.0])

# initialize the controller
init_controller(model, data)
prev_x = data.xpos[1]
time_prev = data.time
# set the controller
mj.set_mjcb_control(controller)

def rotation_matrix_to_euler(rot_matrix):
    r = R.from_matrix(rot_matrix)
    euler = r.as_euler('xyz', degrees=True)
    euler = euler/180*np.pi
    return euler

def angvel_to_euler_rate(w, angle):
    J = [[1,np.sin(angle[0])*np.tan(angle[1]),np.cos(angle[0])*np.tan(angle[1])],
         [0, +np.cos(angle[0]), -np.sin(angle[0])],
         [0, np.sin(angle[0])/np.cos(angle[1]), np.cos(angle[0])/np.cos(angle[1])]]
    return J @ w
m =2.5
l = .2
rho = 0.1
inertia = np.array([2,2,3])
x_d = [0,1,2,0,0,0]
idx=0
while not glfw.window_should_close(window):
    while (data.time - time_prev < 1.0 / 60.0):

        r_mat = data.xmat[1].reshape(3, 3)
        angle = rotation_matrix_to_euler(r_mat)
        x_dot = np.zeros(6)
        mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY,
                             1, x_dot, 0)
        x_dot[0:3] = angvel_to_euler_rate( x_dot[0:3],angle)
        x_dot = np.array([x_dot[3],x_dot[4],x_dot[5],x_dot[0],x_dot[1],x_dot[2]])
        ftou = np.array([1/m*np.array([1, 1, 1, 1]),
                l/inertia[0]*np.array([0, 1, 0, -1]),
                l/inertia[1] *np.array([-1,0,1,0]),
                rho/inertia[2] *np.array([1,-1,1,-1])])
        if r_mat[2,2]==0 and r_mat[1,2]==0:
            r_mat = np.eye(3)
        K=np.diag([1,1,0.7,0.5,0.5,1])
        C= np.diag([5,10,1,1,1,1])

        x = np.concatenate((data.xpos[1],angle))
        e= x-x_d
        s= x_dot + K @ e
        G_ = np.array([[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[1,0,0,0,0,0],[0,1,0,0,0,0]])
        u = G_ @ ([0,0,9.81,0,0,0] -K @ x_dot -0.1*np.sign(s))
        f = np.array(np.linalg.inv(ftou) @ u[0:4])
        data.ctrl[0:4] = f
        idx+=1
        if idx%1000==0:
            print("pos", data.xpos[1])
            print("x_dot",x_dot)
            print("s",s)
            print("e",e)
            print("u",u)
            print("f",f)
            print(1/4*m*9.81)
            print(np.linalg.inv(ftou))
        mj.mj_step(model, data)
    if (data.time >= simend):
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    time_prev = data.time
    # print camera configuration (help to initialize the view)
    if (print_camera_config == 1):
        print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =', cam.elevation, ';', 'cam.distance = ', cam.distance)
        print('cam.lookat =np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
