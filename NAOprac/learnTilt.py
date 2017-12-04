#Nao imports
import almath
import argparse
from naoqi import ALProxy
import motion
import numpy as np
import time
import random as rnd
import copy
import images2
import cv2

#Accelerometer imports
import socket
import traceback


use_nao = True
use_accelerometer = False
#Nao set up
#def setup_nao():
if use_nao:
    NAO_PORT = 9559
    NAO_IP = "172.22.0.2"
    motionProxy = ALProxy("ALMotion", NAO_IP, NAO_PORT)
    autonomousLifeProxy = ALProxy("ALAutonomousLife", NAO_IP, NAO_PORT)
    postureProxy = ALProxy("ALRobotPosture", NAO_IP, NAO_PORT)
    tracker = ALProxy("ALTracker", NAO_IP, NAO_PORT)
    awareness = ALProxy('ALBasicAwareness', NAO_IP, NAO_PORT)
    camProxy = ALProxy("ALVideoDevice", NAO_IP, NAO_PORT)

    client = "python_GVM"
    resolution = 0    # 0: 160x120, 1: 320x240, 2: 640x480, 3: 1280x960
    colorSpace = 13   # BGR (because OpenCV)
    fps = 30
    # if you only want one camera
    videoClient = camProxy.subscribeCamera(client, 0, resolution, colorSpace, fps)

def flip_angles(angs, js):
    joints_flipped = copy.copy(js)
    angs_flipped = copy.copy(angs)
    for j1 in range(0, len(js)):
        joint1_name = js[j1]
        if joint1_name[0] == 'L':
            rest_of_joint1_name = joint1_name[1:]
            for j2 in range(0, len(js)):
                joint2_name = js[j2]
                rest_of_joint2_name = joint2_name[1:]
                if joint2_name[0] == 'R' and rest_of_joint1_name == rest_of_joint2_name and not ("Pitch" in rest_of_joint1_name):
                    angs_flipped[j1] = -angs[j2]
                    angs_flipped[j2] = -angs[j1]
                elif joint2_name[0] == 'R' and rest_of_joint1_name == rest_of_joint2_name:
                    angs_flipped[j1] = angs[j2]
                    angs_flipped[j2] = angs[j1]
    return angs_flipped

# Currently has 2 of the middle value if number of divisions is odd
def interpolate_angles(angs1, angs2, mid_angs, interpolation_value, num_interpolations):
    if num_interpolations % 2 == 0:
        diff_divider = (num_interpolations-1)//2
    else:
        diff_divider = num_interpolations//2
    new_angs = [0 for i in angs1]
    if interpolation_value == 0:
        new_angs = angs1
    elif interpolation_value == num_interpolations - 1:
        new_angs = angs2
    elif interpolation_value > num_interpolations//2:
        for i in range(0, len(angs2)):
            diff = (angs2[i] - mid_angs[i]) / diff_divider
            new_angs[i] = mid_angs[i] + diff * (interpolation_value - (num_interpolations//2))
            print diff * (interpolation_value - (num_interpolations//2))
    elif interpolation_value < num_interpolations//2:
        for i in range(0, len(angs1)):
            diff = (mid_angs[i] - angs1[i]) / diff_divider
            new_angs[i] = angs1[i] + diff * interpolation_value
            print diff * interpolation_value
    else:
        new_angs = mid_angs
    return new_angs


def interpolate2(angs1, angs2, mid_angs, interpolation_value, num_interpolations):
    new_angs = [0 for i in angs1]
    proportion1 = float(interpolation_value) / float(num_interpolations)
    proportion2 = 1 - proportion1
    print(proportion1, proportion2)
    for i in range(0, len(angs1)):
        new_angs[i] = proportion1 * angs2[i] + proportion2 * angs1[i]
    return new_angs
    #if interpolation_value < num_interpolations/2.0:
    #    for i in range(0, angs1):
    #        new_angs[i] = proportion1 * angs1[i] + proportion2 * mid_angs[i]
    #else:
    #    for i in range(0, angs1):
    #        new_angs[i] = proportion1 * mid_angs[i] + proportion2 * angs2[i]



joints = ["LShoulderPitch", "LShoulderRoll",
          "RShoulderPitch", "RShoulderRoll",
          "LElbowYaw", "LElbowRoll", "LWristYaw",
          "RElbowYaw", "RElbowRoll", "RWristYaw"]
starting_angles = [0.4862360954284668, -0.1595778465270996,
                   0.5860300064086914, 0.13955211639404297,
                   -1.4680800437927246, -0.6335000991821289, 0,#-1.5846638679504395,
                   1.5431621074676514, 0.7731781005859375, 0]#1.3376060724258423]

tilted1 = [1.0931761264801025, -0.019782446324825287, 1.305920124053955, 0.13431110978126526, -1.4732160568237305, -0.9601529240608215, -0.09821796417236328, 1.6077550649642944, 1.1949901580810547, -0.21786999702453613]
tilted2 = [0.9257596135139465, 0.006530190818011761, 1.070677638053894, 0.16724123060703278, -1.4790945053100586, -1.2914506196975708, -0.13810205459594727, 1.6072843074798584, 1.4463956356048584, -0.23014187812805176]
tilted3 = [1.0816580057144165, 0.023215852677822113, 1.1921952962875366, 0.17280954122543335, -1.4838372468948364, -1.179466962814331, -0.15497589111328125, 1.6077502965927124, 1.3238054513931274, -0.257753849029541]
tilted4 = [1.122732400894165, -0.07589391618967056, 1.138623833656311, 0.12426231801509857, -1.472051739692688, -0.8558668494224548, -0.5384759902954102, 1.5980267524719238, 1.33295476436615, -0.4817180633544922]

tiltLeft = [0.18250393867492676, -0.08748006820678711, 0.6727747321128845, 0.14530789852142334, -1.4174580574035645, -0.5, 0.15029001235961914, 1.735066294670105, 0.9, -0.07674193382263184]#3313021659851074, 7286896109580994#[0.846985399723053, -0.09535059332847595, 0.14883995056152344, 0.07359004020690918, -1.5987539291381836, -0.8236891627311707, -0.25315189361572266, 1.498676061630249, 0.08287787437438965, -0.3237159252166748]
tiltRight = flip_angles(tiltLeft, joints)#[0.4985031187534332, -0.036845769733190536, 0.617575466632843, 0.14542429149150848, -1.5432469844818115, -0.8221817016601562, 0.22545599937438965, 1.5816208124160767, 0.49399709701538086, 0.19631004333496094]
tiltForward = [0.8811683654785156, -0.03177360072731972, 0.6131321787834167, 0.11051787436008453, -1.5448977947235107, -0.44484609365463257, -0.1764519214630127, 1.5355136394500732, 0.12280456721782684, -0.09668397903442383]
tiltBackward = [0.7209112048149109, 0.049126025289297104, 0.4720867872238159, 0.1430918127298355, -1.5447890758514404, -1.0200631618499756, 0.08125996589660645, 1.5432122945785522, 0.3758842647075653, 0.0858621597290039]

handTiltRight = [0.19430837035179138, 0.09112875163555145, 0.06000255048274994, -1.7133439779281616, -0.21396972239017487, 0.043423689901828766]
handTiltLeft = [0.21018663048744202, 0.0749436765909195, 0.11247345060110092, -1.2786052227020264, -0.31353920698165894, -0.157943457365036]

shoulder_roll_joints = ["LShoulderRoll", "RShoulderRoll"]
shoulder_roll_angles = [-0.1, 0.1]
hip_joints = ["LHipYawPitch"]
hip_angles = [0.25]
#Accelerometer set up
if use_accelerometer:
    ACC_IP = "172.22.0.3"
    ACC_PORT = 5555
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind((ACC_IP, ACC_PORT))

#qwe = 5
#for i in range(0,qwe):
#    print(interpolate_angles([-1.0, -2.0, 5.0, 10.0], [1.0, -10.0, 11.0, 10.0], [0.0, -6.0, 8.0, 0.0], i, qwe))
#print tiltLeft
#print flip_angles(tiltLeft, joints)


def get_acc_data(num_readings, sock):
    x = y = z = 0
    for i in range(0, num_readings):
        try:
            message, address = sock.recvfrom(8192)
            data = str(message).split(",")
            x += float(data[2]) #axis from home button to samsung logo
            y += float(data[3]) #axis from volume button to on/off button
            z += float(data[4])
        except (KeyboardInterrupt, SystemExit):
            return -1
    return x/num_readings, y/num_readings, z/num_readings


def open_hands(mp):
    hands = ["RHand", "LHand"]
    angs = [1, 1]
    mp.setAngles(hands, angs, speed)


def grab_hands(mp):
    hands = ["RHand", "LHand"]
    angs = [0, 0]
    mp.setAngles(hands, angs, speed)

def shoulder_roll(mp):
    mp.setAngles(shoulder_roll_joints, shoulder_roll_angles, speed)

def move_shoulders(mp, rang, lang):
    shols = ["RShoulderPitch", "LShoulderPitch"]
    angs = [rang, lang]
    mp.setAngles(shols, angs, speed)

speed = 0.2
isAbsolute = True


def set_up_first_time():
    autonomousLifeProxy.setState("disabled")
    motionProxy.setBreathEnabled("Body", False)
    motionProxy.setIdlePostureEnabled("Body", False)
    motionProxy.setStiffnesses("Body", 0.1)
    awareness.stopAwareness()


def set_up():
    postureProxy.goToPosture("StandInit", 0.5)
    motionProxy.setAngles(hip_joints, hip_angles, 0.1)
    motionProxy.setAngles(joints, starting_angles, speed)
    motionProxy.setAngles(shoulder_roll_joints, shoulder_roll_angles, speed)

    targetName = "RedBall"
    diameterOfBall = 0.04
    tracker.registerTarget(targetName, diameterOfBall)
    tracker.track(targetName)    


def set_all_stiff():
    motionProxy.setStiffnesses(joints, 1)
    #motionProxy.setStiffness(hip_joints, 1)
    motionProxy.setStiffnesses(shoulder_roll_joints, 1)


def relax_all():
    motionProxy.setStiffnesses(joints, 0.1)
    #motionProxy.setStiffness(hip_joints, 0.1)
    motionProxy.setStiffnesses(shoulder_roll_joints, 0.1)

population = []
population_scores = []
population_size = 5
mutation_chance = 0.5
learning = True
def generate_new_population(pop_size, mutate_chance, init_angles):
    pop = []
    pop_scores = []
    for i in range(0, pop_size):
        motionProxy.setAngles(joints, init_angles, speed)
        time.sleep(0.25)
        entity = copy.copy(init_angles)
        for a in range(0, len(entity)):
            if rnd.random() < mutate_chance:
                change = rnd.randint(-50, 50)/250.0
                entity[a] += change
        motionProxy.setAngles(joints, entity, speed)
        time.sleep(0.5)
        pop.append(entity)
        (xpos, ypos, _) = get_acc_data(5, s)
        fitness = (xpos, ypos)
        pop_scores.append(fitness)
        #time.sleep(0.5)
    return pop, pop_scores


num_iterations = 20


def fittest_member(pop, scores):
    ideal_x = 0
    ideal_y = 3
    best_x = -1
    best_y = -1
    best_x_diff = 100
    best_y_diff = 100
    for i in range(0, len(scores)):
        x_diff = abs(scores[i][0] - ideal_x)
        y_diff = abs(scores[i][1] - ideal_y)
        print i, x_diff, y_diff
        if x_diff < best_x_diff:
            best_x_diff = x_diff
            best_x = i
        if y_diff < best_y_diff:
            best_y_diff = y_diff
            best_y = i
    fittest = -1
    if rnd.random() < 0.8:
        fittest = best_y
    else:
        fittest = best_x
    return fittest


def run_genetic_algorithm(num_its, start_angs):
    angs = start_angs
    for i in range(0, num_its):
        population, population_scores = generate_new_population(population_size, mutation_chance, angs)
        best_angles = fittest_member(population, population_scores)
        print "Iteration", i, best_angles#i , population_scores, population_scores[best_angles]
        angs = population[best_angles]
    print angs, best_angles 

def relax():
    motionProxy.setStiffnesses(joints, 0.0)


def go_to_angles(angles, joynts, mp):
    mp.setAngles(joynts, angles, speed)

def set_stiffness(stiff_val, mp):
    mp.setStiffnesses("Body", stiff_val)

def track_red_ball():
    return tracker.getTargetPosition(0)

def head_forward(mp):
    mp.setAngles(["HeadPitch", "HeadYaw"], [0,0], speed)
    awareness.stopAwareness()
    print mp.getAngles(["HeadPitch", "HeadYaw"], False)

def get_tray_angle():
    xl, yl, _, _, _, _ = motionProxy.getPosition("LHand", 2, True)
    xr, yr, _, _, _, _ = motionProxy.getPosition("RHand", 2, True)
    angle = np.arctan((xl-xr)/(yl-yr))
    return angle

def get_ball_x_velocity(prev_x_pos, curr_x_pos, prev_time, curr_time):
    #print(curr_x_pos, prev_pos)
    pos_change = curr_x_pos - prev_x_pos
    time_change = curr_time - prev_time
    return pos_change / time_change


def is_x_velocity_positive(prev_x_pos, curr_x_pos, prev_time, curr_time):
    vel = get_ball_x_velocity(prev_x_pos, curr_x_pos, prev_time, curr_time)
    res = False
    if vel > 0:
        res = True
    return res

def is_x_position_positive(curr_x_pos):
    res = False
    if curr_x_pos > 0:
        res = True
    return res

def get_inputs(prev_x_pos, curr_x_pos, prev_time, curr_time):
    pos = is_x_position_positive(curr_x_pos)
    vel = is_x_velocity_positive(prev_x_pos, curr_x_pos, prev_time, curr_time)
    ang = get_tray_angle()
    return pos, vel, ang


run_nao = True
while run_nao:
    inp = raw_input("Options: rest, open, grab, setup, init, genetic, best, exit: ")
    if inp == 'rest':
        motionProxy.rest()
    elif inp == 'open':
        open_hands(motionProxy)
    elif inp == 'grab':
        grab_hands(motionProxy)
    elif inp == 'setup':
        set_up()
    elif inp == 'init':
        set_up_first_time()
    elif inp == 'capture':
        print motionProxy.getAngles(joints, False)
    elif inp == 'capturewait':
        time.sleep(15)
        print motionProxy.getAngles(joints, False)
    elif inp == "left":
        go_to_angles(tiltLeft, joints, motionProxy)
    elif inp == "right":
        go_to_angles(tiltRight, joints, motionProxy)
    elif inp == "forward":
        go_to_angles(tiltForward, joints, motionProxy)
    elif inp == "back":
        go_to_angles(tiltBackward, joints, motionProxy)
    elif inp == "stiff":
        set_all_stiff()
    elif inp == "relax":
        relax_all()
    elif inp == "shoulder":
        shoulder_roll(motionProxy)
    elif inp == "images":
        images2.showNaoImage()
    elif inp == "interpolate":
        num_interpolations = 14
        start_angs = interpolate2(tiltLeft, tiltRight, starting_angles, 5, 20)
        go_to_angles(start_angs, joints, motionProxy)
        time.sleep(1)
        for i in range(6, num_interpolations+1):
            new_angs = interpolate2(tiltLeft, tiltRight, starting_angles, i, 20)
            go_to_angles(new_angs, joints, motionProxy)
            time.sleep(0.1)
            print track_red_ball()
            print get_tray_angle()
            images = camProxy.getImageRemote(videoClient)
            imgs = images2.toCVImg(images)
            cv2.imshow("image", imgs)
            cv2.waitKey(1)
        time.sleep(5)
        relax_all()
    elif inp == "getpos":
        chainName = "LHand"
        frame     = motion.FRAME_TORSO
        useSensor = False
        # Get the current position of the chainName in the same frame
        current = motionProxy.getPosition(chainName, frame, useSensor)
        print current
    elif inp == "setpos":
        chainName = "LArm"
        frame     = motion.FRAME_ROBOT
        useSensor = False
        # Get the current position of the chainName in the same frame
        motionProxy.setPositions(chainName, frame, handTiltLeft, speed, 63)
    elif inp == "track":
        print track_red_ball()
    elif inp == "trackshow":
        prev_pos = track_red_ball()[0]
        prev_time = time.time()
        time.sleep(0.05)
        for i in range(0, 300):
            #time.sleep(0.5)
            curr_pos = track_red_ball()[0]
            curr_time = time.time()
            images = camProxy.getImageRemote(videoClient)
            imgs = images2.toCVImg(images)
            cv2.imshow("image", imgs)
            cv2.waitKey(1)
            pos, vel, ang = get_inputs(prev_pos, curr_pos, prev_time, curr_time)
            print(pos, vel, ang)
            #print(get_ball_x_velocity(prev_pos, curr_pos, prev_time, curr_time), is_x_velocity_positive(prev_pos, curr_pos, prev_time, curr_time), tracker.isTargetLost())
            prev_pos = curr_pos
            prev_time = curr_time
    elif inp == "angle":
        print get_tray_angle()
    elif inp == "follow":
        for i in range(0, 100):
            print track_red_ball()
    elif inp == "head":
        head_forward(motionProxy)
    elif inp == 'exit':
        run_nao = False


#TODO 
# Map angles of tray in simulation to angles of tray in robot - just restrict the angles that the simulation can go between - or even just scale the angle of the robot
#   tray to be within the bounds of the angles that the simulated tray goes between
# Find a way to get angle of robot tray - get position of hands, and find line between them
#   for tilting forward and back, can probably just use the angle of the elbow
# Make sure vision can track the ball all the way - view image as it turns its head
# Not exact directions does not matter, since has can just split it into x and y coordinates

#THings to do
# get it into a state where there is a get_all_inputs() function that gets all the nexessary information for training

#Things to do later
# training via a video input a la the atari games

#INputs to get
# position of ball - left or right
# speed of ball - positive or negative
# angle of tray
