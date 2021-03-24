#!/usr/bin/env python3
import rospy
import tf
import numpy as np
from expert_demo_ros.msg import PoseTwistStamped, Float32MultiArrayStamped
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Bool

# turn into ROSPARAMS
SIM_RATE = 500.0
dt = 1.0/SIM_RATE
V_MIN = rospy.get_param("V_MIN", 0.0)
V_MAX = rospy.get_param("V_MAX", 5.0)
A_MAX = rospy.get_param("A_MAX", 3.0)
OM_MAX = rospy.get_param("OM_MAX", 0.2)
DELTA_MAX = rospy.get_param("DELTA_MAX", 0.344)
LF = rospy.get_param("LF", 0.7)
LR = rospy.get_param("LR", 0.5)
CAR_SIZE = LF + LR

MODEL = rospy.get_param("MODEL", "kinematic bicycle")

def wrapToPi(a):
    if isinstance(a, list):    # backwards compatibility for lists (distinct from np.array)
        return [(x + np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi


class RobotSimDynUnicycle:
    def __init__(self):
        rospy.init_node('robot_sim', anonymous=True)
        self.state_pub = rospy.Publisher('/robot/state', PoseTwistStamped, queue_size=1)

        rospy.Subscriber('/robot/control', Float32MultiArrayStamped, self.control_callback)
        rospy.Subscriber('/robot/init_state', PoseTwistStamped, self.init_state_callback)
        rospy.Subscriber('/robot/state', PoseTwistStamped, self.state_callback)
        rospy.Subscriber('/robot/reset', Bool, self.reset_callback)
        self.state = None
        self.ctrl = None
        self.tf_broadcaster = tf.TransformBroadcaster()

    def reset_callback(self, msg):
        if msg.data == True:
            self.state = None
            self.ctrl = None

    def control_callback(self, msg):
        self.ctrl = msg


    def init_state_callback(self, msg):
        self.state = msg

    def state_callback(self, msg):
        self.state = msg


    def propagate(self, state_msg, ctrl_msg):
        x, y = state_msg.pose.position.x, state_msg.pose.position.y
        quat = state_msg.pose.orientation
        euler = list(tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w]))
        th = euler[2]
        V = state_msg.twist.linear.x

        # om = ctrl_msg.data.data[0] * OM_MAX
        # a = ctrl_msg.data.data[1] * A_MAX

        brake = -(ctrl_msg.data.data[0]) * A_MAX
        accel = (ctrl_msg.data.data[1]) * A_MAX 
        a = brake + accel 
        om = ctrl_msg.data.data[2] * OM_MAX

        state_msg.pose.position.x += V*np.cos(th)*dt
        state_msg.pose.position.y += V*np.sin(th)*dt
        euler[2] = wrap2Pi(th + om*dt)
        state_msg.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(*euler))
        state_msg.twist.linear.x = np.clip(V + a*dt, V_MIN, V_MAX)
        return state_msg



    def run(self):
            rate = rospy.Rate(SIM_RATE) #100 Hz
            while not rospy.is_shutdown():
                if (self.state is not None) and (self.ctrl is not None):
                    new_state = self.propagate(self.state, self.ctrl)
                    new_state.header.stamp = rospy.Time.now()
                    trans = (new_state.pose.position.x, new_state.pose.position.y, new_state.pose.position.z)
                    rot = (new_state.pose.orientation.x, new_state.pose.orientation.y, new_state.pose.orientation.z,  new_state.pose.orientation.w)
                    self.tf_broadcaster.sendTransform(trans, rot, new_state.header.stamp, 'robot', 'world')

                    self.state_pub.publish(new_state)


                rate.sleep()



class RobotSimKinematicBicycle:
    def __init__(self, lr=LR, lf=LF):
        rospy.init_node('robot_sim', anonymous=True)
        self.state_pub = rospy.Publisher('/robot/state', PoseTwistStamped, queue_size=1)

        rospy.Subscriber('/robot/control', Float32MultiArrayStamped, self.control_callback)
        rospy.Subscriber('/robot/init_state', PoseTwistStamped, self.init_state_callback)
        rospy.Subscriber('/robot/state', PoseTwistStamped, self.state_callback)
        rospy.Subscriber('/robot/reset', Bool, self.reset_callback)
        self.state = None
        self.ctrl = None
        self.lr = lr
        self.lf = lf
        self.tf_broadcaster = tf.TransformBroadcaster()

    def reset_callback(self, msg):
        if msg.data == True:
            self.state = None
            self.ctrl = None

    def control_callback(self, msg):
        self.ctrl = msg


    def init_state_callback(self, msg):
        self.state = msg

    def state_callback(self, msg):
        self.state = msg


    def propagate(self, state_msg, ctrl_msg):
        x, y = state_msg.pose.position.x, state_msg.pose.position.y
        quat = state_msg.pose.orientation
        euler = list(tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w]))
        th = euler[2]
        V = state_msg.twist.linear.x

        brake = -(ctrl_msg.data.data[0]) * A_MAX
        accel = ( ctrl_msg.data.data[1]) * A_MAX 
        a = brake + accel 
        delta = ctrl_msg.data.data[2] * DELTA_MAX

        beta = np.arctan(self.lr / (self.lr + self.lf) * np.tan(delta))

        state_msg.pose.position.x += V*np.cos(th + beta)*dt
        state_msg.pose.position.y += V*np.sin(th + beta)*dt
        euler[2] = wrapToPi(th + V * np.sin(beta) / self.lr * dt)
        state_msg.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(*euler))
        state_msg.twist.linear.x = np.clip(V + a*dt, V_MIN, V_MAX)
        state_msg.twist.linear.y = beta
        return state_msg



    def run(self):
            rate = rospy.Rate(SIM_RATE) #100 Hz
            while not rospy.is_shutdown():
                if (self.state is not None):
                    new_state = self.state
                    if (self.ctrl is not None):
                        new_state = self.propagate(self.state, self.ctrl)
                    new_state.header.stamp = rospy.Time.now()
                    trans = (new_state.pose.position.x, new_state.pose.position.y, new_state.pose.position.z)
                    rot = (new_state.pose.orientation.x, new_state.pose.orientation.y, new_state.pose.orientation.z,  new_state.pose.orientation.w)
                    self.tf_broadcaster.sendTransform(trans, rot, new_state.header.stamp, 'robot', 'world')

                    self.state_pub.publish(new_state)


                rate.sleep()

if __name__ == '__main__':
    if MODEL == "kinematic bicycle":
        tele = RobotSimKinematicBicycle()
    elif MODEL == "dynamic unicycle":
        tele = RobotSimDynUnicycle()
    tele.run()

