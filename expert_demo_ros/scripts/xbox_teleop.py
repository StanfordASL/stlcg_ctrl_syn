#!/usr/bin/env python 
import rospy
import numpy as np
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from expert_demo_ros.msg import Float32MultiArrayStamped, PoseTwistStamped
from geometry_msgs.msg import Point, Quaternion
import tf

class XboxTeleopSim:
    def __init__(self):
        rospy.init_node('xbox_teleop', anonymous=True)
        self.control_pub = rospy.Publisher('/robot/control', Float32MultiArrayStamped, queue_size=1)
        self.init_state_pub = rospy.Publisher('/robot/init_state', PoseTwistStamped, queue_size=1)
        self.reset_flag_pub = rospy.Publisher('/robot/reset', Bool, queue_size=1)
        rospy.Subscriber('/joy', Joy, self.joy_callback)

        self.control = Float32MultiArrayStamped()
        self.robot_state = None
        self.reset_flag = False
        self.dx = 0.01
        self.tf_broadcaster = tf.TransformBroadcaster()


    def joy_callback(self, msg):
        if not self.reset_flag:
            self.control.header.stamp = rospy.Time.now()
            # self.control.data.data = [msg.axes[0], msg.axes[1]]
            self.control.data.data = [msg.axes[2], msg.axes[5], msg.axes[0]]
        # press the start button to put the robot back to the start region
        if msg.buttons[7] == 1:
            rospy.loginfo("Reset sim")
            self.reset_flag = True
            self.robot_state = PoseTwistStamped()
            self.robot_state.pose.orientation.w = 1.0
            rospy.loginfo("resetting sim mode")

        # if the start button has been pressed, we are in the adjusting the robot's initial state 
        if self.reset_flag:
            self.reset_flag_pub.publish(Bool(True))
            self.robot_state.pose.position.x += self.dx * (-(msg.buttons[2] == 1) + (msg.buttons[1] == 1))
            self.robot_state.pose.position.y += self.dx * (-(msg.buttons[0] == 1) + (msg.buttons[3] == 1))
            quat = self.robot_state.pose.orientation
            euler = list(tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w]))
            euler[2] += self.dx * ((msg.buttons[4] == 1) - (msg.buttons[5] == 1))
            self.robot_state.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(*euler))
            # changing forward velocity (twist.linear.x)
            self.robot_state.twist.linear.x += self.dx * ((msg.axes[4] > 0) - (msg.axes[4] < 0))
            self.robot_state.header.stamp = rospy.Time.now()
            self.init_state_pub.publish(self.robot_state)


        if msg.buttons[6] == 1:
            rospy.loginfo("Initialization completed")
            self.reset_flag = False
            self.reset_flag_pub.publish(Bool(False))
            self.robot_state.header.stamp = rospy.Time.now()
            self.init_state_pub.publish(self.robot_state)
            # self.robot_state = None


    def run(self):
            rate = rospy.Rate(100) #100 Hz
            while not rospy.is_shutdown():
                self.control_pub.publish(self.control)
                if self.reset_flag:
                    trans = (self.robot_state.pose.position.x, self.robot_state.pose.position.y, self.robot_state.pose.position.z)
                    rot = (self.robot_state.pose.orientation.x, self.robot_state.pose.orientation.y, self.robot_state.pose.orientation.z,  self.robot_state.pose.orientation.w)
                    self.tf_broadcaster.sendTransform(trans, rot, rospy.Time.now(), 'robot', 'world')
                rate.sleep()

if __name__ == '__main__':
    tele = XboxTeleopSim()
    tele.run()

