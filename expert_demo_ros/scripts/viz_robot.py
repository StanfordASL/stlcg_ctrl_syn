#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Pose, Quaternion
import numpy as np
from expert_demo_ros.msg import PoseTwistStamped, Float32MultiArrayStamped
import tf


# turn into ROSPARAMS
V_MIN = rospy.get_param("V_MIN", 0.0)
V_MAX = rospy.get_param("V_MAX", 0.2)
A_MAX = rospy.get_param("A_MAX", 0.1)
OM_MAX = rospy.get_param("OM_MAX", 0.2)
LF = rospy.get_param("LF", 1.105)
LR = rospy.get_param("LR", 1.738)
CAR_SIZE = LF + LR
DELTA_MAX = rospy.get_param("DELTA_MAX", 18*np.pi/180)
MODEL = rospy.get_param("MODEL", "kinematic bicycle")


def robot_marker(size):
    # rgb = [152./255, 251./255, 152./255]
    rgb = [220./255, 20./255, 60./255]

    rm = Marker()
    rm.ns = "robot_body"
    rm.header.frame_id = 'robot'
    rm.color.a = 1.0 # Don't forget to set the alpha!
    rm.color.r = rgb[0]
    rm.color.g = rgb[1]
    rm.color.b = rgb[2]
    rm.type = Marker.CUBE
    rm.pose.position.x = (LF - LR)/2
    rm.pose.position.y = 0.0
    rm.pose.position.z = 0.0
    rm.pose.orientation.x = 0.0
    rm.pose.orientation.y = 0.0
    rm.pose.orientation.z = 0.0
    rm.pose.orientation.w = 1.0
    rm.scale.x = size
    rm.scale.y = size/2
    rm.scale.z = size/2.

    return rm


def wheel_marker(size, front_back):
    rgb = [105./255, 105./255, 105./255]
    rm = Marker()
    if front_back == "front":
        rm.id = 1
        rm.pose.position.x = LF
    else:
        rm.id = 2
        rm.pose.position.x = -LR

    rm.ns = "robot_wheels"
    rm.header.frame_id = 'robot'
    rm.color.a = 1.0 # Don't forget to set the alpha!
    rm.color.r = rgb[0]
    rm.color.g = rgb[1]
    rm.color.b = rgb[2]
    rm.type = Marker.CUBE
    rm.pose.position.y = 0.0
    rm.pose.position.z = 0.0
    rm.pose.orientation.x = 0.0
    rm.pose.orientation.y = 0.0
    rm.pose.orientation.z = 0.0
    rm.pose.orientation.w = 1.0
    rm.scale.x = size/4.
    rm.scale.y = size/20
    rm.scale.z = size
    return rm

class RobotVizualization(object):
    def __init__(self, size):
        rospy.init_node("robot_visualization", anonymous=True)
        self.pub = rospy.Publisher("/robot/visualization", Marker, queue_size=10)
        self.pub_speed = rospy.Publisher("/robot/speed_visualization", Marker, queue_size=10)
        self.pub_pose = rospy.Publisher("/robot/pose", PoseStamped, queue_size=10)
        self.pub_ctrl = rospy.Publisher("/robot/ctrl_visualization", MarkerArray, queue_size=10)


        self.marker = robot_marker(CAR_SIZE)
        self.control = MarkerArray(markers=[wheel_marker(CAR_SIZE, "front"), wheel_marker(CAR_SIZE, "rear")])
        self.speed = robot_marker(CAR_SIZE * 1.05)
        self.speed.color.r = 152.0/255
        self.speed.color.g = 251.0/255
        self.speed.color.b = 152.0/255
        pose = Pose()
        pose.orientation.w = 1.0
        self.pose = PoseStamped(pose=pose)
        self.pose.header.frame_id = 'robot'
        rospy.Subscriber("/robot/state", PoseTwistStamped, self.state_callback)
        rospy.Subscriber("/robot/control", Float32MultiArrayStamped, self.control_callback)

    def state_callback(self, msg):
        gamma = msg.twist.linear.x / V_MAX
        self.speed.scale.x = CAR_SIZE * gamma
        self.speed.pose.position.x = -LR + CAR_SIZE * gamma / 2.0
        self.speed.header.stamp = msg.header.stamp
        self.pub_speed.publish(self.speed)

        # self.pose.header.stamp = msg.header.stamp
        # euler = [0.0, 0.0, msg.twist.linear.y]
        # self.pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(*euler))
        # self.pub_pose.publish(self.pose)

    def control_callback(self, msg):
        if MODEL == "kinematic bicycle":
            if len(msg.data.data) > 0:
                delta = msg.data.data[2] * DELTA_MAX
                euler = [0.0, 0.0, delta]
                self.control.markers[0].pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(*euler))
                # for m in self.control.markers:
                #     m.header.stamp = msg.header.stamp
                # self.pub_ctrl.publish(self.control)

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            time_now = rospy.Time.now()
            self.marker.header.stamp = time_now
            self.pub.publish(self.marker)
            for m in self.control.markers:
                m.header.stamp = time_now
            self.pub_ctrl.publish(self.control)
            rate.sleep()

if __name__ == "__main__":
    rv = RobotVizualization(CAR_SIZE)
    rv.run()