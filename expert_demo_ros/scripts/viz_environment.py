#!/usr/bin/env python
# import sys
# sys.path.append('../src')
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from utils.environment import *
import numpy as np

def circle_marker(circle, fill, ns, marker_id, rgb, alpha=1.0):
    cm = Marker()
    cm.header.frame_id = '/world'
    cm.id = marker_id
    cm.ns = ns
    cm.color.a = alpha # Don't forget to set the alpha!
    cm.color.r = rgb[0]
    cm.color.g = rgb[1]
    cm.color.b = rgb[2]

    if fill:
        cm.type = Marker.CYLINDER
        cm.pose.position.x = circle.center[0]
        cm.pose.position.y = circle.center[1]
        cm.pose.position.z = 0.0
        cm.pose.orientation.x = 0.0
        cm.pose.orientation.y = 0.0
        cm.pose.orientation.z = 0.0
        cm.pose.orientation.w = 1.0
        cm.scale.x = circle.radius * 2
        cm.scale.y = circle.radius * 2
        cm.scale.z = 0.5
    else:
        cm.type = Marker.LINE_STRIP
        center = [circle.center[0], circle.center[1]]
        th = np.arange(-np.pi, np.pi+0.1, 0.05)
        xs = center[0] + circle.radius * np.cos(th)
        ys = center[1] + circle.radius * np.sin(th)
        cm.points = [Point(x,y, 0.0) for (x,y) in zip(xs, ys)]
        cm.scale.x = 0.05
    return cm

def box_marker(box, fill, ns, marker_id, rgb, alpha=1.0):
    bm = Marker()
    bm.header.frame_id = '/world'
    bm.id = marker_id
    bm.ns = ns
    bm.color.a = alpha # Don't forget to set the alpha!
    bm.color.r = rgb[0]
    bm.color.g = rgb[1]
    bm.color.b = rgb[2]
    if fill:
        bm.type = Marker.CUBE
        bm.pose.position.x = (box.upper[0] + box.lower[0])/2
        bm.pose.position.y =(box.upper[1] + box.lower[1])/2
        bm.pose.position.z = 0.0
        bm.pose.orientation.x = 0.0
        bm.pose.orientation.y = 0.0
        bm.pose.orientation.z = 0.0
        bm.pose.orientation.w = 1.0
        bm.scale.x = box.upper[0] - box.lower[0]
        bm.scale.y = box.upper[1] - box.lower[1]
        bm.scale.z = 0.5
    else:
        bm.type = Marker.LINE_STRIP
        x, y = [0,1]
        lower = box.lower
        upper = box.upper
        x_corners = [lower[x], upper[x], upper[x], lower[x], lower[x]]
        y_corners = [lower[y], lower[y], upper[y], upper[y], lower[y]]
        bm.points = [Point(x,y, 0.0) for (x,y) in zip(x_corners, y_corners)]
        bm.scale.x = 0.01

    return bm


class EnvironmentVizualization(object):
    def __init__(self, env):
        rospy.init_node("environment_visualization", anonymous=True)
        self.pub = rospy.Publisher("/environment/visualization", MarkerArray, queue_size=10)
        self.init_pub = rospy.Publisher("/environment/visualization/initial_set", Point, queue_size=10)
        self.marker_array = MarkerArray()
        
        init_center = env.initial.center()
        self.initial_state_offset = Point(init_center[0], init_center[1], 0.0)





        fill = True
        for (i, obs) in enumerate(env.obs):
            if isinstance(obs, Circle):
                self.marker_array.markers.append(circle_marker(obs, fill, "obstacle", i, [1,0,0]))
            elif isinstance(obs, Box):
                self.marker_array.markers.append(box_marker(obs, fill, "obstacle", i, [1,0,0]))
            else:
                raise TypeError("Obstacle type unknown for marker construction")

        fill = False
        for (i, covers) in enumerate(env.covers):
            if isinstance(covers, Circle):
                self.marker_array.markers.append(circle_marker(covers, fill, "coverage", i, [0,0,1]))
            elif isinstance(covers, Box):
                self.marker_array.markers.append(box_marker(covers, fill, "coverage", i, [0,0,1]))
            else:
                raise TypeError("Coverage type unknown for marker construction")

        fill = True
        alpha = 0.5
        rgb = [135.0/255, 206.0/255, 250.0/255]
        if isinstance(env.initial, Circle):
            self.marker_array.markers.append(circle_marker(env.initial, fill, "initial", 0, rgb, alpha=alpha))
        elif isinstance(env.initial, Box):
            self.marker_array.markers.append(box_marker(env.initial, fill, "initial", 0, rgb, alpha=alpha))
        else:
            raise TypeError("Initial type unknown for marker construction")

        rgb = [240.0/255, 128.0/255, 128.0/255]
        if isinstance(env.final, Circle):
            self.marker_array.markers.append(circle_marker(env.final, fill, "final", 0, rgb, alpha=alpha))
        elif isinstance(env.final, Box):
            self.marker_array.markers.append(box_marker(env.final, fill, "final", 0, rgb, alpha=alpha))
        else:
            raise TypeError("Initial type unknown for marker construction")



    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            for m in self.marker_array.markers:
                m.header.stamp = rospy.Time.now()
            self.pub.publish(self.marker_array)
            self.init_pub.publish(self.initial_state_offset)
            rate.sleep()


if __name__ == "__main__":
    args = rospy.myargv()

    env = args[1]
    if env == "coverage":
        params = {  "covers": [Box([0., 6.],[4, 8.]), Box([6., 2.],[10.0, 4.0])],
                    "obstacles": [Circle([7., 7.], 2.0)],
                    "initial": Box([-1., -1.],[1., 1.]),
                    "final": Box([9.0, 9.0],[11.0, 11.0])
               }
    elif env == "stop_sign":
        params = {  "covers": [Circle([0., 0.], 8.0)],
                    "obstacles": [Circle([0., 0.], 5.0)],
                    "initial": Box([5., -15.],[8., 0.]),
                    "final": Box([-15., 5.],[0., 8.0])
               }
    elif env == "test":
        params = {  "covers": [],
                    "obstacles": [],
                    "initial": Box([-1., -1.],[1., 1.]),
                    "final": Box([4.0, 4.0],[6.0, 6.0])
               }    
    env = Environment(params)


    cv = EnvironmentVizualization(env)
    cv.run()