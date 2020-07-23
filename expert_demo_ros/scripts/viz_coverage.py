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
    if fill:
        cm.type = Marker.CYLINDER
        cm.pose.position.x = circle.center[0]
        cm.pose.position.y = circle.center[1]
        cm.pose.position.z = 0.0
        cm.pose.orientation.x = 0.0
        cm.pose.orientation.y = 0.0
        cm.pose.orientation.z = 0.0
        cm.pose.orientation.w = 1.0
        cm.scale.x = circle.radius
        cm.scale.y = circle.radius
        cm.scale.z = 0.5
        cm.color.a = alpha # Don't forget to set the alpha!
        cm.color.r = rgb[0]
        cm.color.g = rgb[1]
        cm.color.b = rgb[2]
    else:
        cm.type = Marker.LINE_STRIP
        x, y = [0,1]
        center = [circle.center[x], circle.center[y]]
        th = np.arange(-np.pi, np.pi+0.1, 0.05)
        xs = center[x] + circle.radius * np.cos(th)
        ys = center[y] + circle.radius * np.sin(th)
        cm.points = [Point(x,y, 0.0) for (x,y) in zip(xs, ys)]
        cm.scale.x = 0.01
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


class CoverageVizualization(object):
    def __init__(self, cov_env):
        rospy.init_node("coverage_environment_visualization", anonymous=True)
        self.pub = rospy.Publisher("/coverage/visualization", MarkerArray, queue_size=10)
        self.marker_array = MarkerArray()

        fill = True
        for (i, obs) in enumerate(cov_env.obs):
            if isinstance(obs, Circle):
                self.marker_array.markers.append(circle_marker(obs, fill, "obstacle", i, [1,0,0]))
            elif isinstance(obs, Box):
                self.marker_array.markers.append(box_marker(obs, fill, "obstacle", i, [1,0,0]))
            else:
                raise TypeError("Obstacle type unknown for marker construction")

        fill = False
        for (i, covers) in enumerate(cov_env.covers):
            if isinstance(covers, Circle):
                self.marker_array.markers.append(circle_marker(covers, fill, "coverage", i, [0,0,1]))
            elif isinstance(covers, Box):
                self.marker_array.markers.append(box_marker(covers, fill, "coverage", i, [0,0,1]))
            else:
                raise TypeError("Coverage type unknown for marker construction")

        fill = True
        alpha = 0.5
        rgb = [135.0/255, 206.0/255, 250.0/255]
        if isinstance(cov_env.initial, Circle):
            self.marker_array.markers.append(circle_marker(cov_env.initial, fill, "initial", i, rgb, alpha=alpha))
        elif isinstance(cov_env.initial, Box):
            self.marker_array.markers.append(box_marker(cov_env.initial, fill, "initial", i, rgb, alpha=alpha))
        else:
            raise TypeError("Initial type unknown for marker construction")

        rgb = [240.0/255, 128.0/255, 128.0/255]
        if isinstance(cov_env.final, Circle):
            self.marker_array.markers.append(circle_marker(cov_env.final, fill, "final", i, rgb, alpha=alpha))
        elif isinstance(cov_env.final, Box):
            self.marker_array.markers.append(box_marker(cov_env.final, fill, "final", i, rgb, alpha=alpha))
        else:
            raise TypeError("Initial type unknown for marker construction")



    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            for m in self.marker_array.markers:
                m.header.stamp = rospy.Time.now()
            self.pub.publish(self.marker_array)
            rate.sleep()


if __name__ == "__main__":

    params = {  "covers": [Box([0., 6.],[4, 8.]), Box([6., 2.],[10.0, 4.0])],
                "obstacles": [Circle([7., 7.], 2.0)],
                "initial": Box([-1., -1.],[1., 1.]),
                "final": Box([9.0, 9.0],[11.0, 11.0])
           }

    cov_env = CoverageEnv(params)


    cv = CoverageVizualization(cov_env)
    cv.run()