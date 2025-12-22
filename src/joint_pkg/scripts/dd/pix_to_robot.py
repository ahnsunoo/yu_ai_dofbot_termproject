#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped


def pix_to_robot(u, v, H):
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H.dot(p)
    if abs(q[2]) < 1e-12:
        raise ValueError("Homography normalization failed (q[2] too small)")
    return float(q[0] / q[2]), float(q[1] / q[2])


class PixToRobotNode:
    def __init__(self):
        self.sub_topic = rospy.get_param("~sub_topic", "/grasp_pixel")
        self.pub_topic = rospy.get_param("~pub_topic", "/grasp_xy")
        self.H_path    = rospy.get_param("~H_path", "/home/ubuntu/homography_H_16points.npy")

        self.robot_frame = rospy.get_param("~robot_frame", "robot_base")

        self.H = np.load(self.H_path)
        if self.H.shape != (3, 3):
            raise ValueError("H must be 3x3, got {}".format(self.H.shape))

        self.pub = rospy.Publisher(self.pub_topic, PointStamped, queue_size=10)
        self.sub = rospy.Subscriber(self.sub_topic, PointStamped, self.cb, queue_size=10)

        rospy.loginfo("pix_to_robot_node started. H=%s", self.H_path)
        rospy.loginfo("sub: %s  pub: %s", self.sub_topic, self.pub_topic)

    def cb(self, msg):
        u = float(msg.point.x)
        v = float(msg.point.y)
        conf = float(msg.point.z)

        try:
            X, Y = pix_to_robot(u, v, self.H)

            out = PointStamped()
            out.header.stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
            out.header.frame_id = self.robot_frame
            out.point.x = X
            out.point.y = Y
            out.point.z = conf  # confidence 그대로 전달
            self.pub.publish(out)

            rospy.loginfo_throttle(1.0, "pix=(%.1f,%.1f)->robot=(%.3f,%.3f) conf=%.3f", u, v, X, Y, conf)

        except Exception as e:
            rospy.logerr_throttle(2.0, "pix_to_robot transform error: %s", str(e))


def main():
    rospy.init_node("pix_to_robot_node", anonymous=False)
    try:
        PixToRobotNode()
        rospy.spin()
    except Exception as e:
        rospy.logfatal("Failed to start pix_to_robot_node: %s", str(e))


if __name__ == "__main__":
    main()
