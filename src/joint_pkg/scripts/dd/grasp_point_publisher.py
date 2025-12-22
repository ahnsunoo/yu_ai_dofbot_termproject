#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

import jetson.inference
import jetson.utils


def bbox_center_px(det):
    cx = int(round((det.Left + det.Right) * 0.5))
    cy = int(round((det.Top + det.Bottom) * 0.5))
    return cx, cy


def resolve_path(p, base_dir):
    """상대경로면 base_dir 기준으로 절대경로로 변환"""
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def main():
    rospy.init_node("grasp_pixel_publisher", anonymous=False)

    # ⭐ 중요: rosrun의 작업 디렉토리는 예측 불가하므로, 스크립트 위치 기준으로 상대경로를 해석
    script_dir = os.path.dirname(os.path.abspath(__file__))

    video       = rospy.get_param("~video", "/dev/video0")
    model       = rospy.get_param("~model", "models/final/ssd-mobilenet.onnx")
    labels      = rospy.get_param("~labels", "models/final/labels.txt")
    input_blob  = rospy.get_param("~input_blob", "input_0")
    output_cvg  = rospy.get_param("~output_cvg", "scores")
    output_bbox = rospy.get_param("~output_bbox", "boxes")
    threshold   = float(rospy.get_param("~threshold", 0.5))
    class_name  = rospy.get_param("~class_name", None)

    pub_topic_pixel = rospy.get_param("~pub_topic_pixel", "/grasp_pixel")
    pub_topic_class = rospy.get_param("~pub_topic_class", "/grasp_class")

    overlay     = rospy.get_param("~overlay", "box,labels,conf")
    publish_hz  = float(rospy.get_param("~publish_hz", 10.0))

    # ✅ 경로 해석(상대경로 -> 스크립트 위치 기준 절대경로)
    model_path  = resolve_path(model, script_dir)
    labels_path = resolve_path(labels, script_dir)

    rospy.loginfo("Resolved model : %s", model_path)
    rospy.loginfo("Resolved labels: %s", labels_path)

    # ✅ 파일 존재 체크 (여기서 걸리면 100% 경로 문제)
    if not os.path.isfile(model_path):
        rospy.logfatal("Model file not found: %s", model_path)
        return
    if not os.path.isfile(labels_path):
        rospy.logfatal("Labels file not found: %s", labels_path)
        return

    pub_pixel = rospy.Publisher(pub_topic_pixel, PointStamped, queue_size=10)
    pub_class = rospy.Publisher(pub_topic_class, String, queue_size=10)

    # detectNet 로드 (여기서 실패하면 대부분 경로/블롭이름/권한 문제)
    try:
        net = jetson.inference.detectNet(argv=[
            "--model={}".format(model_path),
            "--labels={}".format(labels_path),
            "--input-blob={}".format(input_blob),
            "--output-cvg={}".format(output_cvg),
            "--output-bbox={}".format(output_bbox),
            "--threshold={}".format(threshold),
        ])
    except Exception as e:
        rospy.logfatal("detectNet failed to load network: %s", str(e))
        rospy.logfatal("Check model/labels path and input/output blob names.")
        return

    cam = jetson.utils.videoSource(video)
    rate = rospy.Rate(publish_hz)

    rospy.loginfo("grasp_pixel_publisher started. pixel=%s class=%s", pub_topic_pixel, pub_topic_class)

    while not rospy.is_shutdown():
        try:
            img = cam.Capture()
            detections = net.Detect(img, overlay=overlay)

            if not detections:
                rospy.loginfo_throttle(2.0, "No detections")
                rate.sleep()
                continue

            if class_name:
                detections = [d for d in detections if net.GetClassDesc(d.ClassID) == class_name]
                if not detections:
                    rospy.loginfo_throttle(2.0, "No detections for class '%s'", class_name)
                    rate.sleep()
                    continue

            best = max(detections, key=lambda d: d.Confidence)
            cx, cy = bbox_center_px(best)
            conf = float(best.Confidence)
            cls  = net.GetClassDesc(best.ClassID)

            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "camera"
            msg.point.x = float(cx)
            msg.point.y = float(cy)
            msg.point.z = conf
            pub_pixel.publish(msg)

            cls_msg = String(data=cls)
            pub_class.publish(cls_msg)

            rospy.loginfo_throttle(1.0, "Best: %s conf=%.3f pixel=(%d,%d)", cls, conf, cx, cy)
            rate.sleep()

        except rospy.ROSInterruptException:
            break
        except Exception as e:
            rospy.logerr_throttle(2.0, "Runtime error: %s", str(e))
            rate.sleep()

    rospy.loginfo("grasp_pixel_publisher stopped.")


if __name__ == "__main__":
    main()
