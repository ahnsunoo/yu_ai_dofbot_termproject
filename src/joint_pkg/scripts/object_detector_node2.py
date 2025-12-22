#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import os
import pickle
import numpy as np
import cv2  # 좌표 보정용
import jetson.inference
import jetson.utils

from std_msgs.msg import Float32MultiArray


class ObjectDetectorNode:
    def __init__(self):
        rospy.init_node('ssd_mobilenet_tracker', anonymous=True)

        # -----------------------------------------------------------
        # 0. 동작 모드 파라미터
        # -----------------------------------------------------------
        # GUI 선택 파이프라인에서는 보통 best 1개 자동 pick을 끄는 게 안전함.
        # 필요하면 rosrun 시 _publish_best:=true 로 켤 수 있음.
        self.publish_best = rospy.get_param("~publish_best", False)

        # -----------------------------------------------------------
        # 1. 캘리브레이션 데이터 로드
        # -----------------------------------------------------------
        script_dir = os.path.dirname(os.path.realpath(__file__))
        calib_path = os.path.join(script_dir, "camera_calibration.pkl")

        self.use_calibration = False
        self.mtx = None
        self.dist = None
        self.newcameramtx = None
        self.roi = None

        if os.path.exists(calib_path):
            try:
                with open(calib_path, 'rb') as f:
                    calib_data = pickle.load(f)
                    self.mtx = calib_data['camera_matrix']
                    self.dist = calib_data['dist_coeffs']
                    self.use_calibration = True
                    rospy.loginfo(f"Calibration data loaded from {calib_path}")
            except Exception as e:
                rospy.logerr(f"Failed to load calibration file: {e}")
        else:
            rospy.logwarn(f"Calibration file not found at {calib_path}. Publishing RAW coordinates.")

        # -----------------------------------------------------------
        # 2. 모델 설정
        # -----------------------------------------------------------
        self.net = jetson.inference.detectNet(
            argv=[
                "--model=/home/ubuntu/jetson-inference/python/training/detection/ssd/models/final/ssd-mobilenet.onnx",
                "--labels=/home/ubuntu/jetson-inference/python/training/detection/ssd/models/final/labels.txt",
                "--input-blob=input_0",
                "--output-cvg=scores",
                "--output-bbox=boxes",
                "--threshold=0.5"
            ]
        )

        # -----------------------------------------------------------
        # 3. 카메라 설정 (Jetson Utils)
        # -----------------------------------------------------------
        self.input = jetson.utils.videoSource("/dev/video0", argv=sys.argv)
        self.display = jetson.utils.videoOutput("display://0", argv=sys.argv)

        # 카메라 해상도 확인
        self.width = self.input.GetWidth()
        self.height = self.input.GetHeight()

        # 캘리브레이션 매트릭스 최적화 (한 번만 계산)
        if self.use_calibration:
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (self.width, self.height), 1, (self.width, self.height)
            )

        # -----------------------------------------------------------
        # 4. ROS 설정
        # -----------------------------------------------------------
        # (1) 기존: best 1개만 내보내던 토픽
        self.pub_best = rospy.Publisher('/dofbot/object_info', Float32MultiArray, queue_size=10)

        # (2) 추가: 프레임 내 "전체 탐지 목록" 토픽 (GUI/Selector용)
        # 포맷: [N, class0, u0, v0, conf0, class1, u1, v1, conf1, ...]
        self.pub_dets = rospy.Publisher('/dofbot/detections', Float32MultiArray, queue_size=1)

        rospy.loginfo("Object Detector Node Started.")
        rospy.loginfo(f" - publish_best (object_info): {self.publish_best}")
        rospy.loginfo(" - publish detections list (/dofbot/detections): True")

    def run(self):
        while not rospy.is_shutdown():
            # 1. 캡처
            img = self.input.Capture()
            if img is None:
                continue

            # 2. 탐지
            detections = self.net.Detect(img)

            # ✅ (추가) 전체 탐지 목록 퍼블리시 (GUI/Selector가 이걸 보고 선택)
            self.publish_detections(detections)

            # 3. (옵션) best 1개 퍼블리시 (원래 방식)
            if self.publish_best:
                best_detection = self.get_best_detection(detections)
                if best_detection:
                    self.publish_best_object(best_detection)

            # 4. 렌더링
            self.display.Render(img)
            self.display.SetStatus("Object Detection | Network {:.0f} FPS".format(self.net.GetNetworkFPS()))

            if not self.input.IsStreaming() or not self.display.IsStreaming():
                break

    def get_best_detection(self, detections):
        if not detections:
            return None
        best_det = None
        max_conf = -1.0
        for det in detections:
            if det.Confidence > max_conf:
                max_conf = det.Confidence
                best_det = det
        return best_det

    def undistort_point(self, u, v):
        """단일 좌표 (u, v)에 대해 왜곡 보정을 수행"""
        if not self.use_calibration:
            return float(u), float(v)

        src_points = np.array([[[float(u), float(v)]]], dtype=np.float32)
        dst_points = cv2.undistortPoints(src_points, self.mtx, self.dist, P=self.newcameramtx)

        return float(dst_points[0][0][0]), float(dst_points[0][0][1])

    def pick_point_uv(self, det):
        """
        det에서 대표 픽셀 좌표 (u,v)를 뽑는 규칙.
        현재 규칙: Top-Center (Center X, Top Y)
        """
        u = float(det.Center[0])
        v = float(det.Top)
        return u, v

    def publish_best_object(self, det):
        """
        기존 토픽: /dofbot/object_info
        포맷: [class_id, u, v]  (u,v는 undistort 적용)
        """
        u_raw, v_raw = self.pick_point_uv(det)
        u, v = self.undistort_point(u_raw, v_raw)

        msg = Float32MultiArray()
        msg.data = [float(det.ClassID), float(u), float(v)]
        self.pub_best.publish(msg)

        rospy.loginfo(
            f"[Best] class={int(det.ClassID)} conf={det.Confidence:.2f} "
            f"Raw({u_raw:.1f},{v_raw:.1f}) -> Undist({u:.1f},{v:.1f})"
        )

    def publish_detections(self, detections):
        """
        추가 토픽: /dofbot/detections
        포맷: [N, class0, u0, v0, conf0, class1, u1, v1, conf1, ...]
        (u,v는 best와 동일하게 Top-Center + undistort 적용)
        """
        msg = Float32MultiArray()

        if not detections:
            msg.data = [0.0]
            self.pub_dets.publish(msg)
            return

        data = [float(len(detections))]
        for det in detections:
            u_raw, v_raw = self.pick_point_uv(det)
            u, v = self.undistort_point(u_raw, v_raw)
            conf = float(det.Confidence)

            data += [float(det.ClassID), float(u), float(v), conf]

        msg.data = data
        self.pub_dets.publish(msg)


if __name__ == '__main__':
    try:
        node = ObjectDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
