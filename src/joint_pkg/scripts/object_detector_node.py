#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import os
import pickle
import numpy as np
import cv2  # 좌표 계산용으로만 사용 (이미지 처리 X)
import jetson.inference
import jetson.utils

from std_msgs.msg import Float32MultiArray

class ObjectDetectorNode:
    def __init__(self):
        rospy.init_node('ssd_mobilenet_tracker', anonymous=True)

        # -----------------------------------------------------------
        # 1. 캘리브레이션 데이터 로드
        # -----------------------------------------------------------
        script_dir = os.path.dirname(os.path.realpath(__file__))
        calib_path = os.path.join(script_dir, "camera_calibration.pkl")
        
        self.use_calibration = False
        self.mtx = None
        self.dist = None
        self.newcameramtx = None

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
        self.pub = rospy.Publisher('/dofbot/object_info', Float32MultiArray, queue_size=10)
        
        rospy.loginfo("Object Detector Node (Top-Center Publish) Started.")

    def run(self):
        while not rospy.is_shutdown():
            # 1. 캡처
            img = self.input.Capture()

            if img is None:
                continue

            # 2. 탐지
            detections = self.net.Detect(img)

            # 3. 로직 처리
            best_detection = self.get_best_detection(detections)

            if best_detection:
                self.publish_coordinates(best_detection)
            
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
        """
        단일 좌표 (u, v)에 대해 왜곡 보정을 수행하는 함수
        """
        if not self.use_calibration:
            return u, v

        src_points = np.array([[[u, v]]], dtype=np.float32)
        dst_points = cv2.undistortPoints(src_points, self.mtx, self.dist, P=self.newcameramtx)
        
        return dst_points[0][0][0], dst_points[0][0][1]

    def publish_coordinates(self, det):
        # -----------------------------------------------------------
        # [수정됨] 좌표 선정 로직 변경
        # 기존: 화면 좌/우에 따라 왼쪽/오른쪽 상단 꼭지점 선택
        # 변경: 위치 상관없이 '물체 중심의 가장 높은 곳' (Center X, Top Y) 선택
        # -----------------------------------------------------------
        
        # det.Center는 (x, y) 튜플입니다. det.Center[0]은 중심의 X좌표입니다.
        target_x = det.Center[0]
        
        # det.Top은 Bounding Box의 가장 상단 Y좌표입니다.
        target_y = det.Top
        
        position_msg = "Using Top-Center Vertex"

        # 2. 좌표 보정 수행
        real_x, real_y = self.undistort_point(target_x, target_y)

        # 3. 보정된 좌표 Publish
        msg = Float32MultiArray()
        msg.data = [float(det.ClassID), real_x, real_y]
        self.pub.publish(msg)
        
        rospy.loginfo(f"[{position_msg}] Raw: ({target_x:.1f}, {target_y:.1f}) -> Undistorted: ({real_x:.1f}, {real_y:.1f})")
        
        
if __name__ == '__main__':
    try:
        node = ObjectDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
