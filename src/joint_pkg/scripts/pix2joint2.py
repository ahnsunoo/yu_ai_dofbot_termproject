#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import numpy as np
from std_msgs.msg import Float32MultiArray, Int32
from geometry_msgs.msg import Pose
from joint_pkg.srv import GetIK, GetIKRequest

class Pix2JointNode:
    def __init__(self):
        # 노드 이름 변경: pix2joint
        rospy.init_node('pix2joint')

        # -----------------------------------------------------------
        # 1. Homography 행렬 로드
        # -----------------------------------------------------------
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "homography_H_16points.npy")

        try:
            self.H = np.load(file_path)
            rospy.loginfo(f"Homography loaded from: {file_path}")
        except Exception as e:
            rospy.logerr(f"H Matrix Load Failed: {e}")
            self.H = np.eye(3)

        # -----------------------------------------------------------
        # 2. ROS 통신 설정
        # -----------------------------------------------------------
        # (1) Subscriber: 객체 인식 정보 수신
        self.sub = rospy.Subscriber('/dofbot/object_info', Float32MultiArray, self.object_callback,queue_size=1)
        self.sub = rospy.Subscriber('/dofbot/selected_object', Float32MultiArray, self.object_callback, queue_size=1)

        # (2) Service Client: IK 서버 연결
        rospy.loginfo("Waiting for IK service...")
        rospy.wait_for_service('/dofbot/solve_ik')
        self.ik_client = rospy.ServiceProxy('/dofbot/solve_ik', GetIK)
        rospy.loginfo("IK Service Connected.")

        # (3) Publisher: 결과 발행
        # 관절 각도 리스트 (Gripper 포함)
        self.pub_joints = rospy.Publisher('/dofbot/target_joints', Float32MultiArray, queue_size=10)
        # 클래스 ID
        self.pub_class = rospy.Publisher('/dofbot/target_class', Int32, queue_size=10)

    def pix_to_robot(self, u, v):
        """ 픽셀 좌표 -> 로봇 좌표 변환 (Homography) """
        p = np.array([u, v, 1.0], dtype=np.float64)
        q = self.H @ p
        
        if q[2] == 0:
            return 0.0, 0.0
            
        real_x = float(q[0] / q[2])
        real_y = float(q[1] / q[2])
        return real_x, real_y

    def get_target_z_by_class(self, class_id):
        """ Class ID별 목표 Z 높이 설정 """
        target_z = 0.0
        
        if class_id == 1:   # AA건전지
            target_z = 0.05
        elif class_id == 2: # 미니 종이컵
            target_z = 0.05
        elif class_id == 3: # 미니 플라스틱컵
            target_z = 0.05
        elif class_id == 4: # 비타500
            target_z = 0.05
        elif class_id == 5: # 나무블럭
            target_z = 0.05
            
        return target_z

    def object_callback(self, msg):
        """ 토픽 수신 시 실행되는 콜백 """
        if len(msg.data) < 3:
            return

        # 1. 데이터 파싱
        class_id = int(msg.data[0])
        u_coord = msg.data[1]
        v_coord = msg.data[2]

        # 2. 좌표 변환
        target_x, target_y = self.pix_to_robot(u_coord, v_coord)
        target_z = self.get_target_z_by_class(class_id)

        # 3. IK 서비스 요청 및 발행
        self.request_ik_and_publish(class_id, target_x, target_y, target_z)

    def request_ik_and_publish(self, class_id, x, y, z):
        """ IK 서비스 호출 후 결과 발행 """
        req = GetIKRequest()
        
        # Request 메시지 채우기
        req.target_pose.position.x = x
        req.target_pose.position.y = y
        req.target_pose.position.z = z
        req.target_pose.orientation.w = 1.0 

        try:
            # 서비스 요청 전송 및 응답 대기
            response = self.ik_client(req)

            # 성공 시 토픽 발행
            if response.success:
                # 관절 각도 발행
                joints_msg = Float32MultiArray()
                joints_msg.data = response.joint_angles
                self.pub_joints.publish(joints_msg)

                # 클래스 ID 발행
                class_msg = Int32()
                class_msg.data = class_id
                self.pub_class.publish(class_msg)

                rospy.loginfo(f"Published Joints for ID {class_id}: {list(np.round(response.joint_angles, 2))}")
            else:
                rospy.logwarn(f"IK Failed for ID {class_id} at ({x:.3f}, {y:.3f}, {z:.3f})")

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    try:
        Pix2JointNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
