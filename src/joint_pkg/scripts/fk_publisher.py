#!/usr/bin/env python3
# coding: utf-8

import rospy
import math
import os
import numpy as np
import ikpy.chain
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

class DofbotFKNode:
    def __init__(self):
        # 1. 노드 초기화
        rospy.init_node('dofbot_fk_calculator', anonymous=False)

        # 2. URDF 파일 로드 (스크립트와 같은 폴더에 있다고 가정)f
        script_dir = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(script_dir, "dofbot.urdf")
        
        try:
            # active_links_mask=None이면 모든 조인트 정보를 자동으로 파싱합니다.
            self.my_chain = ikpy.chain.Chain.from_urdf_file(urdf_path)
            rospy.loginfo(f"URDF Loaded successfully from: {urdf_path}")
            rospy.loginfo(f"Chain links: {len(self.my_chain.links)}")
        except Exception as e:
            rospy.logerr(f"Failed to load URDF: {e}")
            exit(1)

        # 3. 퍼블리셔 설정 (End Effector의 Pose)
        self.pose_pub = rospy.Publisher('/dofbot/end_effector_pose', PoseStamped, queue_size=10)

        # 4. 서브스크라이버 설정 (현재 각도 수신)
        rospy.Subscriber('/dofbot/current_angles', Float32MultiArray, self.angle_callback)
        
        rospy.loginfo("DOFbot FK Node Started. Waiting for joint angles...")

    def angle_callback(self, data):
        try:
            # 수신된 각도 리스트 (Degree 단위: 0~180 or 0~270)
            joints_deg = data.data
            
            # 데이터 유효성 검사
            if len(joints_deg) < 5:
                return
            # --- [수정 포인트] ---
            # 6번 조인트(그리퍼) 데이터가 들어오더라도, 
            # 앞에서부터 5개(관절 1~5)만 짤라서 가져옵니다.
            joints_deg = joints_deg[:5]
            # --- [수정됨] ikpy 입력 준비 ---
            
            n_links = len(self.my_chain.links)
            
            # [핵심 수정] 
            # 서보 모터 값(0~180)에서 90도를 빼서 URDF 기준(-90~+90)으로 맞춤
            # 그 후 라디안으로 변환
            converted_joints = []
            converted_joints.append(0) # Base link (고정)

            for i in range(len(joints_deg)):
                # 모든 관절에 대해 -90도 오프셋 적용
                # (1~4번: 90-90=0, 5번: 90-90=0 / 270-90=180 등)
                urdf_angle = joints_deg[i] - 90.0
                converted_joints.append(math.radians(urdf_angle))
            
            target_joints = converted_joints

            # URDF 링크 개수와 맞추기 (예외 처리)
            if len(target_joints) > n_links:
                target_joints = target_joints[:n_links]
            elif len(target_joints) < n_links:
                target_joints += [0] * (n_links - len(target_joints))

            # --- 순기구학 계산 ---
            fk_matrix = self.my_chain.forward_kinematics(target_joints)


            # --- 메시지 생성 및 변환 ---
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "base_link" # 기준 좌표계

            # 1. 위치 (Translation) 추출: 행렬의 0,1,2행 3열
            pose_msg.pose.position.x = fk_matrix[0, 3]
            pose_msg.pose.position.y = fk_matrix[1, 3]
            pose_msg.pose.position.z = fk_matrix[2, 3]

            # 2. 회전 (Rotation) 추출: 행렬의 3x3 회전 부분 -> 쿼터니언 변환
            rotation_matrix = fk_matrix[:3, :3]
            try:
                r = R.from_matrix(rotation_matrix)
            except AttributeError:
                r = R.from_dcm(rotation_matrix)
            quat = r.as_quat() # [x, y, z, w] 순서

            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            # --- 발행 ---
            self.pose_pub.publish(pose_msg)

        except Exception as e:
            rospy.logerr(f"FK Calculation Error: {e}")

if __name__ == '__main__':
    try:
        node = DofbotFKNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass