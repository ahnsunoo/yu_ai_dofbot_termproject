#!/usr/bin/env python3
# coding: utf-8

import rospy
import math
import os
import ikpy.chain
import numpy as np
from geometry_msgs.msg import Pose
from joint_pkg.srv import GetIK, GetIKResponse

class IKServerNode:
    def __init__(self):
        rospy.init_node('dofbot_ik_server')
        
        # URDF 로드
        script_dir = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(script_dir, "dofbot.urdf")
        
        try:
            self.chain = ikpy.chain.Chain.from_urdf_file(urdf_path)
            rospy.loginfo(f"IK Server Ready. Chain links: {len(self.chain.links)}")
        except Exception as e:
            rospy.logerr(f"URDF Load Failed: {e}")
            exit(1)

        self.server = rospy.Service('/dofbot/solve_ik', GetIK, self.handle_ik_request)

    def handle_ik_request(self, req):
        response = GetIKResponse()
        
        try:
            # 1. Target Position 추출
            target_pos = [req.target_pose.position.x,
                          req.target_pose.position.y,
                          req.target_pose.position.z]
            
            # [참고] Orientation(자세) 제어가 필요하다면 target_orientation 인자를 추가해야 함.
            # Dofbot 같은 5자유도 로봇은 위치만 주면 자세가 고정되지 않아 해가 여러 개일 수 있음.
            # ikpy는 기본적으로 최적의 해를 찾으려 노력함.
            
            # 2. IK 계산
            ik_solution = self.chain.inverse_kinematics(target_position=target_pos)
            
            # 3. 결과 처리 및 [핵심 수정] 좌표계 변환
            real_joints = []
            
            # ikpy 결과의 첫 번째(인덱스 0)는 Base Link(가상)이므로 건너뜀 (range(1, ...))
            for i in range(1, len(ik_solution)):
                # (1) Radian -> Degree 변환
                angle_deg = math.degrees(ik_solution[i])
                
                # (2) [수정] URDF(-90~90) -> 하드웨어(0~180) 변환: +90도 오프셋
                hw_angle = angle_deg + 90.0
                
                # (3) [수정] 안전장치 (Clamping): 0~180 범위를 벗어나지 않도록 제한
                # 단, 5번 조인트(손목 회전)가 0~270도라면 i값에 따라 분기 처리 가능
                # 여기서는 안전을 위해 기본 0~180으로 예시를 듬
                if i == 5: # 만약 마지막 관절(Wrist Roll)이라면 범위가 다를 수 있음 (확인 필요)
                     hw_angle = max(0.0, min(270.0, hw_angle))
                else:
                     hw_angle = max(0.0, min(180.0, hw_angle))
                
                real_joints.append(hw_angle)

            # 4. [수정] 그리퍼(6번) 데이터 추가
            # IK는 팔만 계산하므로, 그리퍼 값은 기본값(90)이나 현재 상태 유지를 위해 추가해야 함
            # 받는 쪽에서 6개의 데이터를 기대한다면 필수
            if len(real_joints) == 5:
                real_joints.append(90.0) # 그리퍼 열림/닫힘 중간값

            response.joint_angles = real_joints
            response.success = True
            
            rospy.loginfo(f"IK Solved (HW Angles): {np.round(real_joints, 2)}")
            
        except Exception as e:
            rospy.logerr(f"IK Calculation Error: {e}")
            response.success = False
            response.joint_angles = []
            
        return response

if __name__ == '__main__':
    node = IKServerNode()
    rospy.spin()