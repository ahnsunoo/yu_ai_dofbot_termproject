#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from Arm_Lib import Arm_Device # 하드웨어 제어 라이브러리
from std_msgs.msg import Float32MultiArray, Int32

class DofbotManipulator:
    def __init__(self):
        rospy.init_node('dofbot_manipulator')

        # 1. 하드웨어 연결
        self.arm = Arm_Device()
        self.joints_home = [90, 111, 4 , 7, 89, 30] # 초기 위치 (사용자 환경에 맞게 수정)

        # 2. 상태 관리 변수 (핵심!)
        self.is_busy = False       # 현재 로봇이 움직이는 중인가?
        self.target_joints = None  # 목표 관절 각도 저장소
        self.target_class = -1     # 목표 클래스 ID 저장소

        # 3. Subscriber
        self.sub_joints = rospy.Subscriber('/dofbot/target_joints', Float32MultiArray, self.joints_callback)
        self.sub_class = rospy.Subscriber('/dofbot/target_class', Int32, self.class_callback)

        # 4. 초기화 동작
        self.move_to_home()
        rospy.loginfo("Manipulator Ready! Waiting for targets...")

    # ---------------------------------------------------
    # 콜백 함수: 데이터 수신 담당
    # ---------------------------------------------------
    def joints_callback(self, msg):
        # 만약 로봇이 작업 중이라면 새로운 데이터 무시 (또는 최신값 덮어쓰기만 하고 트리거 X)
        if self.is_busy:
            return 
        
        self.target_joints = msg.data
        # rospy.loginfo("Target Joints Received")

    def class_callback(self, msg):
        if self.is_busy:
            return
        
        self.target_class = msg.data
        # rospy.loginfo(f"Target Class Received: {self.target_class}")

    # ---------------------------------------------------
    # 로봇 팔 제어 함수들 (하드웨어 제어)
    # ---------------------------------------------------
    def gripper_control(self, state):
        """
        state: 'open' or 'close'
        그리퍼는 보통 6번 서보모터입니다. (각도는 하드웨어에 맞춰 조정 필요)
        """
        id = 6
        if state == 'open':
            angle = 30  # 열린 각도
        else:
            angle = 175 # 물체를 꽉 잡는 각도
            
        self.arm.Arm_serial_servo_write(id, angle, 500)
        time.sleep(1) # 동작 완료 대기

    def move_robot(self, joints, duration_ms=1000):
        """ 6축 관절 이동 함수 """
        # joints 리스트: [j1, j2, j3, j4, j5, gripper]
        # Arm_Lib의 write6 함수는 인자가 (j1, j2, j3, j4, j5, j6, time)
        
        # 안전장치: 관절 범위 체크 등을 추가할 수 있음
        self.arm.Arm_serial_servo_write6(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], duration_ms)
        
        # [중요] 하드웨어가 움직이는 시간 동안 코드도 대기해야 함
        time.sleep(duration_ms / 1000.0)
        time.sleep(0.5) # 여유 시간

    def move_to_home(self):
        rospy.loginfo("Moving to Home...")
        self.move_robot(self.joints_home, 1500)
        self.gripper_control('open')

    def move_to_drop_zone(self, class_id):
        """ Class ID에 따라 분류할 위치로 이동 """
        rospy.loginfo(f"Moving to drop zone for Class {class_id}")
        
        # 예시 좌표 (사용자가 직접 티칭해서 값을 채워야 함)
        if class_id == 1:   # AA 건전지 
            drop_joints = [37.0, 21.0, 79.0, 32.0, 89.0, 175.0] 
        elif class_id == 2: # 종이컵 
            drop_joints = [139.0, 23.0, 78.0, 32.0, 89.0, 175.0]
        elif class_id ==3:   #플라스틱 컵           
            drop_joints = [139.0, 23.0, 78.0, 32.0, 89.0, 175.0]
        elif class_id == 4: # 비타500
            drop_joints = [139.0, 23.0, 78.0, 32.0, 89.0, 175.0]
        elif class_id == 5: #  나무블럭
            drop_joints = [37.0, 21.0, 79.0, 32.0, 89.0, 175.0]
        
        drop_joints[5] = 175

        # 분류 위치로 이동
        self.move_robot(drop_joints, 2000)

    # ---------------------------------------------------
    # 메인 동작 로직 (시나리오 실행)
    # ---------------------------------------------------
    def run_pipeline(self):
        rate = rospy.Rate(10) # 10Hz
        
        while not rospy.is_shutdown():
            # 조건: 바쁘지 않고 + 관절 데이터가 있고 + 클래스 데이터가 있을 때
            if not self.is_busy and self.target_joints is not None and self.target_class != -1:
                
                # 1. 깃발 올리기 (작업 시작, 방해 금지)
                self.is_busy = True
                rospy.loginfo(">>> Start Pick and Place Sequence")

                try:
                    # 2. 물체 위치로 이동 (IK 결과값)
                    # 주의: IK 결과의 마지막 값(6번)이 그리퍼 각도인지 확인. 
                    # 보통 IK는 1~5번만 주고 6번은 따로 처리하거나 포함해서 줌.
                    # 여기서는 받은 그대로 이동한다고 가정.
                    rospy.loginfo("1. Approaching Object...")
                    
                    # 수신 받은 데이터는 tuple일 수 있으므로 list로 변환
                    target_pose = list(self.target_joints)
                    
                    # 그리퍼 열고 접근 (안전을 위해)
                    target_pose[5] = 30 # Open Angle
                    self.move_robot(target_pose, 2000)

                    # 3. 그리퍼 닫기 (Grasping)
                    rospy.loginfo("2. Grasping...")
                    self.gripper_control('close')
                    
                    # 4. 살짝 들어올리기 (옵션)
                    # 바닥에 긁히지 않게 현재 자세에서 2번 관절 등을 조금 움직여 들어올림
                    rospy.loginfo("3. Lifting...")
                    lift_pose = list(target_pose)
                    lift_pose[1] += 30 # 예: 2번 관절을 살짝 듦
                    lift_pose[5] = 175
                    self.move_robot(lift_pose, 1000)

                    # 5. 분류 위치로 이동 (Class ID 기반)
                    rospy.loginfo(f"4. Moving to Drop Zone (Class {self.target_class})...")
                    self.move_to_drop_zone(self.target_class)

                    # 6. 그리퍼 열기 (Placing)
                    rospy.loginfo("5. Releasing Object...")
                    self.gripper_control('open')

                    # 7. 초기 위치 복귀
                    rospy.loginfo("6. Resetting to Home...")
                    self.move_to_home()

                    # [핵심 수정] 8. 데이터 초기화 (Reset)
                    # 변수를 비워줍니다.
                    self.target_joints = None
                    self.target_class = -1
                    
                    rospy.loginfo("<<< Sequence Complete. Cooldown for 2 seconds...")

                    # [추가] 9. 쿨다운 (Cooldown)
                    # 로봇이 홈으로 돌아온 뒤 2초 동안 멍하니 있습니다.
                    # 이 시간 동안 is_busy는 여전히 True이므로, 들어오는 모든 토픽을 무시합니다.
                    time.sleep(4) 

                except Exception as e:
                    rospy.logerr(f"Error during sequence: {e}")
                    self.move_to_home()
                    # 에러가 났을 때도 데이터를 비워야 재시도 시 꼬이지 않습니다.
                    self.target_joints = None
                    self.target_class = -1

                # 10. 깃발 내리기 (이제부터 새로운 명령 수신 가능)
                self.is_busy = False
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = DofbotManipulator()
        controller.run_pipeline()
    except rospy.ROSInterruptException:
        pass
