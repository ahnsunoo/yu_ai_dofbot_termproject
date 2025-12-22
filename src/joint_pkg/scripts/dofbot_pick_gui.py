#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
from std_msgs.msg import Int32, String

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import pyqtSignal

CLASS_NAME = {
    1: "AA건전지",
    2: "미니 종이컵",
    3: "미니 플라스틱컵",
    4: "비타500",
    5: "나무블럭",
}

class PickGUI(QWidget):
    feedback_signal = pyqtSignal(str)  # ✅ UI 안전 업데이트용

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DOFBOT Pick GUI (A)")

        self.pub = rospy.Publisher("/dofbot/select_class", Int32, queue_size=1)
        self.sub_fb = rospy.Subscriber("/dofbot/select_feedback", String, self.cb_feedback, queue_size=1)

        layout = QVBoxLayout()
        self.status = QLabel("버튼을 누르면 해당 클래스 1개를 집습니다.")
        layout.addWidget(self.status)

        for cid, name in CLASS_NAME.items():
            b = QPushButton(f"{name} 집기")
            b.clicked.connect(lambda _, c=cid: self.select_class(c))
            layout.addWidget(b)

        self.setLayout(layout)

        # ✅ signal -> label 연결
        self.feedback_signal.connect(self.status.setText)

    def select_class(self, class_id):
        msg = Int32()
        msg.data = class_id
        self.pub.publish(msg)

        name = CLASS_NAME.get(class_id, f"ID{class_id}")
        self.status.setText(f"선택 요청: {name} (class_id={class_id})")
        rospy.loginfo(f"[GUI] select_class={class_id}")

    def cb_feedback(self, msg):
        # msg.data 예: "NOT_FOUND 3" / "STALE 2" / "OK 1 conf=0.92 u=... v=..."
        parts = msg.data.split()
        if len(parts) >= 2:
            tag = parts[0]
            try:
                cid = int(parts[1])
            except:
                cid = None
            name = CLASS_NAME.get(cid, f"ID{cid}") if cid is not None else "알 수 없음"

            if tag == "NOT_FOUND":
                text = f"❌ 현재 화면에 '{name}' 없음"
            elif tag == "STALE":
                text = f"⚠️ 탐지 결과가 오래됨. '{name}' 다시 시도해줘"
            elif tag == "OK":
                # OK 뒤 내용을 그대로 표시
                rest = " ".join(parts[2:]) if len(parts) > 2 else ""
                text = f"✅ 선택 완료: {name} {rest}".strip()
            else:
                text = msg.data
        else:
            text = msg.data

        # ✅ UI는 signal로 업데이트
        self.feedback_signal.emit(text)

def main():
    rospy.init_node("dofbot_pick_gui", anonymous=False)
    app = QApplication(sys.argv)
    w = PickGUI()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
