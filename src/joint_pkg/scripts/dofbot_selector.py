#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32MultiArray, Int32, String

class SelectorNode:
    def __init__(self):
        rospy.init_node("dofbot_selector")

        self.last_dets = []          # [{id,u,v,conf}, ...]
        self.last_dets_time = rospy.Time(0)

        self.sub_dets = rospy.Subscriber("/dofbot/detections", Float32MultiArray, self.cb_dets, queue_size=1)
        self.sub_sel  = rospy.Subscriber("/dofbot/select_class", Int32, self.cb_select, queue_size=1)

        self.pub_selected = rospy.Publisher("/dofbot/selected_object", Float32MultiArray, queue_size=1)

        # ✅ 추가: GUI에 성공/실패를 알려주는 피드백 토픽
        self.pub_feedback = rospy.Publisher("/dofbot/select_feedback", String, queue_size=1)

        rospy.loginfo("[Selector] ready.")

    def cb_dets(self, msg):
        arr = msg.data
        dets = []
        if len(arr) < 1:
            return
        N = int(arr[0])
        if len(arr) < 1 + 4 * N:
            return

        idx = 1
        for _ in range(N):
            cid  = int(arr[idx]);   idx += 1
            u    = float(arr[idx]); idx += 1
            v    = float(arr[idx]); idx += 1
            conf = float(arr[idx]); idx += 1
            dets.append({"id": cid, "u": u, "v": v, "conf": conf})

        self.last_dets = dets
        self.last_dets_time = rospy.Time.now()

    def cb_select(self, msg):
        class_id = int(msg.data)

        # detections가 너무 오래된 상태면 무시(예: 0.7초 이상)
        age = (rospy.Time.now() - self.last_dets_time).to_sec()
        if age > 0.7:
            rospy.logwarn(f"[Selector] detections too old ({age:.2f}s).")
            self.pub_feedback.publish(String(data=f"STALE {class_id}"))
            return

        candidates = [d for d in self.last_dets if d["id"] == class_id]
        if not candidates:
            rospy.logwarn(f"[Selector] no object of class {class_id} in current frame.")
            self.pub_feedback.publish(String(data=f"NOT_FOUND {class_id}"))
            return

        best = max(candidates, key=lambda d: d["conf"])

        out = Float32MultiArray()
        out.data = [float(class_id), float(best["u"]), float(best["v"])]
        self.pub_selected.publish(out)

        ok_msg = f"OK {class_id} conf={best['conf']:.2f} u={best['u']:.1f} v={best['v']:.1f}"
        self.pub_feedback.publish(String(data=ok_msg))

        rospy.loginfo(f"[Selector] {ok_msg}")

if __name__ == "__main__":
    SelectorNode()
    rospy.spin()
