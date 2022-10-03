#!/usr/bin/env python3
import rospy
import smach
import time
from std_msgs.msg import Float64
from std_msgs.msg import String
from vision_msgs.msg import BoundingBox2D


class Input:
    def __init__(self):
        self.current_yaw = 0
        self.initial_yaw = 0
        self.gate_pos = BoundingBox2D()
        self.flare_pos = BoundingBox2D()
        self.flare_color = "Red"


class Output:
    def __init__(self, on_change_callback):
        self._speed = 0
        self._target_yaw = 0
        self._target_depth = 0
        self._on_change_callback = on_change_callback

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, new_speed):
        self._speed = new_speed
        self._on_change_callback()

    @property
    def target_yaw(self):
        return self._target_yaw

    @target_yaw.setter
    def target_yaw(self, new_target_yaw):
        self._target_yaw = new_target_yaw
        self._on_change_callback()

    @property
    def target_depth(self):
        return self._target_depth

    @target_depth.setter
    def target_depth(self, new_target_depth):
        self._target_depth = new_target_depth
        self._on_change_callback()


class BaseStrategy:
    def __init__(self, outcomes=[], input_keys=[], output_keys=[]):
        rospy.init_node('strategy')

        self.sm = smach.StateMachine(outcomes, input_keys, output_keys)

        try:
            import smach_ros
            sis = smach_ros.IntrospectionServer('server_name', self.sm, '/sm')
            sis.start()
        except Exception:
            pass

        self.input = Input()
        self.output = Output(self._on_output_update)

        rospy.Publisher('/merlion_pid/roll/setpoint', Float64, queue_size=10).publish(Float64(0))
        rospy.Publisher('/merlion_pid/pitch/setpoint', Float64, queue_size=10).publish(Float64(0))
        self._yaw_pub = rospy.Publisher('/merlion_pid/yaw/setpoint', Float64, queue_size=10)
        self._depth_pub = rospy.Publisher('/merlion_pid/depth/setpoint', Float64, queue_size=10)
        self._speed_pub = rospy.Publisher('/merlion_pid/speed/setpoint', Float64, queue_size=10)

        self._yaw_first_update = True
        rospy.Subscriber('/merlion_state/yaw', Float64, self._on_yaw_update)
        rospy.Subscriber('/merlion_cv/gate/pos', BoundingBox2D, self._on_gate_pos_update)
        rospy.Subscriber('/merlion_cv/flare/pos', BoundingBox2D, self._on_flare_pos_update)
        rospy.Subscriber('/merlion_cv/flare/color', String,  self._on_flare_color_update)

        self._time_last_publish = 0

    def _on_yaw_update(self, msg):
        if self._yaw_first_update:
            self._yaw_first_update = False
            self.input.initial_yaw = msg.data
        self.input.current_yaw = msg.data

    def _on_gate_pos_update(self, msg):
        self.input.gate_pos = msg

    def _on_flare_pos_update(self, msg):
        self.input.flare_pos = msg

    def _on_flare_color_update(self, msg):
        self.input.flare_color = msg

    def _on_output_update(self):
        if time.time() - self._time_last_publish > 0.01:
            self._time_last_publish = time.time()
            self._yaw_pub.publish(Float64(self.output.target_yaw))
            self._depth_pub.publish(Float64(self.output.target_depth))
            self._speed_pub.publish(Float64(self.output.speed))

    def execute(self):
        return self.sm.execute()
