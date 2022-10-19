#!/usr/bin/env python

import rospy
import numpy
from std_msgs.msg import String
from merlion_hardware.msg import Motor


m1 = m2 =m3=m4=m5=0
def callback(data):
    
    rospy.loginfo(data)
    
    data.m1 +=1
    data.m2 = data.m2+1
    data.m3 = data.m3+1
    data.m4 = data.m4+1
    data.m5 = data.m5+1
    
    pub.publish(data)
    
def callbackTest(data):
    rospy.loginfo(data)
    
pub = rospy.Publisher('con', Motor)
sub = rospy.Subscriber('feed', Motor, callback)
pubTest = rospy.Subscriber('chatter' , String , callbackTest)
    

def talker():
    
   
    rospy.init_node('con', anonymous=True)
    rate = rospy.Rate(10)
    msg=setMotorVals(m1,m2,m3,m4,m5,Motor())
    rospy.loginfo(msg)
    
    rate.sleep()
    
    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()
        
        

def setMotorVals(_m1,_m2,_m3,_m4,_m5,msg):
    msg.m1 = _m1
    msg.m2 = _m2
    msg.m3 = _m3
    msg.m4 = _m4
    msg.m5 = _m5
    
    return msg


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass