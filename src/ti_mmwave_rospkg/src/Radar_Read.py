#!/usr/bin/env python
import rospy

from ti_mmwave_rospkg.msg import RadarScan

def callback(data):
    print data

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("RadarScan", RadarScan, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
