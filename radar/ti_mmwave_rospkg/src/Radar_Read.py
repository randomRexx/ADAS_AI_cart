#!/usr/bin/env python
import rospy


from ti_mmwave_rospkg.msg import RadarScan
from rospy.numpy_msg import numpy_msg

def callback(data):
    print "X: ", data.x
    print "Y: ", data.y
    print "Velocity: ", data.velocity
    print "Range: ", data.range
    print "\n"

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/ti_mmwave/radar_scan", numpy_msg(RadarScan), callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
