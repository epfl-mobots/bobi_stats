#!/usr/bin/env python
import rospy
from bobi_msgs.msg import PoseVec

import os
import socket
from datetime import datetime


class PoseStat:
    def __init__(self):
        self._fp_sub = rospy.Subscriber(
            'filtered_poses', PoseVec, self._filtered_poses_cb)
        self._up_sub = rospy.Subscriber(
            'individual_poses', PoseVec, self._unfiltered_poses_cb)
        self._ro_sub = rospy.Subscriber(
            'robot_poses', PoseVec, self._unfiltered_poses_cb)

        self._rate = rospy.get_param("top_camera/fps", 60)

        self._of = rospy.get_param('pose_stat/output_folder', '.')
        if '~' in self._of:
            home = os.path.expanduser('~')
            self._of = self._of.replace('~', home)
        hostname = socket.gethostname()
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y-%H_%M_%S")
        self._of = '{}/{}_{}'.format(self._of, hostname, date_time)
        os.makedirs(self._of)

        self._fp_file = open('{}/filtered_poses.txt'.format(self._of), 'w')
        self._up_file = open('{}/unfiltered_poses.txt'.format(self._of), 'w')
        self._ro_file = open('{}/robot_poses.txt'.format(self._of), 'w')

        self._prev_filtered_poses = None
        self._prev_unfiltered_poses = None
        self._prev_robot_poses = None
        self._time = [0.0] * 3

    def update(self):
        pass

    def _filtered_poses_cb(self, msg):
        data = msg.poses
        if self._prev_filtered_poses is not None and len(data):
            time = (data[0].header.stamp -
                    self._prev_unfiltered_poses[0].header.stamp).to_sec()
            if time >= 0:
                self._time[0] += time
            else:
                self._time[0] += 1 / self._rate

        self._fp_file.write('{:.3f}'.format(self._time[0]))
        for p in data:
            self._fp_file.write(' {:.4f} {:.4f} {:.4f}'.format(
                p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw))
        self._fp_file.write('\n')
        self._fp_file.flush()

        self._prev_filtered_poses = data

    def _unfiltered_poses_cb(self, msg):
        data = msg.poses
        if self._prev_unfiltered_poses is not None and len(data):
            time = (data[0].header.stamp -
                    self._prev_unfiltered_poses[0].header.stamp).to_sec()
            if time >= 0:
                self._time[1] += time
            else:
                self._time[1] += 1 / self._rate

        self._up_file.write('{:.3f}'.format(self._time[1]))
        for p in data:
            self._up_file.write(' {:.4f} {:.4f} {:.4f}'.format(
                p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw))
        self._up_file.write('\n')
        self._up_file.flush()

        self._prev_unfiltered_poses = data

    def _robot_poses_cb(self, msg):
        data = msg.poses
        if self._prev_robot_poses is not None and len(data):
            time = (data[0].header.stamp -
                    self._prev_unfiltered_poses[0].header.stamp).to_sec()
            if time >= 0:
                self._time[2] += time
            else:
                self._time[2] += 1 / self._rate

        self._ro_file.write('{:.3f}'.format(self._time[2]))
        for p in data:
            self._ro_file.write(' {:.4f} {:.4f} {:.4f}'.format(
                p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw))
        self._ro_file.write('\n')
        self._ro_file.flush()

        self._prev_robot_poses = data

    def __del__(self):
        self._fp_file.close()
        self._up_file.close()
        self._ro_file.close()


def main():
    rospy.init_node('pose_stat_node')
    rate = rospy.get_param("top_camera/fps", 60)
    rate = rospy.Rate(rate)
    ps = PoseStat()
    while not rospy.is_shutdown():
        ps.update()
        rate.sleep()


if __name__ == '__main__':
    main()
