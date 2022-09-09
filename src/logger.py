#!/usr/bin/env python
import rospy
import geometry_msgs
from bobi_msgs.msg import PoseVec, PoseStamped, KickSpecs
from bobi_msgs.srv import ConvertCoordinates
from copy import deepcopy

import os
import socket
from datetime import datetime
import numpy as np


class PoseStat:
    def __init__(self):
        rospy.wait_for_service('/convert_bottom2top')
        rospy.wait_for_service('/convert_top2bottom')

        self._rate = rospy.get_param("top_camera/fps", 60)

        self._of = rospy.get_param('logger/output_folder', '.')
        if '~' in self._of:
            home = os.path.expanduser('~')
            self._of = self._of.replace('~', home)
        hostname = socket.gethostname()
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y-%H_%M_%S")
        self._of = '{}/{}_{}'.format(self._of, hostname, date_time)
        os.makedirs(self._of)

        self._fp_top_file = open(
            '{}/filtered_poses_top.txt'.format(self._of), 'w')
        self._fp_bot_file = open(
            '{}/filtered_poses_bot.txt'.format(self._of), 'w')

        self._up_top_file = open(
            '{}/unfiltered_poses_top.txt'.format(self._of), 'w')
        self._up_bot_file = open(
            '{}/unfiltered_poses_bot.txt'.format(self._of), 'w')

        self._ro_top_file = open(
            '{}/robot_poses_top.txt'.format(self._of), 'w')
        self._ro_bot_file = open(
            '{}/robot_poses_bot.txt'.format(self._of), 'w')

        self._ks_top_file = open(
            '{}/kick_specs_top.txt'.format(self._of), 'w')
        self._ks_bot_file = open(
            '{}/kick_specs_bot.txt'.format(self._of), 'w')

        self._fp_sub = rospy.Subscriber(
            'filtered_poses', PoseVec, self._filtered_poses_cb)
        self._up_sub = rospy.Subscriber(
            'naive_poses', PoseVec, self._unfiltered_poses_cb)
        self._ro_sub = rospy.Subscriber(
            'robot_poses', PoseVec, self._robot_poses_cb)
        self._ks_sub = rospy.Subscriber(
            'kick_specs', KickSpecs, self._kick_specs_cb)

    def __del__(self):
        self._fp_top_file.close()
        self._fp_bot_file.close()
        self._up_top_file.close()
        self._up_bot_file.close()
        self._ro_top_file.close()
        self._ro_bot_file.close()
        self._ks_top_file.close()
        self._ks_bot_file.close()

    def update(self):
        pass

    def _filtered_poses_cb(self, msg):
        data = msg.poses

        conv_data = []
        for p in data:
            try:
                convSrv = rospy.ServiceProxy(
                    '/convert_top2bottom', ConvertCoordinates)
                point = geometry_msgs.msg.Point(p.pose.xyz.x, p.pose.xyz.y, 0.)
                resp = convSrv(point)
                pose = PoseStamped()
                pose = deepcopy(p)
                pose.pose.xyz.x = resp.converted_p.x
                pose.pose.xyz.y = resp.converted_p.y
                conv_data.append(pose)
            except rospy.ServiceException as e:
                rospy.logerr('Failed to convert unfiltered pose: {}'.format(e))

        nsecs = rospy.Time.now().nsecs
        secs = rospy.Time.now().secs

        self._fp_top_file.write('{} {}'.format(secs, nsecs))
        if (len(data) == 0):
            self._fp_top_file.write(' {} {} {} {}'.format(
                np.nan, np.nan, np.nan, np.nan))
        else:
            for p in data:
                self._fp_top_file.write(' {:.4f} {:.4f} {:.4f} {}'.format(
                    p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, p.pose.is_filtered))
        self._fp_top_file.write('\n')
        self._fp_top_file.flush()

        self._fp_bot_file.write('{} {}'.format(secs, nsecs))
        if (len(conv_data) == 0):
            self._fp_bot_file.write(' {} {} {} {}'.format(
                np.nan, np.nan, np.nan, np.nan))
        else:
            for p in conv_data:
                self._fp_bot_file.write(' {:.4f} {:.4f} {:.4f} {}'.format(
                    p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, p.pose.is_filtered))
        self._fp_bot_file.write('\n')
        self._fp_bot_file.flush()

    def _unfiltered_poses_cb(self, msg):
        data = msg.poses

        conv_data = []
        for p in data:
            try:
                convSrv = rospy.ServiceProxy(
                    '/convert_top2bottom', ConvertCoordinates)
                point = geometry_msgs.msg.Point(p.pose.xyz.x, p.pose.xyz.y, 0.)
                resp = convSrv(point)
                pose = PoseStamped()
                pose = deepcopy(p)
                pose.pose.xyz.x = resp.converted_p.x
                pose.pose.xyz.y = resp.converted_p.y
                conv_data.append(pose)
            except rospy.ServiceException as e:
                rospy.logerr('Failed to convert unfiltered pose: {}'.format(e))

        nsecs = rospy.Time.now().nsecs
        secs = rospy.Time.now().secs

        self._up_top_file.write('{} {}'.format(secs, nsecs))
        if (len(data) == 0):
            self._up_top_file.write(' {} {} {} {}'.format(
                np.nan, np.nan, np.nan, np.nan))
        else:
            for p in data:
                self._up_top_file.write(' {:.4f} {:.4f} {:.4f} {}'.format(
                    p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, p.pose.is_filtered))
        self._up_top_file.write('\n')
        self._up_top_file.flush()

        self._up_bot_file.write('{} {}'.format(secs, nsecs))
        if (len(conv_data) == 0):
            self._up_bot_file.write(' {} {} {} {}'.format(
                np.nan, np.nan, np.nan, np.nan))
        else:
            for p in conv_data:
                self._up_bot_file.write(' {:.4f} {:.4f} {:.4f} {}'.format(
                    p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, p.pose.is_filtered))
        self._up_bot_file.write('\n')
        self._up_bot_file.flush()

    def _robot_poses_cb(self, msg):
        data = msg.poses

        conv_data = []
        for p in data:
            try:
                convSrv = rospy.ServiceProxy(
                    '/convert_bottom2top', ConvertCoordinates)
                point = geometry_msgs.msg.Point(p.pose.xyz.x, p.pose.xyz.y, 0.)
                resp = convSrv(point)
                pose = PoseStamped()
                pose = deepcopy(p)
                pose.pose.xyz.x = resp.converted_p.x
                pose.pose.xyz.y = resp.converted_p.y
                conv_data.append(pose)
            except rospy.ServiceException as e:
                rospy.logerr('Failed to convert robot pose: {}'.format(e))

        nsecs = rospy.Time.now().nsecs
        secs = rospy.Time.now().secs

        self._ro_bot_file.write('{} {}'.format(secs, nsecs))
        if (len(data) == 0):
            self._ro_bot_file.write(' {} {} {} {}'.format(
                np.nan, np.nan, np.nan, np.nan))
        else:
            for p in data:
                self._ro_bot_file.write(' {:.4f} {:.4f} {:.4f} {}'.format(
                    p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, p.pose.is_filtered))
        self._ro_bot_file.write('\n')
        self._ro_bot_file.flush()

        self._ro_top_file.write('{} {}'.format(secs, nsecs))
        if (len(conv_data) == 0):
            self._ro_top_file.write(' {} {} {} {}'.format(
                np.nan, np.nan, np.nan, np.nan))
        else:
            for p in conv_data:
                self._ro_top_file.write(' {:.4f} {:.4f} {:.4f} {}'.format(
                    p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, p.pose.is_filtered))
        self._ro_top_file.write('\n')
        self._ro_top_file.flush()

    def _kick_specs_cb(self, msg):

        point = geometry_msgs.msg.Point(np.nan, np.nan, 0.)
        try:
            convSrv = rospy.ServiceProxy(
                '/convert_bottom2top', ConvertCoordinates)
            point = geometry_msgs.msg.Point(msg.target_x, msg.target_y, 0.)
            point = convSrv(point).converted_p
        except rospy.ServiceException as e:
            rospy.logerr('Failed to convert robot position: {}'.format(e))

        nsecs = rospy.Time.now().nsecs
        secs = rospy.Time.now().secs

        self._ks_bot_file.write('{} {}'.format(secs, nsecs))
        self._ks_bot_file.write(' {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
            msg.target_x, msg.target_y, msg.dl, msg.phi, msg.dphi, msg.tau, msg.tau0))
        self._ks_bot_file.write('\n')
        self._ks_bot_file.flush()

        self._ks_top_file.write('{} {}'.format(secs, nsecs))
        self._ks_top_file.write(' {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
            point.x, point.y, msg.dl, msg.phi, msg.dphi, msg.tau, msg.tau0))
        self._ks_top_file.write('\n')
        self._ks_top_file.flush()


def main():
    rospy.init_node('logger_node')
    rate = rospy.get_param("top_camera/fps", 60)
    rate = rospy.Rate(rate)
    ps = PoseStat()
    while not rospy.is_shutdown():
        ps.update()
        rate.sleep()


if __name__ == '__main__':
    main()
