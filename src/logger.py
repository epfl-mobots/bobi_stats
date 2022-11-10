#!/usr/bin/env python
import rospy
import geometry_msgs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from bobi_msgs.msg import PoseVec, PoseStamped, KickSpecs, DLISpecs, MotorVelocities
from bobi_msgs.srv import ConvertCoordinates
from copy import deepcopy
from config.global_params import NUM_AGENTS, NUM_ROBOTS, NUM_VIRTU_AGENTS
from video_rec import VideoRec

import os
import socket
from datetime import datetime
import numpy as np

NOT_FOUND = 9999999.9
START_TIME = None


class PoseStat:
    def __init__(self):
        rospy.wait_for_service('/convert_bottom2top')
        rospy.wait_for_service('/convert_top2bottom')

        self._rate = rospy.get_param("top_camera/fps", 60)
        self._num_agents = NUM_AGENTS
        self._num_robots = NUM_ROBOTS
        self._num_virtu_agents = NUM_VIRTU_AGENTS

        self._of = rospy.get_param('logger/output_folder', '.')
        if '~' in self._of:
            home = os.path.expanduser('~')
            self._of = self._of.replace('~', home)
        hostname = socket.gethostname()
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y-%H_%M_%S")
        self._of = '{}/{}_{}'.format(self._of, hostname, date_time)
        os.makedirs(self._of)

        self._fp_init = False
        self._up_init = False
        self._ro_init = False
        self._ks_init = False
        self._dli_init = False
        self._sv_init = False

        self._fp_top_file = open(
            '{}/id_poses_top.txt'.format(self._of), 'w')
        self._fp_bot_file = open(
            '{}/id_poses_bot.txt'.format(self._of), 'w')

        self._up_top_file = open(
            '{}/raw_poses_top.txt'.format(self._of), 'w')
        self._up_bot_file = open(
            '{}/raw_poses_bot.txt'.format(self._of), 'w')

        self._ro_top_file = open(
            '{}/robot_poses_top.txt'.format(self._of), 'w')
        self._ro_bot_file = open(
            '{}/robot_poses_bot.txt'.format(self._of), 'w')

        self._ks_top_file = open(
            '{}/kick_specs_top.txt'.format(self._of), 'w')
        self._ks_bot_file = open(
            '{}/kick_specs_bot.txt'.format(self._of), 'w')

        self._dli_top_file = open(
            '{}/dli_specs_top.txt'.format(self._of), 'w')
        self._dli_bot_file = open(
            '{}/dli_specs_bot.txt'.format(self._of), 'w')

        self._sv_file = open(
            '{}/set_velocities.txt'.format(self._of), 'w')

        self._fp_sub = rospy.Subscriber(
            'filtered_poses', PoseVec, self._filtered_poses_cb)
        self._up_sub = rospy.Subscriber(
            'naive_poses', PoseVec, self._unfiltered_poses_cb)
        self._ro_sub = rospy.Subscriber(
            'robot_poses', PoseVec, self._robot_poses_cb)
        self._ks_sub = rospy.Subscriber(
            'kick_specs', KickSpecs, self._kick_specs_cb)
        self._dl_sub = rospy.Subscriber(
            'dli_specs', DLISpecs, self._dli_specs_cb)

        self._sv_sub = rospy.Subscriber(
            'set_velocities', MotorVelocities, self._motor_velocities_cb)

        self._bridge = CvBridge()

        self._top_annot_vr = None
        self._top_annot_img_sub = rospy.Subscriber(
            'top_camera/image_annot', Image, self._top_img_annot_cb)

        self._top_masked_vr = None
        self._top_masked_img_sub = rospy.Subscriber(
            'top_camera/image_masked', Image, self._top_img_masked_cb)

        self._bot_annot_vr = None
        self._bot_annot_img_sub = rospy.Subscriber(
            'bottom_camera/image_annot', Image, self._bot_img_annot_cb)

    def __del__(self):
        self._fp_top_file.close()
        self._fp_bot_file.close()
        self._up_top_file.close()
        self._up_bot_file.close()
        self._ro_top_file.close()
        self._ro_bot_file.close()
        self._ks_top_file.close()
        self._ks_bot_file.close()
        self._dli_top_file.close()
        self._dli_bot_file.close()
        self._sv_file.close()


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

        t = self._get_stamp()
        if not self._fp_init:
            self._fp_top_file.write('t ')
            self._fp_bot_file.write('t ')
            for _ in range(self._num_agents + self._num_virtu_agents):
                self._fp_top_file.write('x y yaw is_filtered is_swapped ')
                self._fp_bot_file.write('x y yaw is_filtered is_swapped ')
            self._fp_top_file.write('\n')
            self._fp_bot_file.write('\n')
            self._fp_init = True

        self._fp_top_file.write('{:5f}'.format(t))

        for p in data:
            self._fp_top_file.write(' {:.6f} {:.6f} {:.6f} {} {}'.format(
                p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, int(p.pose.is_filtered == True), int(p.pose.is_swapped == True)))
        if len(data) < self._num_agents + self._num_virtu_agents:
            for _ in range(self._num_agents + self._num_virtu_agents - len(data)):
                self._fp_top_file.write(' {} {} {} {} {}'.format(
                    NOT_FOUND, NOT_FOUND, NOT_FOUND, 0, 0))

        self._fp_top_file.write('\n')
        self._fp_top_file.flush()

        self._fp_bot_file.write('{:5f}'.format(t))

        for p in conv_data:
            self._fp_bot_file.write(' {:.6f} {:.6f} {:.6f} {} {}'.format(
                p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, int(p.pose.is_filtered == True), int(p.pose.is_swapped == True)))
        if len(conv_data) < self._num_agents + self._num_virtu_agents:
            for _ in range(self._num_agents + self._num_virtu_agents - len(conv_data)):
                self._fp_bot_file.write(' {} {} {} {} {}'.format(
                    NOT_FOUND, NOT_FOUND, NOT_FOUND, 0, 0))

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

        t = self._get_stamp()
        if not self._up_init:
            self._up_top_file.write('t ')
            self._up_bot_file.write('t ')
            for _ in range(self._num_agents + self._num_virtu_agents):
                self._up_top_file.write('x y yaw is_filtered is_swapped ')
                self._up_bot_file.write('x y yaw is_filtered is_swapped ')
            self._up_top_file.write('\n')
            self._up_bot_file.write('\n')
            self._up_init = True

        self._up_top_file.write('{:5f}'.format(t))
        for p in data:
            self._up_top_file.write(' {:.6f} {:.6f} {:.6f} {} {}'.format(
                p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, int(p.pose.is_filtered == True), int(p.pose.is_swapped == True)))
        if len(data) < self._num_agents + self._num_virtu_agents:
            for _ in range(self._num_agents + self._num_virtu_agents - len(data)):
                self._up_top_file.write(' {} {} {} {} {}'.format(
                    NOT_FOUND, NOT_FOUND, NOT_FOUND, 0, 0))

        self._up_top_file.write('\n')
        self._up_top_file.flush()

        self._up_bot_file.write('{:5f}'.format(t))
        for p in conv_data:
            self._up_bot_file.write(' {:.6f} {:.6f} {:.6f} {} {}'.format(
                p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, int(p.pose.is_filtered == True), int(p.pose.is_swapped == True)))
        if len(conv_data) < self._num_agents + self._num_virtu_agents:
            for _ in range(self._num_agents + self._num_virtu_agents - len(conv_data)):
                self._up_bot_file.write(' {} {} {} {} {}'.format(
                    NOT_FOUND, NOT_FOUND, NOT_FOUND, 0, 0))
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

        t = self._get_stamp()
        if not self._ro_init:
            self._ro_bot_file.write('t ')
            self._ro_top_file.write('t ')
            for _ in range(self._num_agents + self._num_virtu_agents):
                self._ro_bot_file.write('x y yaw is_filtered is_swapped ')
                self._ro_top_file.write('x y yaw is_filtered is_swapped ')
            self._ro_bot_file.write('\n')
            self._ro_top_file.write('\n')
            self._ro_init = True

        self._ro_bot_file.write('{:5f}'.format(t))
        for p in data:
            self._ro_bot_file.write(' {:.6f} {:.6f} {:.6f} {} {}'.format(
                p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, int(p.pose.is_filtered == True), int(p.pose.is_swapped == True)))
        if len(data) < self._num_agents + self._num_virtu_agents:
            for _ in range(self._num_agents + self._num_virtu_agents - len(data)):
                self._ro_bot_file.write(' {} {} {} {} {}'.format(
                    NOT_FOUND, NOT_FOUND, NOT_FOUND, 0, 0))

        self._ro_bot_file.write('\n')
        self._ro_bot_file.flush()

        self._ro_top_file.write('{:5f}'.format(t))
        for p in conv_data:
            self._ro_top_file.write(' {:.6f} {:.6f} {:.6f} {} {}'.format(
                p.pose.xyz.x, p.pose.xyz.y, p.pose.rpy.yaw, int(p.pose.is_filtered == True), int(p.pose.is_swapped == True)))
        if len(conv_data) < self._num_agents + self._num_virtu_agents:
            for _ in range(self._num_agents + self._num_virtu_agents - len(conv_data)):
                self._ro_top_file.write(' {} {} {} {} {}'.format(
                    NOT_FOUND, NOT_FOUND, NOT_FOUND, 0, 0))

        self._ro_top_file.write('\n')
        self._ro_top_file.flush()

    def _kick_specs_cb(self, msg):

        point = geometry_msgs.msg.Point(NOT_FOUND, NOT_FOUND, 0.)
        cagent = PoseStamped()
        cneighs = PoseVec()
        try:
            convSrv = rospy.ServiceProxy(
                '/convert_bottom2top', ConvertCoordinates)
            point = geometry_msgs.msg.Point(msg.target_x, msg.target_y, 0.)
            point = convSrv(point).converted_p

            p = geometry_msgs.msg.Point(
                msg.agent.pose.xyz.x, msg.agent.pose.xyz.y, 0.)
            p = convSrv(p).converted_p
            cagent = deepcopy(msg.agent)
            cagent.pose.xyz.x = p.x
            cagent.pose.xyz.y = p.y

            for n in msg.neighs.poses:
                p = geometry_msgs.msg.Point(n.pose.xyz.x, n.pose.xyz.y, 0.)
                p = convSrv(p).converted_p
                cn = deepcopy(n)
                cn.pose.xyz.x = p.x
                cn.pose.xyz.y = p.y
                cneighs.poses.append(cn)

        except rospy.ServiceException as e:
            rospy.logerr('Failed to convert robot position: {}'.format(e))

        t = self._get_stamp()
        if not self._ks_init:
            self._ks_bot_file.write('t x y yaw')
            for i in range(self._num_agents + self._num_virtu_agents - 1):
                self._ks_bot_file.write(' n_x n_y n_yaw')
            self._ks_bot_file.write(' target_x target_y dl phi dphi tau ta0')
            self._ks_bot_file.write('\n')

            self._ks_top_file.write('t x y yaw')
            for i in range(self._num_agents + self._num_virtu_agents - 1):
                self._ks_top_file.write(' n_x n_y n_yaw')
            self._ks_top_file.write(' target_x target_y dl phi dphi tau ta0')
            self._ks_top_file.write('\n')
            self._ks_init = True

        self._ks_bot_file.write('{:5f}'.format(t))
        self._ks_bot_file.write(' {:.6f} {:.6f} {:.6f}'.format(
            msg.agent.pose.xyz.x, msg.agent.pose.xyz.y, msg.agent.pose.rpy.yaw))
        for n in msg.neighs.poses:
            self._ks_bot_file.write(' {:.6f} {:.6f} {:.6f}'.format(
                n.pose.xyz.x, n.pose.xyz.y, n.pose.rpy.yaw))
        if len(msg.neighs.poses) < self._num_agents + self._num_virtu_agents - 1:
            for _ in range(self._num_agents + self._num_virtu_agents - 1 - len(msg.neighs.poses)):
                self._ks_bot_file.write(' {} {} {}'.format(
                    NOT_FOUND, NOT_FOUND, NOT_FOUND))

        self._ks_bot_file.write(' {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
            msg.target_x, msg.target_y, msg.dl, msg.phi, msg.dphi, msg.tau, msg.tau0))
        self._ks_bot_file.write('\n')
        self._ks_bot_file.flush()

        self._ks_top_file.write('{:5f}'.format(t))
        self._ks_top_file.write(' {:.6f} {:.6f} {:.6f}'.format(
            cagent.pose.xyz.x, cagent.pose.xyz.y, cagent.pose.rpy.yaw))
        for n in cneighs.poses:
            self._ks_top_file.write(' {:.6f} {:.6f} {:.6f}'.format(
                n.pose.xyz.x, n.pose.xyz.y, n.pose.rpy.yaw))
        if len(cneighs.poses) < self._num_agents + self._num_virtu_agents - 1:
            for _ in range(self._num_agents + self._num_virtu_agents - 1 - len(cneighs.poses)):
                self._ks_top_file.write(' {} {} {}'.format(
                    NOT_FOUND, NOT_FOUND, NOT_FOUND))

        self._ks_top_file.write(' {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
            point.x, point.y, msg.dl, msg.phi, msg.dphi, msg.tau, msg.tau0))
        self._ks_top_file.write('\n')
        self._ks_top_file.flush()

    def _dli_specs_cb(self, msg):

        point = geometry_msgs.msg.Point(NOT_FOUND, NOT_FOUND, 0.)
        cagent = PoseStamped()
        cneigh = PoseStamped()
        try:
            convSrv = rospy.ServiceProxy(
                '/convert_bottom2top', ConvertCoordinates)
            point = geometry_msgs.msg.Point(msg.target_x, msg.target_y, 0.)
            point = convSrv(point).converted_p

            p = geometry_msgs.msg.Point(
                msg.agent.pose.xyz.x, msg.agent.pose.xyz.y, 0.)
            p = convSrv(p).converted_p
            cagent = deepcopy(msg.agent)
            cagent.pose.xyz.x = p.x
            cagent.pose.xyz.y = p.y

            p = geometry_msgs.msg.Point(
                msg.neigh.pose.xyz.x, msg.neigh.pose.xyz.y, 0.)
            p = convSrv(p).converted_p
            cagent = deepcopy(msg.neigh)
            cneigh.pose.xyz.x = p.x
            cneigh.pose.xyz.y = p.y

        except rospy.ServiceException as e:
            rospy.logerr('Failed to convert robot position: {}'.format(e))

        t = self._get_stamp()
        if not self._dli_init:
            self._dli_bot_file.write('t gx gy sx sy target_x target_y')
            self._dli_bot_file.write(' x y yaw')
            self._dli_bot_file.write(' n_x n_y n_yaw')
            self._dli_bot_file.write('\n')

            self._dli_top_file.write('t gx gy sx sy target_x target_y')
            self._dli_top_file.write(' x y yaw')
            self._dli_top_file.write(' n_x n_y n_yaw')
            self._dli_top_file.write('\n')
            self._dli_init = True

        self._dli_bot_file.write('{:5f}'.format(t))
        self._dli_bot_file.write(' {:6f} {:6f} {:6f} {:6f}'.format(
            msg.gx, msg.gy, msg.sx, msg.sy))
        self._dli_bot_file.write(' {:.6f} {:.6f} {:.6f}'.format(
            msg.agent.pose.xyz.x, msg.agent.pose.xyz.y, msg.agent.pose.rpy.yaw))
        self._dli_bot_file.write(' {:.6f} {:.6f} {:.6f}'.format(
            msg.neigh.pose.xyz.x, msg.neigh.pose.xyz.y, msg.neigh.pose.rpy.yaw))
        self._dli_bot_file.write('\n')
        self._dli_bot_file.flush()

        self._dli_top_file.write('{:5f}'.format(t))
        self._dli_top_file.write(' {:6f} {:6f} {:6f} {:6f}'.format(
            msg.gx, msg.gy, msg.sx, msg.sy))
        self._dli_top_file.write(' {:.6f} {:.6f} {:.6f}'.format(
            cagent.pose.xyz.x, cagent.pose.xyz.y, cagent.pose.rpy.yaw))
        self._dli_top_file.write(' {:.6f} {:.6f} {:.6f}'.format(
            cneigh.pose.xyz.x, cneigh.pose.xyz.y, cneigh.pose.rpy.yaw))
        self._dli_top_file.write('\n')
        self._dli_top_file.flush()

    def _motor_velocities_cb(self, msg):
        t = self._get_stamp()
        self._sv_file.write('{} {} {}'.format(t, msg.left, msg.right))
        self._sv_file.write('\n')
        self._sv_file.flush()

    def _top_img_annot_cb(self, msg):
        t = self._get_stamp()
        img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        if img.shape[0] > 0 and img.shape[1] > 0:
            if self._top_annot_vr is None:
                fps = rospy.get_param('top_camera/fps')
                self._top_annot_vr = VideoRec(
                    '{}/top_annot'.format(self._of), img.shape[1], img.shape[0], fps)
            self._top_annot_vr.write(img, t)

    def _top_img_masked_cb(self, msg):
        t = self._get_stamp()
        img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        if img.shape[0] > 0 and img.shape[1] > 0:
            if self._top_masked_vr is None:
                fps = rospy.get_param('top_camera/fps')
                self._top_masked_vr = VideoRec(
                    '{}/top_masked'.format(self._of), img.shape[1], img.shape[0], fps)
            self._top_masked_vr.write(img, t)

    def _bot_img_annot_cb(self, msg):
        t = self._get_stamp()
        img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        if img.shape[0] > 0 and img.shape[1] > 0:
            if self._bot_annot_vr is None:
                fps = rospy.get_param('bottom_camera/fps')
                self._bot_annot_vr = VideoRec(
                    '{}/bot_annot'.format(self._of), img.shape[1], img.shape[0], fps)
            self._bot_annot_vr.write(img, t)

    def _get_stamp(self):
        global START_TIME, NOT_FOUND
        nsecs = rospy.Time.now().nsecs
        secs = rospy.Time.now().secs
        t = secs + (nsecs / 10 ** 9)
        if START_TIME is None:
            START_TIME = t
            t = 0
        else:
            t -= START_TIME
        return t


def main():
    rospy.init_node('logger_node')
    rate = rospy.get_param("top_camera/fps", 80)
    rate = rospy.Rate(rate)
    ps = PoseStat()
    while not rospy.is_shutdown():
        ps.update()
        rate.sleep()


if __name__ == '__main__':
    main()
