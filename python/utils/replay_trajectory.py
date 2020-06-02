from casadi import *
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tf as ros_tf
import geometry_msgs.msg
import rospy
from utils.normalize_quaternion import *

class replay_trajectory:
    def __init__(self, dt, joint_list, q_replay, contact_dict={}, kindyn=None):
        """
        Contructor
        Args:
            dt:
            joint_list:
            q_replay:
            contact_dict: dictionary containing:
                'frame': force
            kindyn: if passed the forces are rotated (only) in local frame

        """
        self.dt = dt
        self.joint_list = joint_list
        self.q_replay = q_replay
        self.force_pub = []
        self.__kindyn = kindyn

        self.__contact_dict = contact_dict
        if self.__kindyn != None:
            for frame in self.__contact_dict:
                FK = None
                for k in range(len(self.__contact_dict[frame])):
                    FK = Function.deserialize(self.__kindyn.fk(frame))
                    w_R_f = FK(q=self.q_replay[k])['ee_rot']
                    self.__contact_dict[frame][k] = mtimes(w_R_f.T, self.__contact_dict[frame][k]).T

    def publishContactForces(self, time, k):
        i = 0
        for frame in self.__contact_dict:
            f_msg = geometry_msgs.msg.WrenchStamped()
            f_msg.header.stamp = time
            f_msg.header.frame_id = frame

            f = self.__contact_dict[frame][k]

            # FK = None
            # if self.__kindyn != None:
            #     FK = Function.deserialize(self.__kindyn.fk(frame))
            #     w_R_f = FK(q=self.q_replay[k])['ee_rot']
            #     f = mtimes(w_R_f.T, self.__contact_dict[frame][k])

            f_msg.wrench.force.x = f[0]
            f_msg.wrench.force.y = f[1]
            f_msg.wrench.force.z = f[2]

            f_msg.wrench.torque.x = 0.
            f_msg.wrench.torque.y = 0.
            f_msg.wrench.torque.z = 0.

            self.force_pub[i].publish(f_msg)
            i += 1


    def replay(self):
        pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        rospy.init_node('joint_state_publisher')
        rate = rospy.Rate(1. / self.dt)
        joint_state_pub = JointState()
        joint_state_pub.header = Header()
        joint_state_pub.name = self.joint_list

        br = ros_tf.TransformBroadcaster()
        m = geometry_msgs.msg.TransformStamped()
        m.header.frame_id = 'world_odom'
        m.child_frame_id = 'base_link'

        nq = np.shape(self.q_replay)[1]
        n_res = np.shape(self.q_replay)[0]

        q_replay = normalize_quaternion(self.q_replay)

        for key in self.__contact_dict:
            self.force_pub.append(rospy.Publisher(key+'_forces',geometry_msgs.msg.WrenchStamped, queue_size=1))

        while not rospy.is_shutdown():
            for k in range(int(round(n_res))):
                qk = q_replay[k]

                m.transform.translation.x = qk[0]
                m.transform.translation.y = qk[1]
                m.transform.translation.z = qk[2]
                m.transform.rotation.x = qk[3]
                m.transform.rotation.y = qk[4]
                m.transform.rotation.z = qk[5]
                m.transform.rotation.w = qk[6]

                br.sendTransform((m.transform.translation.x, m.transform.translation.y, m.transform.translation.z),
                                 (m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z,
                                  m.transform.rotation.w),
                                 rospy.Time.now(), m.child_frame_id, m.header.frame_id)

                t = rospy.Time.now()
                joint_state_pub.header.stamp = t
                joint_state_pub.position = qk[7:nq]
                joint_state_pub.velocity = []
                joint_state_pub.effort = []
                pub.publish(joint_state_pub)
                self.publishContactForces(t, k)
                rate.sleep()