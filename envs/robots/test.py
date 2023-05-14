from trac_ik_python.trac_ik import IK
urdfstring = ''.join(open('/home/tumu/anaconda3/envs/spinningup/kuka-reach-drl/models/franka_panda/panda.urdf', 'r').readlines())
ik = IK('panda_link0', 'panda_hand', urdf_string=urdfstring)

print("link names:",ik.link_names)
print("base link:",ik.base_link)
print("tip link:", ik.tip_link)
print("joint names:",ik.joint_names)
print("number of joints:", ik.number_of_joints)

q_in = [   -0.222097,0.102015, 0.19183,-2.01216, -0.0403435, 2.27308, 0.83161]
print(ik.get_ik(q_in, 
                0.05,-0.01,0.38,
                0.996, -0.027, 0.082, 0.006))
