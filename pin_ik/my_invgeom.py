"""
Stand-alone inverse geometry for a manipulator robot with a 6d objective.

Implement and solve the following nonlinear program:
decide q \in R^NQ
minimizing   || log( M(q)^-1 M^* ||^2
with M(q) \in SE(3) the placement of the robot end-effector, and M^* the
target.

The following tools are used:
- the ur10 model (loaded by example-robot-data)
- pinocchio.casadi for writing the problem and computing its derivatives
- the IpOpt solver wrapped in casadi
- the meshcat viewer
"""

# %jupyter_snippet imports
import time
import unittest
import example_robot_data as robex
import numpy as np
import casadi
import pinocchio as pin
import pinocchio.casadi as cpin
from visualizer import MeshcatVisualizer
# %end_jupyter_snippet

import sys
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton
)
from PyQt5.QtCore import Qt

class TargetPoseControl(QWidget):
    def __init__(self, init_pose):
        super().__init__()
        self.target_pose = init_pose.copy()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Target Pose Control")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.pos_sliders = []
        self.pos_edits = []
        for i, axis in enumerate(['X', 'Y', 'Z']):
            label = QLabel(f'Position {axis}')
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(int(self.target_pose[i]*100))
            slider.valueChanged.connect(self.make_pos_slider_changed(i))
            edit = QLineEdit(str(self.target_pose[i]))
            edit.editingFinished.connect(self.make_pos_edit_changed(i))
            self.pos_sliders.append(slider)
            self.pos_edits.append(edit)

            layout = QHBoxLayout()
            layout.addWidget(label)
            layout.addWidget(slider)
            layout.addWidget(edit)
            self.layout.addLayout(layout)

        self.quat_sliders = []
        self.quat_edits = []
        quat_names = ['w', 'x', 'y', 'z']
        for i, axis in enumerate(quat_names):
            label = QLabel(f'Quat {axis}')
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(int(self.target_pose[3 + i]*100))
            slider.valueChanged.connect(self.make_quat_slider_changed(i))
            edit = QLineEdit(str(self.target_pose[3 + i]))
            edit.editingFinished.connect(self.make_quat_edit_changed(i))
            self.quat_sliders.append(slider)
            self.quat_edits.append(edit)

            layout = QHBoxLayout()
            layout.addWidget(label)
            layout.addWidget(slider)
            layout.addWidget(edit)
            self.layout.addLayout(layout)

        btn = QPushButton("Normalize Quaternion")
        btn.clicked.connect(self.normalize_quaternion)
        self.layout.addWidget(btn)

    def make_pos_slider_changed(self, idx):
        def f(value):
            val = value / 100
            self.pos_edits[idx].setText(f"{val:.3f}")
            self.target_pose[idx] = val
        return f

    def make_pos_edit_changed(self, idx):
        def f():
            try:
                val = float(self.pos_edits[idx].text())
                val = max(min(val, 1), -1)
                self.target_pose[idx] = val
                self.pos_sliders[idx].setValue(int(val * 100))
            except:
                pass
        return f

    def make_quat_slider_changed(self, idx):
        def f(value):
            val = value / 100
            self.quat_edits[idx].setText(f"{val:.3f}")
            self.target_pose[3 + idx] = val
        return f

    def make_quat_edit_changed(self, idx):
        def f():
            try:
                val = float(self.quat_edits[idx].text())
                val = max(min(val, 1), -1)
                self.target_pose[3 + idx] = val
                self.quat_sliders[idx].setValue(int(val * 100))
            except:
                pass
        return f

    def normalize_quaternion(self):
        q = self.target_pose[3:]
        norm = (q @ q) ** 0.5
        if norm > 1e-9:
            self.target_pose[3:] = q / norm
            for i in range(4):
                self.quat_sliders[i].setValue(int(self.target_pose[3 + i]*100))
                self.quat_edits[i].setText(f"{self.target_pose[3 + i]:.3f}")

    def get_target_pose(self):
        return self.target_pose.copy()

def array2se(placement):
    R = pin.Quaternion(placement[3:]).matrix()
    return pin.SE3(R, placement[:3])

class MyEnv:

    def __init__(self):

        # --- ROBOT AND VIZUALIZER

        # %jupyter_snippet robot
        self.robot = robex.load('panda')
        self.model = self.robot.model
        self.data = self.robot.data
        self.positionLimit = np.vstack([self.model.lowerPositionLimit, self.model.upperPositionLimit])
        # %end_jupyter_snippet

        # %jupyter_snippet visualizer
        self.viz = MeshcatVisualizer(self.robot)
        self.viz.display(self.robot.q0)
        # %end_jupyter_snippet

        # %jupyter_snippet task_params
        self.robot.q0 = np.concatenate([np.sum(self.positionLimit[:, :7], axis=0)/2, self.positionLimit[1, -2:]])

        self.tool_id = self.model.getFrameId("panda_hand")

        # %end_jupyter_snippet

        print("Let's go to pdes ... with casadi")

        self.target_pose = np.array([0.4, 0.1, 0.3, 1, 0, 0, 0])
        self.last_target_pose = None

        self.ui = TargetPoseControl(self.target_pose)
        self.ui.show()

        # %jupyter_snippet visualizer_callback
        # --- Add box to represent target
        # Add a vizualization for the target
        self.boxID = "world/box"
        self.viz.addBox(self.boxID, [0.05, 0.1, 0.1], [1.0, 1.0, 0.3, 0.8])

        self.attachSphereID = "world/tcp_sphere"
        self.tcp_frame_id = self.model.getFrameId("panda_hand_tcp")
        self.viz.addSphere(self.attachSphereID, 0.03, [0, 0, 1, 0.7])

        self.joint_frame_id = [self.model.getFrameId(f"panda_joint{i}") for i in range(1, 8)]
        self.sphere_names = []
        for i in range(7):
            name = f"world/jointSphere_{i}"
            self.viz.addSphere(name, 0.07, [0,1,0,0.8])
            self.sphere_names.append(name)

    def displayScene(self, q):
        """
        Given the robot configuration, display:
        - the robot
        - a box representing tool_id
        - a box representing in_world_M_target
        """
        pin.framesForwardKinematics(self.model, self.data, q)
        self.viz.applyConfiguration(self.boxID, self.target_pose)
        tcp_pose = self.data.oMf[self.tcp_frame_id]
        self.viz.applyConfiguration(self.attachSphereID, tcp_pose)

        tau = pin.rnea(self.model, self.data, q, np.zeros(self.model.nq), np.zeros(self.model.nq))
        # print("----Joint torque: ", tau)
        tau_norm = np.clip(abs(tau[:7]) / 20.0, 0, 1)
        for i in range(7):
            self.viz.viewer[self.sphere_names[i]].set_property(key='color', value=[tau_norm[i], 1-tau_norm[i], 0, 0.8])
            self.viz.applyConfiguration(self.sphere_names[i], self.data.oMf[self.joint_frame_id[i]])

        self.viz.display(q)
        time.sleep(1e-1)
    # %end_jupyter_snippet

    def ik_solver(self):
        # --- CASADI MODEL AND HELPERS

        # %jupyter_snippet casadi_model
        # --- Casadi helpers
        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()
        # %end_jupyter_snippet

        # %jupyter_snippet cq
        cq = casadi.SX.sym("q", self.model.nq, 1)
        # %end_jupyter_snippet

        # %jupyter_snippet casadi_fk
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        # %end_jupyter_snippet

        # %jupyter_snippet casadi_error
        error_tool = casadi.Function(
            "etool",
            [cq],
            [
                cpin.log6(
                    cdata.oMf[self.tool_id].inverse() * cpin.SE3(array2se(self.target_pose))
                ).vector
            ],
        )
        # %end_jupyter_snippet

        # --- OPTIMIZATION PROBLEM

        # %jupyter_snippet casadi_computation_graph
        opti = casadi.Opti()
        var_q = opti.variable(self.model.nq)
        opti.set_initial(var_q, self.robot.q0)
        totalcost = casadi.sumsqr(error_tool(var_q))
        # %end_jupyter_snippet

        # %jupyter_snippet ipopt
        opti.minimize(totalcost)
        opti.solver("ipopt")  # select the backend solver
        opti.callback(lambda i: self.displayScene(opti.debug.value(var_q)))

        opti.subject_to([self.positionLimit[0] <= var_q, var_q <= self.positionLimit[1]])
        opti.subject_to(totalcost < 0.1)
        # %end_jupyter_snippet

        # %jupyter_snippet solve
        # Caution: in case the solver does not converge, we are picking the candidate values
        # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
        try:
            sol = opti.solve_limited()
            sol_q = opti.value(var_q)
        except:
            print("ERROR in convergence, plotting debug info.")
            sol_q = opti.debug.value(var_q)
        # %end_jupyter_snippet

        # %jupyter_snippet check_final_placement
        pin.framesForwardKinematics(self.model,self.data,sol_q)
        print(
            "The robot finally reached effector placement at\n",
            self.data.oMf[self.tool_id]
        )
        # %end_jupyter_snippet

    def update(self):
        ## read target pose
        self.target_pose = self.ui.get_target_pose()
        if self.last_target_pose is None or np.any(self.last_target_pose!=self.target_pose):
            self.ik_solver()
            self.last_target_pose = self.target_pose

if __name__=='__main__':
    app = QApplication(sys.argv)
    env = MyEnv()
    def env_loop():
        while True:
            time.sleep(0.1)
            env.update()

    threading.Thread(target=env_loop, daemon=True).start()
    sys.exit(app.exec_())