import os
import pickle
from math import factorial
from copy import deepcopy

import numpy as np
import seaborn as sns
import os
from PyQt5.Qt import QStandardItemModel, QStandardItem
from qtrangeslider import QRangeSlider
import tensorflow as tf
import json
from utils import NumpyEncoder
import importlib
sns.set_theme(style="dark")

import matplotlib.pyplot as plt
import sys
import configparser

from motivation.random_network_distillation import RND
#from clustering.cluster_im import cluster
from clustering.clustering import cluster_trajectories as cluster_simple
from clustering.clustering import frechet_distance
from matplotlib import cm
import collections
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from vispy import app, visuals, scene, gloo

import PyQt5
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QDateTime, Qt, QTimer, QObject
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFrame, QStackedLayout, QListView, QTreeView)

PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

EPSILON = sys.float_info.epsilon
from PyQt5.QtCore import QThread, pyqtSignal
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class WorlModelCanvas(QObject, scene.SceneCanvas):
    heatmap_signal = pyqtSignal(bool)
    filtering_mean_signal = pyqtSignal(float)
    cluster_size_signal = pyqtSignal(int)
    in_time_signal = pyqtSignal(int)
    plot_traj_signal = pyqtSignal(list)
    traj_list_signal = pyqtSignal(dict)
    cluster_selected_signal = pyqtSignal(int)
    load_ending_signal = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        self.current_line = None
        self.lines = None
        self.traj_visuals = []
        self.traj_positions = []
        self.im_rews = []
        self.index = -1
        self.timer = None
        self.agent = None
        self.agent_timer = None
        self.agent_speed = 5
        self.animation_index = None
        self.movement_vector = (0, 0, 0)
        self.camera = None
        self.actions = []
        self.colors = []
        self.gradients = []
        self.trajs = []
        self.demonstrations = []
        self.dem_visuals = []
        self.view = None
        self.default_colors = (0, 1, 1, 1)
        self.default_color = False
        self.gradient_color = False
        self.current_color_mode = 'none'
        self.one_line = False
        self.covermap = None
        self.heatmap = None
        self.heatmap_in_time = []
        self.label = None
        self.loading = None
        self.cluster_size = 20
        self.smooth_window = 5
        self.trajectory_visualizer = True
        self.heatmap_in_time_discr = 10000

        self.tm = 0
        self.tn = 0

        self.config = configparser.ConfigParser()

        self.mean_moti_thr = 0.04
        self.sum_moti_thr = 16
        self.tmp_line = None
        self.tmp_traj = None
        self.tmp_im_rew = None

        super(WorlModelCanvas, self).__init__()
        self.unfreeze()
        scene.SceneCanvas.__init__(self, *args, **kwargs)
        # QObject.__init__(self, *args, **kwargs)
        self.size = (1920, 1080)
        self.title = 'World Model Analysis'
        self.freeze()

    def set_camera(self, camera):
        self.camera = camera

    def set_view(self, view):
        self.view = view

    def set_label(self, label):
        self.label = label

    def set_loading(self,loading):
        self.loading = loading

    def on_key_press(self, event):
        if event.key.name == 'L':
            self.toggle_lines()

        if event.key.name == 'R':
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            else:
                self.rotate()

        if event.key.name == 'Space':
            self.reset_index()

        if event.key.name == 'F1':
            if self.current_color_mode == 'none':
                self.change_line_colors('gradient')
                self.current_color_mode = 'gradient'
            elif self.current_color_mode == 'gradient':
                self.change_line_colors('default')
                self.current_color_mode = 'default'
            elif self.current_color_mode == 'default':
                self.change_line_colors('none')
                self.current_color_mode = 'none'


        if event.key.name == 'F2':
            self.change_map()

        if event.key.name == '0':
            self.plot_3D_alpha_map(self.world_model, 0)

        if event.key.name == '1':
            self.plot_3D_alpha_map(self.world_model, 1)

        if event.key.name == '4':
            self.plot_3D_alpha_map(self.world_model, 4)

        if event.key.name == '6':
            self.plot_3D_alpha_map(self.world_model, 6)

        if event.key.name == '8':
            self.plot_3D_alpha_map(self.world_model, 8)

        if event.key.name == '9':
            self.plot_3D_alpha_map(self.world_model, 9)

        if event.key.name == 'O':
            self.plot_3D_alpha_map(self.world_model, 10)

        if event.key.name == 'I':
            self.plot_3D_alpha_map(self.world_model, 11)

        if event.key.name == 'Up' or event.key.name == 'Down':

            if event.key.name == 'Up':
                self.index -= 1

            if event.key.name == 'Down':
                self.index += 1

            self.index = np.clip(self.index, -1, len(self.traj_visuals))

            if self.index == -1 or self.index == len(self.traj_visuals):
                self.one_line = False
                self.hide_all_lines()
                self.toggle_lines()
                self.plot_traj_signal.emit(np.asarray([]))
                self.cluster_selected_signal.emit(-1)
                return

            self.plot_one_traj(self.index)

        if event.key.name == 'A':
            self.delete_agent()
            self.create_agent(self.trajs[self.index][0, :3])
            self.animation_index = 1
            self.move_agent()

        if event.key.name == 'P':
            plt.close('all')
            if self.index == -1 or self.index == len(self.traj_visuals):
                return

            if self.im_rews[self.index] is not None:
                plt.figure()
                plt.title("sum: {}, mean: {}".format(self.sum_moti_rews_dict[self.index], self.mean_moti_rews_dict[self.index]))
                plot_data = self.im_rews[self.index]
                # plot_data = self.savitzky_golay(plot_data, 21, 3)
                # plot_data = (plot_data - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))
                # plt.plot(range(len(plot_data)), plot_data)
                plot_data = np.asarray(plot_data)
                # self.im_rew_signal.emit(plot_data)
                plt.plot(range(len(plot_data)), plot_data)

                if self.actions[self.index] is not None:
                    plt.figure()
                    plt.hist(self.actions[self.index])

                plt.show()

    def plot_one_traj(self, line_index):
        self.delete_agent()
        self.hide_all_lines()
        self.traj_visuals[line_index].visible = True
        self.cluster_selected_signal.emit(line_index)

        if self.im_rews[line_index] is not None:
            self.plot_traj_data(self.im_rews[line_index], self.actions[line_index])

        actions_to_save = dict(actions=self.actions[line_index])
        json_str = json.dumps(actions_to_save, cls=NumpyEncoder)
        f = open("arrays/actions.json", "w")
        f.write(json_str)
        f.close()


        # traj_to_save = dict(x_s=self.trajs[line_index][:, 0], z_s=self.trajs[line_index][:, 1], y_s=self.trajs[line_index][:, 2],
        #                     im_values=np.zeros(501), il_values=np.zeros(501))
        # json_str = json.dumps(traj_to_save, cls=NumpyEncoder)
        # f = open("../Playtesting-Env/Assets/Resources/traj.json", "w")
        # f.write(json_str)
        # f.close()

        self.one_line = True

    def plot_traj_data(self, im_rews, actions):
        # plt.figure()
        # plt.title("im: {}".format(np.sum(self.im_rews[self.index])))
        plot_data = im_rews
        plot_data = self.savitzky_golay(plot_data, 21, 3)
        # plot_data = (plot_data - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))
        # plt.plot(range(len(plot_data)), plot_data)
        plot_data = np.asarray(plot_data)
        self.plot_traj_signal.emit([plot_data, actions])

    def change_line_colors(self, mode='default'):
        for i, v in enumerate(self.traj_visuals):
            v.set_data(meshdata=v._meshdata)
            colors = np.ones((len(v._meshdata._vertices), 4))
            if mode == 'default':
                colors[:, ] = self.default_colors
            elif mode == 'gradient':
                colors[:, ] = self.gradients[i]
            else:
                colors[:, ] = self.colors[i]

            v._meshdata.set_vertex_colors(colors)

    def toggle_lines(self):
        if self.index == -1 or self.index == len(self.traj_visuals) or not self.one_line:
            if self.traj_visuals is not None and len(self.traj_visuals) > 0:
                for v in self.traj_visuals:
                    v.visible = not v.visible
        else:
            self.traj_visuals[self.index].visible = not self.traj_visuals[self.index].visible

    def hide_all_lines(self):
        if self.traj_visuals is not None and len(self.traj_visuals) > 0:
            for v in self.traj_visuals:
                v.visible = False

        if self.tmp_line is not None:
            self.tmp_line.visible = False
            self.tmp_line = None
            self.tmp_traj = None
            self.tmp_im_rew = None

    def reset_index(self):
        if self.index == -1 or self.index == len(self.traj_visuals):
            return
        if self.one_line:
            tmp_index = self.index
            self.index = -1
            self.hide_all_lines()
            self.toggle_lines()
            self.one_line = False
            self.index = tmp_index
        else:
            self.hide_all_lines()
            plt.close('all')
            self.traj_visuals[self.index].visible = True
            self.one_line = True

    def on_mouse_press(self, event):
        return

    def change_map(self):
        if self.heatmap == None:
            return
        self.heatmap.visible = not self.heatmap.visible
        self.covermap.visible = not self.covermap.visible
        to_emit = True if self.heatmap.visible else False
        self.heatmap_signal.emit(to_emit)

    def set_line(self, line):
        self.current_line = line

    def set_lines(self, lines):
        self.lines = lines

    def remove_maps(self):
        if self.heatmap == None:
            return

        self.heatmap.visible = False
        self.covermap.visible = False
        self.heatmap.parent = None
        self.covermap.parent = None
        self.heatmap = None
        self.covermap = None

        for h in self.heatmap_in_time:
            h.visible = False
            h.parent = None

        del self.heatmap_in_time[:]
        self.heatmap_in_time = []

        # del self.demonstrations[:]
        self.demonstrations = []

        for d in self.dem_visuals:
            d.visible = False
        del self.dem_visuals[:]
        self.dem_visuals = []

        self.remove_lines()

    def remove_lines(self, only_visuals=False):
        for v in self.traj_visuals:
            v.visible = False
            v.parent = None

        del self.traj_visuals[:]
        self.traj_visuals = []

        if self.tmp_line is not None:
            self.tmp_line.visible = False

        if not only_visuals:
            del self.im_rews[:]
            self.im_rews = []

            del self.actions[:]
            self.actions = []

            del self.colors[:]
            self.colors = []

            del self.gradients[:]
            self.gradients = []

            del self.trajs[:]
            self.trajs = []

            self.index = -1

            self.tmp_line = None
            self.tmp_traj = None
            self.tmp_im_rew = None

    def set_maps(self, heatmap, covermap):
        self.heatmap = heatmap
        self.covermap = covermap

    def set_line_visuals(self, visual, positions, im_rews=None, actions=None, color=None):
        self.traj_visuals.append(visual)
        self.im_rews.append(im_rews)
        self.actions.append(actions)
        self.colors.append(color)
        self.traj_positions.append(positions)

    def random_color(self, value):
        return (np.random.uniform(), np.random.uniform(), np.random.uniform(), 1)

    def change_text(self, filename, points, episodes):

        label = '{}\ncoverage of points: {}\ntotal training episodes: {}'.format(filename, points, episodes)

        self.label.text = label

    def normalize(self, value, rmin, rmax, tmin, tmax):
        rmin = float(rmin)
        rmax = float(rmax)
        tmin = float(tmin)
        tmax = float(tmax)
        return ((value - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin

    def convert_to_rgb(self, minval, maxval, val, colors=[(150, 0, 0), (255, 255, 0), (255, 255, 255)]):

        i_f = float(val - minval) / float(maxval - minval) * (len(colors) - 1)
        i, f = int(i_f // 1), i_f % 1
        if f < EPSILON:
            return colors[i][0] / 255, colors[i][1] / 255, colors[i][2] / 255, 1
        else:
            (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i + 1]
            return int(r1 + f * (r2 - r1)) / 255, int(g1 + f * (g2 - g1)) / 255, int(b1 + f * (b2 - b1)) / 255, 1

    def trajectories_to_pos_buffer(self, trajectories, rnd=None):
        pos_buffer = dict()
        rnd_buffer = dict()
        pos_buffer_in_time = dict()
        pos_buffers_in_time = []
        pos_buffer_is_grounded = dict()
        pos_buffer_alpha = dict()
        count = 0

        rmin_x = float(self.config['DEFAULT']['rmin_x'])
        rmax_x = float(self.config['DEFAULT']['rmax_x'])
        tmin_x = float(self.config['DEFAULT']['tmin_x'])
        tmax_x = float(self.config['DEFAULT']['tmax_x'])

        rmin_y = float(self.config['DEFAULT']['rmin_y'])
        rmax_y = float(self.config['DEFAULT']['rmax_y'])
        tmin_y = float(self.config['DEFAULT']['tmin_y'])
        tmax_y = float(self.config['DEFAULT']['tmax_y'])

        rmin_z = float(self.config['DEFAULT']['rmin_z'])
        rmax_z = float(self.config['DEFAULT']['rmax_z'])
        tmin_z = float(self.config['DEFAULT']['tmin_z'])
        tmax_z = float(self.config['DEFAULT']['tmax_z'])

        self.heatmap_in_time_discr = int(len(list(trajectories.values())) / 60)

        for traj in list(trajectories.values())[:30000]:
            count += 1
            for state in traj:
                position = np.asarray(state[:3])
                # pos_key = ' '.join(map(str, position))
                # if pos_key not in rnd_buffer:
                #     rnd_buffer[pos_key] = 1
                position[0] = self.normalize(position[0],
                                             rmin_x,
                                             rmax_x,
                                             tmin_x,
                                             tmax_x)
                position[1] = self.normalize(position[1],
                                             rmin_y,
                                             rmax_y,
                                             tmin_y,
                                             tmax_y)
                position[2] = self.normalize(position[2],
                                             rmin_z,
                                             rmax_z,
                                             tmin_z,
                                             tmax_z)

                position = position.astype(int)
                pos_key = ' '.join(map(str, position))
                if pos_key in pos_buffer.keys():
                    if state[3] == 1:
                        pos_buffer[pos_key] += 1
                        pos_buffer_is_grounded[pos_key] = state[3]
                        pos_buffer_alpha[pos_key][int(state[-2] * 10)] += 1
                else:
                    pos_buffer_is_grounded[pos_key] = state[3]
                    if state[3] == 1:
                        pos_buffer[pos_key] = 1
                        alphas = np.zeros(11)
                        alphas[int(state[-2] * 10)] = 1
                        pos_buffer_alpha[pos_key] = alphas

                if pos_key in pos_buffer_in_time.keys():
                    if state[3] == 1:
                        pos_buffer_in_time[pos_key] += 1
                else:
                    if state[3] == 1:
                        pos_buffer_in_time[pos_key] = 1

            if count % self.heatmap_in_time_discr == 0:
                pos_buffers_in_time.append(pos_buffer_in_time)
                pos_buffer_in_time = dict()

        world_model = []
        for k in pos_buffer.keys():
            k_value = list(map(float, k.split(" ")))
            if pos_buffer_is_grounded[k] == 1:
                heat = pos_buffer[k]
                alpha_heat = pos_buffer_alpha[k]
                world_model.append(k_value[:3] + [heat] + list(alpha_heat))

        # rnd_heatmap = []
        # batch_size = 512
        # k_list = list(rnd_buffer.keys())
        # num_iter = int(np.ceil(len(k_list) / batch_size))
        # for i in range(num_iter):
        #     batch = k_list[i * batch_size: i * batch_size + batch_size]
        #     state_batch = []
        #     for s in batch:
        #         s = list(map(float, s.split(" ")))
        #         s = np.concatenate([s, np.zeros(70)])
        #         state_batch.append(dict(global_in=s))
        #
        #     rnd_values = rnd.eval(state_batch)
        #     for s, v in zip(batch, rnd_values):
        #         s = list(map(float, s.split(" ")))
        #         s = np.asarray(s[:3])
        #         s[0] = (((s[0] + 1) / 2) * 500)
        #         s[1] = (((s[1] + 1) / 2) * 500)
        #         s[2] = (((s[2] + 1) / 2) * 60)
        #         s = s.astype(int)
        #         k = ' '.join(map(str, s))
        #         if pos_buffer_is_grounded[k] == 1:
        #             rnd_heatmap.append(list(s) + [v])

        return pos_buffer, pos_buffer_alpha, world_model, pos_buffers_in_time, pos_buffer_is_grounded

    def heatmap_in_time_given_tm(self, tm):
        for h in self.heatmap_in_time:
            h.visible = False
        del self.heatmap_in_time[:]
        self.heatmap_in_time = []

        if self.model_name == 'play_2_500_2':
            min_perc = np.percentile(self.world_model[:, 3], 2)
            max_perc = np.percentile(self.world_model[:, 3], 98)
        else:
            min_perc = np.percentile(self.world_model[:, 3], 5)
            max_perc = np.percentile(self.world_model[:, 3], 95)

        tmp_buffer = dict()
        for buffer in self.pos_buffers_in_time[tm:]:
            world_model = []
            for k in buffer.keys():
                if self.pos_buffer_is_grounded[k] == 1:
                    heat = buffer[k]
                    if k in tmp_buffer.keys():
                        tmp_buffer[k] += heat
                    else:
                        tmp_buffer[k] = heat
            for tmp_k in tmp_buffer.keys():
                k_value = list(map(float, tmp_k.split(" ")))
                heat = tmp_buffer[tmp_k]
                world_model.append(k_value[:3] + [heat])

            world_model = np.asarray(world_model)
            world_model[:, 3] = np.clip(world_model[:, 3], min_perc, max_perc)

            min_value = np.min(world_model[:, 3])
            max_value = np.max(world_model[:, 3])

            colors = []
            for c in world_model[:, 3]:
                colors.append(self.convert_to_rgb(min_value, max_value, c))
            colors = np.asarray(colors)

            Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
            heatmap = Scatter3D(parent=view.scene)
            heatmap.set_gl_state('additive', blend=True, depth_test=True)
            heatmap.set_data(world_model[:, :3], face_color=colors, symbol='o', size=0.7, edge_width=0,
                             edge_color=colors,
                             scaling=True)

            heatmap.visible = False
            self.heatmap_in_time.append(heatmap)



    def plot_3d_map(self, world_model, color_index=3, percentile=95,
                    colors_gradient=[(150, 0, 0), (255, 255, 0), (255, 255, 255)], set_map=True):
        world_model_array = deepcopy(np.asarray(world_model))

        if self.model_name == 'play_2_500_2':
            percentile = 98

        world_model_array[:, color_index] = np.clip(world_model_array[:, color_index],
                                                    np.percentile(world_model_array[:, 3], 100 - percentile),
                                                    np.percentile(world_model_array[:, 3], percentile))

        min_value = np.min(world_model_array[:, color_index])
        max_value = np.max(world_model_array[:, color_index])

        print(min_value)
        print(max_value)

        colors = []
        for i, c in enumerate(world_model[:, color_index]):
            if c != 0:
                colors.append(self.convert_to_rgb(min_value, max_value, world_model_array[i, color_index],
                                                  colors=colors_gradient))
            else:
                colors.append((0,0,0,0))

        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        colors = np.asarray(colors)
        heatmap = Scatter3D(parent=view.scene)
        heatmap.set_gl_state('additive', blend=True, depth_test=True)
        heatmap.set_data(world_model_array[:, :3], face_color=colors, symbol='o', size=0.7, edge_width=0,
                         edge_color=colors,
                         scaling=True)

        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        covermap = Scatter3D(parent=view.scene)
        covermap.set_gl_state('additive', blend=True, depth_test=True)
        covermap.set_data(world_model_array[:, :3], face_color=(0.61, 0, 0, 1), symbol='o', size=0.7, edge_width=0,
                          edge_color=(1, 0, 0, 1), scaling=True)
        covermap.visible = False

        if set_map:
            self.set_maps(heatmap, covermap)
        return world_model_array

    def plot_3D_alpha_map(self, world_model, alpha=0):

        for h in self.heatmap_in_time:
            h.visible = False

        self.heatmap.visible = False
        self.plot_3d_map(world_model, color_index=alpha + 3)

    def plot_3d_map_in_time(self, world_model_in_time):
        return
        min_perc = np.percentile(np.asarray(world_model_in_time[-1])[:, 3], 2)
        max_perc = np.percentile(np.asarray(world_model_in_time[-1])[:, 3], 95)
        for world_model in world_model_in_time:
            world_model = np.asarray(world_model)
            world_model[:, 3] = np.clip(world_model[:, 3], min_perc,
                                        max_perc)

            min_value = np.min(world_model[:, 3])
            max_value = np.max(world_model[:, 3])

            colors = []
            for c in world_model[:, 3]:
                colors.append(self.convert_to_rgb(min_value, max_value, c))
            colors = np.asarray(colors)

            Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
            heatmap = Scatter3D(parent=view.scene)
            heatmap.set_gl_state('additive', blend=True, depth_test=True)
            heatmap.set_data(world_model[:, :3], face_color=colors, symbol='o', size=0.7, edge_width=0, edge_color=colors,
                             scaling=True)

            heatmap.visible = False
            self.heatmap_in_time.append(heatmap)

    def show_heatmap_in_time(self, index):
        if self.heatmap is not None:
            self.heatmap.visible = False
            self.covermap.visible = False
            for h in self.heatmap_in_time:
                h.visible = False
            index = np.clip(index, 0, len(self.heatmap_in_time) - 1)
            if index == len(self.heatmap_in_time) - 1 and self.tm == 0:
                self.heatmap.visible = True
                self.heatmap_signal.emit(True)
            else:
                self.heatmap_in_time[index].visible = True

    def start_loading(self):
        self.loading.visible = True

    def stop_loading(self):
        self.loading.visible = False

    def create_agent(self, starting_position=(255, 255, 0)):
        starting_position = starting_position.astype(int)
        if self.agent == None:
            Cube = scene.visuals.create_visual_node(visuals.CubeVisual)
            self.agent = Cube(parent=self.view.scene, color='white', size=10)
            # Define a scale and translate transformation :
            self.agent.transform = visuals.transforms.STTransform(translate=starting_position)
        else:
            self.agent.visible = True
            self.movement_vector = (0, 0, 0)
            self.agent.transform = visuals.transforms.STTransform(translate=starting_position)

    def delete_agent(self):
        if self.agent is None:
            return
        if self.agent_timer is not None:
            self.agent_timer.cancel()
            self.agent_timer = None
        self.agent.visible = False

    def rotate(self):
        self.timer = threading.Timer(1/60, self.rotate)
        self.camera.azimuth = self.camera.azimuth + 1
        self.timer.start()


    def move_agent_at_traj_index(self, index):
        self.delete_agent()
        index = int(index)
        print(index)
        input('...')
        smoothed_traj = deepcopy(self.trajs[self.index][:, :3])
        smoothed_traj[:, 0] = self.normalize(np.asarray(smoothed_traj[:, 0]),
                                     self.config['DEFAULT']['rmin_x'],
                                     self.config['DEFAULT']['rmax_x'],
                                     self.config['DEFAULT']['tmin_x'],
                                     self.config['DEFAULT']['tmax_x'])
        smoothed_traj[:, 1] = self.normalize(np.asarray(smoothed_traj[:, 1]),
                                     self.config['DEFAULT']['rmin_y'],
                                     self.config['DEFAULT']['rmax_y'],
                                     self.config['DEFAULT']['tmin_y'],
                                     self.config['DEFAULT']['tmax_y'])
        smoothed_traj[:, 2] = self.normalize(np.asarray(smoothed_traj[:, 2]),
                                     self.config['DEFAULT']['rmin_z'],
                                     self.config['DEFAULT']['rmax_z'],
                                     self.config['DEFAULT']['tmin_z'],
                                     self.config['DEFAULT']['tmax_z'])

        if self.smooth_window > 0:
            smoothed_traj[:, 0] = self.savitzky_golay(smoothed_traj[:, 0], self.smooth_window, 3)
            smoothed_traj[:, 1] = self.savitzky_golay(smoothed_traj[:, 1], self.smooth_window, 3)
            smoothed_traj[:, 2] = self.savitzky_golay(smoothed_traj[:, 2], self.smooth_window, 3)

        print(smoothed_traj[index, :3])

        self.create_agent(smoothed_traj[index, :3])

    def move_agent(self):

        self.agent_timer = threading.Timer(1/60, self.move_agent)
        to_move = self.trajs[self.index][self.animation_index, :3]
        to_move = np.asarray(to_move)
        to_move[0] = self.normalize(to_move[0],
                                     self.config['DEFAULT']['rmin_x'],
                                     self.config['DEFAULT']['rmax_x'],
                                     self.config['DEFAULT']['tmin_x'],
                                     self.config['DEFAULT']['tmax_x'])
        to_move[1] = self.normalize(to_move[1],
                                     self.config['DEFAULT']['rmin_y'],
                                     self.config['DEFAULT']['rmax_y'],
                                     self.config['DEFAULT']['tmin_y'],
                                     self.config['DEFAULT']['tmax_y'])
        to_move[2] = self.normalize(to_move[2],
                                     self.config['DEFAULT']['rmin_z'],
                                     self.config['DEFAULT']['rmax_z'],
                                     self.config['DEFAULT']['tmin_z'],
                                     self.config['DEFAULT']['tmax_z'])
        to_move = to_move.astype(int)
        current_tr = self.agent.transform.translate
        if np.linalg.norm(to_move - current_tr[:3]) < 0.05 or np.linalg.norm(self.movement_vector) < 0.05:
            self.animation_index += 1
            if self.animation_index == len(self.trajs[self.index]):
                self.agent_timer.cancel()
                self.agent_timer = None
                self.delete_agent()
                return
            to_move = self.trajs[self.index][self.animation_index, :3]
            to_move = np.asarray(to_move)
            to_move[0] = self.normalize(to_move[0],
                                        self.config['DEFAULT']['rmin_x'],
                                        self.config['DEFAULT']['rmax_x'],
                                        self.config['DEFAULT']['tmin_x'],
                                        self.config['DEFAULT']['tmax_x'])
            to_move[1] = self.normalize(to_move[1],
                                        self.config['DEFAULT']['rmin_y'],
                                        self.config['DEFAULT']['rmax_y'],
                                        self.config['DEFAULT']['tmin_y'],
                                        self.config['DEFAULT']['tmax_y'])
            to_move[2] = self.normalize(to_move[2],
                                        self.config['DEFAULT']['rmin_z'],
                                        self.config['DEFAULT']['rmax_z'],
                                        self.config['DEFAULT']['tmin_z'],
                                        self.config['DEFAULT']['tmax_z'])
            to_move = to_move.astype(int)
            self.movement_vector = to_move - current_tr[:3]
            self.movement_vector = self.movement_vector / self.agent_speed
        current_tr[0] += self.movement_vector[0]
        current_tr[1] += self.movement_vector[1]
        current_tr[2] += self.movement_vector[2]

        self.agent.transform.translate = current_tr
        self.agent_timer.start()


    def extrapolate_trajectories(self, motivation, trajectories, actions):

        if motivation is not None:

            # Filler the state
            # TODO: I do this because the state that I saved is only the points AND inventory, not the complete state
            # TODO: it is probably better to save everything
            filler = np.zeros((int(self.config['MODEL']['filler'])))
            traj_to_observe = []
            episodes_to_observe = []
            acts_to_observe = []

            desired_point_y = float(self.config['GOAL']['area_y'])
            goal_area_x = float(self.config['GOAL']['area_x'])
            goal_area_z = float(self.config['GOAL']['area_z'])
            goal_area_height = float(self.config['GOAL']['area_height'])
            goal_area_width = float(self.config['GOAL']['area_width'])
            threshold = float(self.config['GOAL']['area_threshold'])

            # Save the motivation rewards and the imitation rewards
            mean_moti_rews = []
            mean_moti_rews_dict = dict()

            sum_moti_rews = []
            sum_moti_rews_dict = dict()

            sum_il_rews = []
            moti_rews = []

            step_moti_rews = []
            step_il_rews = []

            rmin_x = float(self.config['DEFAULT']['rmin_x'])
            rmax_x = float(self.config['DEFAULT']['rmax_x'])
            tmin_x = float(self.config['DEFAULT']['tmin_x'])
            tmax_x = float(self.config['DEFAULT']['tmax_x'])

            rmin_y = float(self.config['DEFAULT']['rmin_y'])
            rmax_y = float(self.config['DEFAULT']['rmax_y'])
            tmin_y = float(self.config['DEFAULT']['tmin_y'])
            tmax_y = float(self.config['DEFAULT']['tmax_y'])

            rmin_z = float(self.config['DEFAULT']['rmin_z'])
            rmax_z = float(self.config['DEFAULT']['rmax_z'])
            tmin_z = float(self.config['DEFAULT']['tmin_z'])
            tmax_z = float(self.config['DEFAULT']['tmax_z'])

            # Get only those trajectories that touch the desired points
            for keys, traj in zip(list(trajectories.keys())[:], list(trajectories.values())[:]):
                # if traj[0][-1] > 0.5:
                #     continue
                # to_observe = False
                # for point in traj:
                #     de_point = np.zeros(3)
                #     de_point[0] = ((np.asarray(point[0]) + 1) / 2) * 500
                #     de_point[1] = ((np.asarray(point[1]) + 1) / 2) * 500
                #     de_point[2] = ((np.asarray(point[2]) + 1) / 2) * 40
                #     if np.abs(de_point[0] - 461) < threshold and \
                #             np.abs(de_point[1] - 102) < threshold and \
                #             np.abs(de_point[2] - 15) < threshold:
                #         to_observe = True
                #         break
                #
                # if to_observe:
                for i, point in enumerate(traj):
                    de_point = np.zeros(3)
                    de_point[0] = self.normalize((np.asarray(point[0])),
                                                 rmin_x,
                                                 rmax_x,
                                                 tmin_x,
                                                 tmax_x)
                    de_point[1] = self.normalize((np.asarray(point[1])),
                                                 rmin_y,
                                                 rmax_y,
                                                 tmin_y,
                                                 tmax_y)
                    de_point[2] = self.normalize((np.asarray(point[2])),
                                                 rmin_z,
                                                 rmax_z,
                                                 tmin_z,
                                                 tmax_z)

                    if goal_area_x < de_point[0] < (goal_area_x + goal_area_width) and \
                            goal_area_z < de_point[1] < (goal_area_z + goal_area_height) and \
                            np.abs(de_point[2] - desired_point_y) < threshold:
                        #         if True:
                        traj_to_observe.append(np.asarray(traj))
                        episodes_to_observe.append(keys)
                        acts_to_observe.append(actions[keys])
                        break

            # Get the value of the motivation and imitation models of the extracted trajectories
            for key, traj, idx_traj in zip(episodes_to_observe, traj_to_observe, range(len(traj_to_observe))):
                states_batch = []
                actions_batch = []

                for state, action in zip(traj, actions[key]):
                    # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
                    # TODO: correct form
                    state = np.asarray(state)
                    # state[:3] = 2 * (state[:3]/40) - 1
                    state = np.concatenate([state, filler])
                    state[-2:] = state[3:5]
                    # Create the states batch to feed the models
                    state = dict(global_in=state)
                    states_batch.append(state)
                    actions_batch.append(action)
                    de_point = np.zeros(3)
                    de_point[0] = self.normalize(np.asarray(state['global_in'][0]),
                                                 rmin_x,
                                                 rmax_x,
                                                 tmin_x,
                                                 tmax_x)
                    de_point[1] = self.normalize(np.asarray(state['global_in'][1]),
                                                 rmin_y,
                                                 rmax_y,
                                                 tmin_y,
                                                 tmax_y)
                    de_point[2] = self.normalize(np.asarray(state['global_in'][2]),
                                                 rmin_z,
                                                 rmax_z,
                                                 tmin_z,
                                                 tmax_z)

                    if goal_area_x < de_point[0] < (goal_area_x + goal_area_width) and \
                            goal_area_z < de_point[1] < (goal_area_z + goal_area_height) and \
                            np.abs(de_point[2] - desired_point_y) < threshold:
                        break

                # The actions is one less than the states, so add the last state
                state = traj[-1]
                state = np.concatenate([state, filler])
                state[-2:] = state[3:5]
                state = dict(global_in=state)
                states_batch.append(state)

                # il_rew = reward_model.eval(states_batch[:-1], states_batch, actions_batch)
                il_rew = np.zeros(len(states_batch[:-1]))
                step_il_rews.extend(il_rew)
                il_rew = np.sum(il_rew)
                sum_il_rews.append(il_rew)

                moti_rew = motivation.eval(states_batch)
                moti_rews.append(moti_rew)
                step_moti_rews.extend(moti_rew)
                mean_moti_rew = np.mean(moti_rew)
                mean_moti_rews.append(mean_moti_rew)
                mean_moti_rews_dict[idx_traj] = mean_moti_rew

                sum_moti_rew = np.sum(moti_rew)
                sum_moti_rews.append(sum_moti_rew)
                sum_moti_rews_dict[idx_traj] = sum_moti_rew

                del states_batch[:]
                del actions_batch[:]

            traj_to_observe = np.asarray(traj_to_observe)
            acts_to_observe = np.asarray(acts_to_observe)

            return traj_to_observe, acts_to_observe, mean_moti_rews_dict, sum_moti_rews_dict

    def filtering_trajectory(self):

        motivation = self.motivation
        traj_to_observe = self.unfiltered_trajs
        acts_to_observe = self.unfiltered_actions
        mean_moti_rews_dict = self.mean_moti_rews_dict
        sum_moti_rews_dict = self.sum_moti_rews_dict

        mean_moti_rews = list(mean_moti_rews_dict.values())
        sum_moti_rews = list(sum_moti_rews_dict.values())

        print(" ")
        try:
            print("Max mean moti: {}".format(np.max(mean_moti_rews)))
            print("Mean mean moti: {}".format(np.mean(mean_moti_rews)))
            print("Min mean moti: {}".format(np.min(mean_moti_rews)))
            print(" ")
            print("Max sum moti: {}".format(np.max(sum_moti_rews)))
            print("Mean sum moti: {}".format(np.mean(sum_moti_rews)))
            print("Min sum moti: {}".format(np.min(sum_moti_rews)))
        except Exception as e:
            print("Nothing")
        print(" ")

        # im_heatmap = []
        filler = np.zeros((int(self.config['MODEL']['filler'])))

        moti_to_observe = []
        mean_to_observe = []
        sum_to_observe = []
        for k, v in zip(mean_moti_rews_dict.keys(), mean_moti_rews_dict.values()):
            if v > self.mean_moti_thr and sum_moti_rews_dict[k] > self.sum_moti_thr:
                moti_to_observe.append(k)
                mean_to_observe.append(v)
                sum_to_observe.append(sum_moti_rews_dict[k])

        print(self.mean_moti_thr)

        moti_to_observe = np.reshape(moti_to_observe, -1)
        mean_to_observe = np.asarray(mean_to_observe)
        sum_to_observe = np.asarray(sum_to_observe)

        idxs_to_observe = moti_to_observe
        print(moti_to_observe)
        print(idxs_to_observe)

        print("The bugged trajectories are {}".format(len(idxs_to_observe)))

        all_normalized_im_rews = []
        all_sum_fitlered_im_rews = []
        traj_to_observe = traj_to_observe[idxs_to_observe]
        acts_to_observe = acts_to_observe[idxs_to_observe]
        # Plot the trajectories
        for traj, idx in zip(traj_to_observe, idxs_to_observe):
            states_batch = []
            actions_batch = []
            # key = episodes_to_observe[idx]

            # for state, action in zip(traj, actions[key]):
            for state in traj:
                # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
                # TODO: correct form
                state = np.asarray(state)
                state = np.concatenate([state, filler])
                state[-2:] = state[3:5]

                # Create the states batch to feed the models
                state = dict(global_in=state)
                states_batch.append(state)

            im_rew = motivation.eval(states_batch)
            # im_rew = savitzky_golay(im_rew, 51, 3)
            # im_rew = (im_rew - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))
            all_normalized_im_rews.append(im_rew)
            all_sum_fitlered_im_rews.append(np.sum(im_rew))

        # if False:
        if len(all_normalized_im_rews) > self.cluster_size:
            # cluster_indices, centroids, cluster_labels = cluster(all_normalized_im_rews, clusters=self.cluster_size,
            #                           means=mean_to_observe, sums=sum_to_observe)
            cluster_indices, centroids, cluster_labels = cluster_simple(traj_to_observe, latents=all_normalized_im_rews,
                                                                        means=mean_to_observe, config=self.config,
                                                                        num_clusters=self.cluster_size)
        else:
            cluster_indices = np.arange(len(all_normalized_im_rews))
            centroids = all_sum_fitlered_im_rews
            cluster_labels = np.asarray([])

        self.unfreeze()
        self.centroids = centroids
        self.cluster_labels = cluster_labels
        self.freeze()

        # episodes_to_observe = np.asarray(episodes_to_observe)[idxs_to_observe][cluster_indices]
        all_normalized_im_rews = np.asarray(all_normalized_im_rews)

        new_sum_moti_rews_dict = dict()
        new_mean_moti_rews_dict = dict()
        for a, i in enumerate(cluster_indices):
            new_sum_moti_rews_dict[a] = mean_to_observe[i]
            new_mean_moti_rews_dict[a] = sum_to_observe[i]

        for i, traj, im_rews, actions in zip(range(len(cluster_indices)),
                                    traj_to_observe[cluster_indices],
                                    all_normalized_im_rews[cluster_indices],
                                    acts_to_observe[cluster_indices]):

            self.im_rews.append(im_rews)
            self.actions.append(actions)
            self.trajs.append(traj)

        self.unfreeze()
        all_distances = []

        for tr1 in self.trajs:
            tr1_dist = []
            for tr2 in self.trajs:
                tr1_dist.append(frechet_distance(tr1, tr2, self.config))
            all_distances.append(tr1_dist)

        all_distances = np.asarray(all_distances)
        mean_distances = np.mean(all_distances, axis=1)

        min = np.argmin(mean_distances)

        self.all_distances = all_distances[min]
        self.all_distances = np.clip(self.all_distances, np.percentile(self.all_distances, 5), np.percentile(self.all_distances, 95))

        self.freeze()

        for i, traj, im_rews in zip(range(len(cluster_indices)),
                                    traj_to_observe[cluster_indices],
                                    all_normalized_im_rews[cluster_indices]):
            self.print_3d_traj(traj, index=i)

        tree_dict = {
            'cluster_trajs': {},
            'sub_trajs': {},
            'sub_latents': {},
            'sub_actions': {}
        }
        count = 0
        for i, centroid_idx in enumerate(cluster_indices):
            traj_centroid = traj_to_observe[centroid_idx]
            tree_dict['cluster_trajs']['cl_{}'.format(i)] = traj_centroid
            tree_dict['sub_trajs']['cl_{}'.format(i)] = []
            tree_dict['sub_latents']['cl_{}'.format(i)] = []
            tree_dict['sub_actions']['cl_{}'.format(i)] = []
            if len(self.cluster_labels > 0):
                cluster_label = self.cluster_labels[centroid_idx]
                sub_traj_idxs = np.where(self.cluster_labels == cluster_label)
                for sub_traj, sub_latent, sub_actions in zip(traj_to_observe[sub_traj_idxs],
                                                             all_normalized_im_rews[sub_traj_idxs],
                                                             acts_to_observe[sub_traj_idxs]):
                    count += 1
                    tree_dict['sub_trajs']['cl_{}'.format(i)].append(sub_traj)
                    tree_dict['sub_latents']['cl_{}'.format(i)].append(sub_latent)
                    tree_dict['sub_actions']['cl_{}'.format(i)].append(sub_actions)

        self.traj_list_signal.emit(tree_dict)

        # im_heatmap = np.asarray(im_heatmap)
        # im_heatmap[:, 3] = (im_heatmap[:, 3] - np.min(im_heatmap[:, 3]))/(np.max(im_heatmap[:, 3]) - np.min(im_heatmap[:, 3]))
        # self.plot_3d_map(im_heatmap, color_index=3, colors_gradient=[(0, 0, 255), (0, 255, 0)], percentile=100,
        #                  set_map=False)

    def load_precomputed_models(self, model_name, folder='arrays'):
        try:
            with open('{}/{}/{}_buffer.pickle'.format(folder, model_name, model_name), 'rb') as f:
                buffer = pickle.load(f)
            with open('{}/{}/{}_buffer_alpha.pickle'.format(folder, model_name, model_name), 'rb') as f:
                buffer_alpha = pickle.load(f)
            with open('{}/{}/{}_worldmodel.npy'.format(folder, model_name, model_name), 'rb') as f:
                world_model = np.load(f, allow_pickle=True)
            with open('{}/{}/{}_stats.pickle'.format(folder, model_name, model_name), 'rb') as f:
                stats = pickle.load(f)
            return buffer, buffer_alpha, world_model, stats

        except Exception as e:
            print(e)
            return None, None, None, None

    def load_precomputed_time_models(self, model_name, folder='arrays'):
        try:
            with open('{}/{}/{}_buffer_in_time.pickle'.format(folder, model_name, model_name), 'rb') as f:
                buffer_in_time = pickle.load(f)
            with open('{}/{}/{}_buffer_is_grounded.pickle'.format(folder, model_name, model_name), 'rb') as f:
                buffer_is_grounded = pickle.load(f)
            return buffer_in_time, buffer_is_grounded
        except Exception as e:
            print(e)
            return None, None

    def load_demonstrations(self, model_name, folder='arrays'):
        try:
            with open('{}/{}/{}.pkl'.format(folder, model_name, model_name), 'rb') as f:
                raw_demonstrations = pickle.load(f)

            demonstrations = []
            for i, ep_len in enumerate(raw_demonstrations['episode_len']):
                new_dem = []
                start = np.sum(raw_demonstrations['episode_len'][:i])
                for s in range(ep_len):
                    new_dem.append(raw_demonstrations['obs'][start + s]['global_in'])
                new_dem = np.asarray(new_dem)
                demonstrations.append(new_dem)
            demonstrations = np.asarray(demonstrations)
            return demonstrations
        except Exception as e:
            return None

    def save_precomputed_models(self, model_name, buffer, buffer_alpha, world_model, stats,
                                folder='arrays'):
        with open('{}/{}/{}_buffer.pickle'.format(folder, model_name, model_name), 'wb') as f:
            pickle.dump(buffer, f)
        with open('{}/{}/{}_buffer_alpha.pickle'.format(folder, model_name, model_name), 'wb') as f:
            pickle.dump(buffer_alpha, f)
        with open('{}/{}/{}_worldmodel.npy'.format(folder, model_name, model_name), 'wb') as f:
            np.save(f, world_model)
        with open('{}/{}/{}_stats.pickle'.format(folder, model_name, model_name), 'wb') as f:
            pickle.dump(stats, f)

    def save_precomputed_time_models(self, model_name, buffer_in_time, is_grounded, folder='arrays'):
        with open('{}/{}/{}_buffer_in_time.pickle'.format(folder, model_name, model_name), 'wb') as f:
            pickle.dump(buffer_in_time, f)
        with open('{}/{}/{}_buffer_is_grounded.pickle'.format(folder, model_name, model_name), 'wb') as f:
            pickle.dump(is_grounded, f)

    def load_unfiltered_trajs(self, model_name, folder='arrays'):
        try:
            with open('{}/{}/{}_unf_trajs.npy'.format(folder, model_name, model_name), 'rb') as f:
                unfiltered_trajs = np.load(f, allow_pickle=True)
            with open('{}/{}/{}_unf_actions.npy'.format(folder, model_name, model_name), 'rb') as f:
                unfiltered_actions = np.load(f, allow_pickle=True)
            with open('{}/{}/{}_moti.pickle'.format(folder, model_name, model_name), 'rb') as f:
                moti = pickle.load(f)

            return unfiltered_trajs, unfiltered_actions, moti

        except Exception as e:
            print(e)
            return None, None, None

    def save_unfiltered_trajs(self, model_name, trajs, actions, mean_moti, sum_moti, folder='arrays'):
        with open('{}/{}/{}_unf_trajs.npy'.format(folder, model_name, model_name), 'wb') as f:
            np.save(f, trajs)
        with open('{}/{}/{}_unf_actions.npy'.format(folder, model_name, model_name), 'wb') as f:
            np.save(f, actions)
        with open('{}/{}/{}_moti.pickle'.format(folder, model_name, model_name), 'wb') as f:
            moti = dict(mean=mean_moti, sum=sum_moti)
            pickle.dump(moti, f)

    def load_data_from_disk(self, model_name):
        trajectories = dict()
        actions = dict()
        for filename in os.listdir("arrays/{}/".format(model_name)):
            if 'trajectories' in filename:
                with open("arrays/{}/{}".format(model_name, filename), 'r') as f:
                    trajectories.update(json.load(f))
            elif 'actions' in filename:
                with open("arrays/{}/{}".format(model_name, filename), 'r') as f:
                    actions.update(json.load(f))
        trajectories = {int(k): v for k, v in trajectories.items()}
        trajectories = collections.OrderedDict(sorted(trajectories.items()))
        actions = {int(k): v for k, v in actions.items()}
        actions = collections.OrderedDict(sorted(actions.items()))

        return trajectories, actions

    def load_config(self, model_name):
        if os.path.exists('arrays/{}/{}.ini'.format(model_name, model_name)):
            self.config.read('arrays/{}/{}.ini'.format(model_name, model_name))
        else:
            # Create config
            self.config['DEFAULT'] = {
                'rmin_x': -1,
                'rmax_x': 1,
                'tmin_x': -1,
                'tmax_x': 1,
                'rmin_y': -1,
                'rmax_y': 1,
                'tmin_y': -1,
                'tmax_y': 1,
                'rmin_z': -1,
                'rmax_z': 1,
                'tmin_z': -1,
                'tmax_z': 1,
            }

            self.config['ACTIONS'] = {}
            for i in range(20):
                self.config['ACTIONS']["Action {}".format(i)] = "Action {}".format(i)

            self.config['GOAL'] = {
                'area_x': 0,
                'area_z': 0,
                'area_width': 0,
                'area_height': 0,
                'area_y': 0,
                'area_threshold': 0,
            }

            self.config['FILTERING'] = {
                'mean_moti_thr': 0.04,
                'sum_moti_thr': 16,
                "cluster_size": 20
            }

            self.config['MODEL'] = {
                'architecture': 'architecture',
                'filler': 0
            }

            with open('arrays/{}/{}.ini'.format(model_name, model_name), 'w') as configfile:
                self.config.write(configfile)

    def load_data(self, model_name):

        self.load_config(model_name)
        self.mean_moti_thr = float(self.config['FILTERING']['mean_moti_thr'])
        self.cluster_size = int(self.config['FILTERING']['cluster_size'])
        self.sum_moti_thr = float(self.config['FILTERING']['sum_moti_thr'])
        view.camera.center = (((float(self.config['DEFAULT']['tmax_x']) - float(self.config['DEFAULT']['tmin_x'])) / 2) + float(self.config['DEFAULT']['tmin_x']),
                              ((float(self.config['DEFAULT']['tmax_y']) - float(self.config['DEFAULT']['tmin_y'])) / 2) + float(self.config['DEFAULT']['tmin_y']),
                              60)

        self.remove_maps()
        motivation = self.load_motivation(model_name)

        buffer, buffer_alpha, world_model, stats = self.load_precomputed_models(model_name)
        pos_buffers_in_time, pos_buffer_is_grounded = self.load_precomputed_time_models(model_name)
        unfiltered_trajs, unfiltered_actions, unfiltered_moti = self.load_unfiltered_trajs(model_name)

        trajectories = None
        actions = None

        if buffer is None or world_model is None:

            trajectories, actions = self.load_data_from_disk(model_name)

            buffer, buffer_alpha, world_model, pos_buffers_in_time, pos_buffer_is_grounded \
                = self.trajectories_to_pos_buffer(trajectories, rnd=motivation)
            world_model = np.asarray(world_model)
            stats = dict(episodes=len(trajectories))

            self.save_precomputed_models(model_name, buffer, buffer_alpha, world_model, stats)
            self.save_precomputed_time_models(model_name, pos_buffers_in_time, pos_buffer_is_grounded)

            unfiltered_trajs = None
            unfiltered_moti = None

        self.unfreeze()
        self.world_model = world_model
        self.buffer_alpha = buffer_alpha
        self.pos_buffers_in_time = pos_buffers_in_time
        self.pos_buffer_is_grounded = pos_buffer_is_grounded
        self.model_name = model_name
        self.freeze()

        self.heatmap_in_time_discr = int(stats['episodes'] / len(pos_buffers_in_time))
        self.in_time_signal.emit(len(pos_buffers_in_time))
        self.plot_3d_map(world_model)
        self.heatmap_in_time_given_tm(0)

        if self.trajectory_visualizer:

            if unfiltered_trajs is None:

                if trajectories is None:
                    trajectories, actions = self.load_data_from_disk(model_name)

                unfiltered_trajs, unfiltered_actions, mean_moti_rews_dict, sum_moti_rews_dict = \
                    self.extrapolate_trajectories(motivation, trajectories, actions)

                self.save_unfiltered_trajs(model_name, unfiltered_trajs, unfiltered_actions,
                                           mean_moti_rews_dict, sum_moti_rews_dict)

            if unfiltered_moti is not None:
                mean_moti_rews_dict, sum_moti_rews_dict = unfiltered_moti.values()

            self.unfreeze()
            self.motivation = motivation
            self.unfiltered_trajs = unfiltered_trajs
            self.unfiltered_actions = unfiltered_actions
            self.mean_moti_rews_dict = mean_moti_rews_dict
            self.sum_moti_rews_dict = sum_moti_rews_dict
            self.freeze()
            self.filtering_mean_signal.emit(self.mean_moti_thr)
            self.cluster_size_signal.emit(self.cluster_size)
            self.filtering_trajectory()

            # Print demonstrations
            self.demonstrations = self.load_demonstrations(model_name)
            if self.demonstrations is not None:
                for d in self.demonstrations:
                    self.print_3d_dem(d)


        self.change_text(model_name, len(list(buffer.keys())), stats['episodes'])
        self.stop_loading()
        self.load_ending_signal.emit(True)

    def load_motivation(self, model_name):
        arch = importlib.import_module('architectures.{}'.format(self.config['MODEL']['architecture']))
        graph = tf.compat.v1.Graph()
        motivation = None
        reward_model = None
        try:
            # Load motivation model
            with graph.as_default():
                tf.compat.v1.disable_eager_execution()
                motivation_sess = tf.compat.v1.Session(graph=graph)
                motivation = RND(motivation_sess, input_spec=arch.input_spec,
                                 network_spec_predictor=arch.network_spec_rnd_predictor,
                                 network_spec_target=arch.network_spec_rnd_target, obs_normalization=False,
                                 obs_to_state=arch.obs_to_state_rnd, motivation_weight=1)
                init = tf.compat.v1.global_variables_initializer()
                motivation_sess.run(init)
                motivation.load_model(name=model_name, folder='saved')
        except Exception as e:
            reward_model = None
            motivation = None
            print(e)
        return motivation

    # TODO: make a generic method for print a 3D traj
    def print_3d_dem(self, dem, with_denorm=True):
        if with_denorm:
            dem[:, 0] = self.normalize(np.asarray(dem[:, 0]),
                                                 self.config['DEFAULT']['rmin_x'],
                                                 self.config['DEFAULT']['rmax_x'],
                                                 self.config['DEFAULT']['tmin_x'],
                                                 self.config['DEFAULT']['tmax_x'])
            dem[:, 1] = self.normalize(np.asarray(dem[:, 1]),
                                                 self.config['DEFAULT']['rmin_y'],
                                                 self.config['DEFAULT']['rmax_y'],
                                                 self.config['DEFAULT']['tmin_y'],
                                                 self.config['DEFAULT']['tmax_y'])
            dem[:, 2] = self.normalize(np.asarray(dem[:, 2]),
                                                 self.config['DEFAULT']['rmin_z'],
                                                 self.config['DEFAULT']['rmax_z'],
                                                 self.config['DEFAULT']['tmin_z'],
                                                 self.config['DEFAULT']['tmax_z'])

        Tube3D = scene.visuals.create_visual_node(visuals.TubeVisual)

        noise = np.random.uniform(1e-5, 1e-7, (len(dem), 3))

        p1 = Tube3D(parent=view.scene, points=dem[:, :3] + noise, color=self.default_colors, radius=0.5)
        p1.shading_filter.enabled = False
        p1.visible = False
        self.dem_visuals.append(p1)

    def show_demonstrations(self):
        if self.dem_visuals == None or len(self.dem_visuals) == 0:
            return

        if self.dem_visuals[0].visible:
            for d in self.dem_visuals:
                d.visible = False
        else:
            for d in self.dem_visuals:
                d.visible = True

    def print_3d_traj(self, traj, index=None, with_denorm=True, max=None, min=None, tmp_line=False, im_rews=None,
                      color=None):
        """
        Method that will plot the trajectory
        """
        ep_trajectory = np.asarray(deepcopy(traj))
        smooth_window = self.smooth_window

        if im_rews is None:
            if self.im_rews[index] is not None:
                im_rews = self.im_rews[index]

        if with_denorm:
            ep_trajectory[:, 0] = self.normalize(np.asarray(ep_trajectory[:, 0]),
                                         self.config['DEFAULT']['rmin_x'],
                                         self.config['DEFAULT']['rmax_x'],
                                         self.config['DEFAULT']['tmin_x'],
                                         self.config['DEFAULT']['tmax_x'])
            ep_trajectory[:, 1] = self.normalize(np.asarray(ep_trajectory[:, 1]),
                                         self.config['DEFAULT']['rmin_y'],
                                         self.config['DEFAULT']['rmax_y'],
                                         self.config['DEFAULT']['tmin_y'],
                                         self.config['DEFAULT']['tmax_y'])
            ep_trajectory[:, 2] = self.normalize(np.asarray(ep_trajectory[:, 2]),
                                         self.config['DEFAULT']['rmin_z'],
                                         self.config['DEFAULT']['rmax_z'],
                                         self.config['DEFAULT']['tmin_z'],
                                         self.config['DEFAULT']['tmax_z'])

        noise = np.random.uniform(1e-5, 1e-7, (len(traj), 3))
        smoothed_traj = deepcopy(ep_trajectory)

        if smooth_window > 5:
            smoothed_traj[:, 0] = self.savitzky_golay(smoothed_traj[:, 0], smooth_window, 3)
            smoothed_traj[:, 1] = self.savitzky_golay(smoothed_traj[:, 1], smooth_window, 3)
            smoothed_traj[:, 2] = self.savitzky_golay(smoothed_traj[:, 2], smooth_window, 3)
            noise = 0

        if color is None:
            default_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255)]

            cluster_zero = np.argmin(self.all_distances)
            all_distances = np.asarray(self.all_distances)
            max_distance = np.max(all_distances)
            min_distance = np.min(all_distances)
            dist_from_cluster_zero = frechet_distance(traj, self.trajs[cluster_zero], self.config)
            dist_from_cluster_zero = np.clip(dist_from_cluster_zero, min_distance, max_distance)
            color = self.convert_to_rgb(min_distance, max_distance, dist_from_cluster_zero, default_colors)

        if im_rews is not None:
            im_rews = deepcopy(im_rews)
            im_rews = self.savitzky_golay(im_rews, 51, 5)
            im_rews = np.clip(im_rews, np.percentile(im_rews, 25), np.max(im_rews))
            max_im_rews = np.max(im_rews)
            mean_im_rews = np.mean(im_rews)
            gradient = []
            vertex_gradient = []
            for i in im_rews:
                if i > mean_im_rews:
                    traj_color = self.convert_to_rgb(mean_im_rews, max_im_rews, i,
                                                     colors=[np.asarray(color)[:3] * 255, (255, 0, 0)])
                    gradient.append(traj_color)
                    for i in range(8):
                        vertex_gradient.append(traj_color)
                else:
                    gradient.append(color)
                    for i in range(8):
                        vertex_gradient.append(color)
            colors = gradient
        else:
            colors = color

        if self.current_color_mode == 'default':
            colors = self.default_colors
        elif self.current_color_mode == 'gradient':
            colors = gradient
        elif self.current_color_mode == 'none':
            colors = color

        Tube3D = scene.visuals.create_visual_node(visuals.TubeVisual)

        p1 = Tube3D(parent=view.scene, points=smoothed_traj[:, :3] + noise, color=colors, radius=0.5)
        p1.shading_filter.enabled = False
        if tmp_line:
            if self.tmp_line is not None:
                self.tmp_line.visible = False

            p1.visible = True
            self.tmp_line = p1
            self.tmp_traj = traj
            self.tmp_im_rew = im_rews
        else:
            self.traj_visuals.append(p1)
            self.colors.append(color)
            self.gradients.append(vertex_gradient)

        return p1

    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        try:
            window_size = np.abs(int(window_size))
            order = np.abs(int(order))
        except ValueError as msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

class MplCanvas(FigureCanvasQTAgg):

    hover_signal = pyqtSignal(int)

    def __init__(self, canvas):
        self.fig = plt.figure()
        self.axes = plt.subplot()
        self.canvas = canvas
        self.hover_line = None
        # plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.clf()
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def show_data(self, x, y):
        self.axes.clear()
        self.plot(x, y)
        self.axes.set_xlim(0, 501)
        # fig = plt.gcf()
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover_plot)
        if self.hover_line is not None:
            self.hover_line.remove()
        self.hover_line = None
        self.draw()

    def plot(self, x, y):
        raise NotImplementedError("Please Implement this method")

    def hover_plot(self, event):
        x = event.xdata
        if x is not None:
            self.hover_signal.emit(x)

    def catch_signal(self, x):
        self.canvas.move_agent_at_traj_index(x)
        self.draw_hover_line(x)

    def draw_hover_line(self, x):
        if self.hover_line is not None:
            self.hover_line.remove()
            del self.hover_line
            self.hover_line = None

        self.hover_line = self.axes.axvline(x)
        self.draw()

    def clear_plot(self):
        self.axes.clear()
        self.draw()

class CuriosityPlot(MplCanvas):
    def __init__(self, canvas):
        MplCanvas.__init__(self, canvas)

    def plot(self, x, y):
        self.axes.plot(x, y)

class ActionPlot(MplCanvas):
    def __init__(self, canvas):
        self.cmap = cm.get_cmap('tab10')
        MplCanvas.__init__(self, canvas)

    def plot(self, x, y):
        num_actions = np.max(y) + 1
        colors = []
        for p in y:
            colors.append(self.cmap(p))

        self.axes.scatter(x, y, c=colors, s=10, marker='s')
        self.axes.set_ylim(-1, num_actions)
        # self.axes.plot(x, y)

class LegendPlot(MplCanvas):
    def __init__(self, canvas):
        self.cmap = cm.get_cmap('tab10')
        MplCanvas.__init__(self, canvas)

    def plot(self, x, y):
        num_actions = np.max(x) + 1
        tick_colors = [self.cmap(p)[:3] for p in range(num_actions)]
        self.axes.scatter(np.ones(num_actions) * 100, np.arange(num_actions), c=tick_colors, s=10, marker='s')

        y_ticks = list(self.canvas.config['ACTIONS'].values())[:num_actions]
        # y_ticks = ['Action_{}'.format(i) for i in range(num_actions)]

        self.axes.set_yticks(np.arange(num_actions))
        self.axes.set_yticklabels(y_ticks)

        self.axes.tick_params(axis='y', direction='in', labelsize=5, pad=-40)
        for ticklabel, tickcolor in zip(self.axes.get_yticklabels(), tick_colors):
            ticklabel.set_color(tickcolor)

        self.axes.set_ylim(-1, num_actions)

    def clear_plot(self):
        self.axes.clear()
        y_ticks = ['' for i in range(20)]

        self.axes.set_yticks(np.arange(20))
        self.axes.set_yticklabels(y_ticks)
        self.draw()

class WorldModelApplication(QDialog):
    def __init__(self, canvas, parent=None):
        super(WorldModelApplication, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        self.canvas = canvas

        self.modelNameCombo = QComboBox()

        areas = []
        for file in os.listdir('arrays'):
            d = os.path.join('arrays', file)
            if os.path.isdir(d):
                areas.append(file)
        areas.sort()
        self.modelNameCombo.addItems([""] + areas)
        self.last_model_name = ""

        styleLabel = QLabel("&Model Name:")
        styleLabel.setBuddy(self.modelNameCombo)

        self.modelNameCombo.activated[str].connect(self.name_combo_changed)

        self.heatmapCheck = QCheckBox("&heatmap")
        self.heatmapCheck.setChecked(True)
        self.canvas.heatmap_signal.connect(lambda b: self.set_state_checkbox(b))
        self.heatmapCheck.toggled.connect(canvas.change_map)

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(self.modelNameCombo)
        topLayout.addStretch(1)
        topLayout.addWidget(self.heatmapCheck)

        midLeftLayout = QGridLayout()

        trajsLayout = QVBoxLayout()
        self.timeLabelText = "&Heatmap in time: {} - {}"
        self.timeLabel = QLabel()
        self.timeSlider = QSlider(Qt.Horizontal)
        self.timeLabel.setBuddy(self.timeSlider)
        self.timeSlider.setMaximum(100)
        self.timeSlider.setValue(0)
        self.timeLabel.setText(self.timeLabelText.format(0, self.timeSlider.value() * self.canvas.heatmap_in_time_discr))
        self.timeSlider.setMinimumSize(200, 0)

        self.doubleTimeSlider = QRangeSlider(Qt.Horizontal)
        self.doubleTimeSlider.setMinimum(0)
        self.doubleTimeSlider.setMaximum(100)
        # self.doubleTimeSlider.setMaximum(100)
        self.doubleTimeSlider.setValue([0, 100])
        self.doubleTimeSlider.setMinimumSize(200, 0)

        self.doubleTimeSlider.sliderReleased.connect(self.time_slider_released)
        self.doubleTimeSlider.valueChanged.connect(self.time_slider_changed)

        self.alphaCombo = QComboBox()
        self.alphaLabel = QLabel('&alpha:')
        self.alphaLabel.setBuddy(self.alphaCombo)
        self.alphaCombo.addItems(['all', '0.0', '0.3', '0.5', '0.7', '0.8', '0.9', '1.0'])
        self.alphaCombo.activated[str].connect(self.alpha_name_changed)

        self.smoothSlider = QSlider(Qt.Horizontal)
        self.smoothLabel = QLabel("&Smooth:")
        self.smoothLabel.setBuddy(self.smoothSlider)
        self.smoothSlider.setMaximum(101)
        self.smoothSlider.setMinimum(0)
        self.smoothSlider.setTickInterval(2)
        self.smoothSlider.setSingleStep(2)
        self.smoothSlider.sliderReleased.connect(self.smooth_changed)

        self.trajTreeView = QTreeView()
        self.trajTreeViewLabel = QLabel('&Trajectories:')
        self.trajTreeViewLabel.setBuddy(self.trajTreeView)
        self.trajTreeView.setHeaderHidden(True)
        self.trajTreeModel = QStandardItemModel()
        self.treeRootNode = self.trajTreeModel.invisibleRootItem()
        self.trajTreeView.setModel(self.trajTreeModel)
        self.trajTreeView.expandAll()
        self.trajTreeView.setMinimumSize(0, 50)
        # self.trajTreeView.clicked.connect(self.tree_view_clicked)
        self.trajTreeView.selectionModel().selectionChanged.connect(self.tree_view_selected)
        self.canvas.traj_list_signal.connect(self.populate_tree_view)
        self.canvas.cluster_selected_signal.connect(self.tree_view_set_index)

        self.demTrajView = QCheckBox('&Demonstrations')
        demBox = QHBoxLayout()
        demBox.addWidget(self.demTrajView)

        self.demTrajView.toggled.connect(lambda x: self.canvas.show_demonstrations())

        demBox.addStretch(1)

        trajsLayout.addWidget(self.timeLabel)
        # trajsLayout.addWidget(self.timeSlider)
        trajsLayout.addWidget(self.doubleTimeSlider)
        trajsLayout.addWidget(self.alphaLabel)
        trajsLayout.addWidget(self.alphaCombo)
        trajsLayout.addWidget(self.smoothLabel)
        trajsLayout.addWidget(self.smoothSlider)
        trajsLayout.addWidget(self.trajTreeViewLabel)
        trajsLayout.addWidget(self.trajTreeView)
        trajsLayout.addLayout(demBox)

        trajsLayout.addStretch(1)
        trajsLayout.setContentsMargins(20, 20, 20, 20)
        self.timeSlider.valueChanged.connect(self.time_slider_changed)
        self.canvas.in_time_signal.connect(lambda x: {
            self.timeSlider.blockSignals(True),
            self.timeSlider.setMaximum(x),
            self.doubleTimeSlider.setMaximum(x),
            self.timeSlider.setValue(x),
            self.timeLabel.setText(self.timeLabelText.format(0, x * self.canvas.heatmap_in_time_discr)),
            self.timeSlider.blockSignals(False)})

        controlLayout = QVBoxLayout()
        controlLayout.addStretch(1)

        self.filteringMean = QSlider(Qt.Horizontal)
        self.filteringMean.setMinimumSize(200, 0)
        self.filteringMeanLabelText = '&Mean IM: {}'
        self.filteringMeanLabel = QLabel(self.filteringMeanLabelText.format(self.filteringMean.value()))
        self.filteringMeanLabel.setBuddy(self.filteringMean)

        self.clusterSize = QSlider(Qt.Horizontal)
        self.clusterSize.setMinimumSize(200, 0)
        self.clusterSize.setMaximum(30)
        self.clusterSizeText = '&Clusters: {}'
        self.clusterSizeLabel = QLabel(self.clusterSizeText.format(self.clusterSize.value()))
        self.clusterSizeLabel.setBuddy(self.clusterSize)

        self.filteringButton = QPushButton('&Filter')
        self.filteringButton.pressed.connect(self.change_thr_filtering)

        controlLayout.addWidget(self.filteringMeanLabel)
        controlLayout.addWidget(self.filteringMean)
        controlLayout.addWidget(self.clusterSizeLabel)
        controlLayout.addWidget(self.clusterSize)
        controlLayout.addWidget(self.filteringButton)
        controlLayout.setContentsMargins(20, 20, 20, 20)

        self.filteringMean.valueChanged.connect(lambda x: self.filteringMeanLabel.setText(
            self.filteringMeanLabelText.format(
                self.normalize_value(x, np.max(list(self.canvas.mean_moti_rews_dict.values()))))))

        self.clusterSize.valueChanged.connect(lambda x: self.clusterSizeLabel.setText(
            self.clusterSizeText.format(x)))

        try:
            canvas.filtering_mean_signal.connect(lambda x: {
                self.filteringMean.setValue(self.de_normalize(x, np.max(list(self.canvas.mean_moti_rews_dict.values()))))
            })
        except:
            canvas.filtering_mean_signal.connect(lambda x: {
                self.filteringMean.setValue(
                    self.de_normalize(x, 1))
            })

        canvas.cluster_size_signal.connect(lambda x: self.clusterSize.setValue(x))

        midLeftLayout.addLayout(trajsLayout, 0, 0)
        midLeftLayout.addLayout(controlLayout, 1, 0)

        midLayout = QHBoxLayout()
        midLayout.addWidget(canvas.native)
        midLayout.addLayout(midLeftLayout)

        bottomLayout = QVBoxLayout()

        curiosityLayout = QHBoxLayout()
        curiosityLayout.setSpacing(0)


        self.curiosityPlotWidget = CuriosityPlot(canvas)
        self.curiosityPlotWidget.setMinimumSize(0, 80)
        self.curiosityPlotWidget.setMaximumSize(100000, 300)

        self.legendCuriosityPlotWidget = LegendPlot(canvas)
        self.legendCuriosityPlotWidget.setMinimumSize(70, 80)
        self.legendCuriosityPlotWidget.setMaximumSize(70, 300)

        actionLayout = QHBoxLayout()
        actionLayout.setSpacing(0)

        self.actionPlotWidget = ActionPlot(canvas)
        self.actionPlotWidget.setMinimumSize(0, 80)
        self.actionPlotWidget.setMaximumSize(100000, 300)

        self.legendActionPlotWidget = LegendPlot(canvas)
        self.legendActionPlotWidget.setMinimumSize(70, 80)
        self.legendActionPlotWidget.setMaximumSize(70, 300)

        self.curiosityPlotWidget.hover_signal.connect(lambda x: {
            self.actionPlotWidget.catch_signal(x),
            self.curiosityPlotWidget.catch_signal(x)
        })
        self.actionPlotWidget.hover_signal.connect(lambda x: {
            self.actionPlotWidget.catch_signal(x),
            self.curiosityPlotWidget.catch_signal(x)
        })

        self.canvas.plot_traj_signal.connect(self.plot_traj_data)

        actionLayout.addWidget(self.legendActionPlotWidget)
        actionLayout.addWidget(self.actionPlotWidget)

        curiosityLayout.addWidget(self.legendCuriosityPlotWidget)
        curiosityLayout.addWidget(self.curiosityPlotWidget)
        bottomLayout.addLayout(curiosityLayout)
        bottomLayout.addLayout(actionLayout)
        # bottomLayout.addStretch(1)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(midLayout)
        mainLayout.addLayout(bottomLayout)

        self.load_thread = None
        self.setLayout(mainLayout)

    class MyThread(QThread):
        finished = pyqtSignal()

        def __init__(self, function):
            self.function = function
            super(WorldModelApplication.MyThread, self).__init__()

        def run(self):
            self.function()
            self.finished.emit()

    def set_state_checkbox(self, b):
        self.heatmapCheck.blockSignals(True)
        self.heatmapCheck.setChecked(b)
        self.heatmapCheck.blockSignals(False)

    def smooth_changed(self):
        value = self.smoothSlider.value()
        if value % 2 == 0:
            value -= 1
        self.canvas.smooth_window = value
        self.canvas.remove_lines(only_visuals=True)
        for i, traj in enumerate(self.canvas.trajs):
            p = self.canvas.print_3d_traj(traj, index=i, with_denorm=True)
            if self.canvas.index != i and self.canvas.index != -1 and self.canvas.index != len(self.canvas.trajs):
                p.visible = False

        if self.canvas.tmp_line is not None:
            self.canvas.print_3d_traj(self.canvas.tmp_traj, index=None, with_denorm=True,
                                          im_rews=self.canvas.tmp_im_rew)


    def tree_view_clicked(self, index):
        parent = index.parent().row()
        if parent < 0:
            self.canvas.plot_one_traj(index.row())
            self.canvas.index = index.row()
        else:
            self.canvas.blockSignals(True)
            self.trajTreeView.blockSignals(True)
            self.canvas.plot_one_traj(index.parent().row())
            self.canvas.index = index.parent().row()
            self.canvas.print_3d_traj(self.trajTreeModel.itemFromIndex(index).data()[0], tmp_line=True,
                                      im_rews=self.trajTreeModel.itemFromIndex(index).data()[1],
                                      color=self.canvas.colors[index.parent().row()])
            self.canvas.blockSignals(False)
            self.canvas.plot_traj_data(self.trajTreeModel.itemFromIndex(index).data()[1], self.trajTreeModel.itemFromIndex(index).data()[2])
            self.trajTreeView.blockSignals(False)

    def tree_view_selected(self, item):
        if len(item.indexes()) > 0:
            index = item.indexes()[0]
            self.tree_view_clicked(index)

    def tree_view_set_index(self, index):
        if index == -1:
            self.trajTreeView.selectionModel().clearSelection()
        else:
            self.trajTreeView.setCurrentIndex(self.trajTreeModel.index(index, 0))

    def populate_tree_view(self, treeDict):

        for i, traj in enumerate(treeDict['sub_trajs'].keys()):
            cl_item = QStandardItem('Cluster {}'.format(i))
            cl_item.setEditable(False)
            for j, sub_traj, sub_latent, sub_actions in zip(range(len(treeDict['sub_trajs'][traj])),
                                                            treeDict['sub_trajs'][traj], treeDict['sub_latents'][traj],
                                                            treeDict['sub_actions'][traj]):
                sub_item = QStandardItem('Traj {}'.format(j))
                sub_item.setData([sub_traj, sub_latent, sub_actions])
                sub_item.setEditable(False)
                cl_item.appendRow(sub_item)
            self.treeRootNode.appendRow(cl_item)

    def name_combo_changed(self, model_name):
        if model_name == "" or self.last_model_name == model_name:
            return

        self.load_thread = WorldModelApplication.MyThread(function=lambda : {
            self.canvas.load_data(model_name),
            self.legendActionPlotWidget.clear_plot(),
            self.actionPlotWidget.clear_plot(),
            self.curiosityPlotWidget.clear_plot()
        })
        self.last_model_name = model_name
        self.canvas.start_loading()
        self.load_thread.start()
        self.load_thread.finished.connect(self.enable_inputs)
        self.disable_inputs()

    def plot_traj_data(self, x):
        if len(x) == 0:
            self.curiosityPlotWidget.clear_plot()
            self.actionPlotWidget.clear_plot()
            self.legendActionPlotWidget.clear_plot()
        else:
            curiosity = x[0]
            actions = x[1]
            self.curiosityPlotWidget.show_data(np.arange(len(curiosity)), curiosity)
            self.actionPlotWidget.show_data(np.arange(len(actions)), actions)
            self.legendActionPlotWidget.show_data(actions, None)

    def alpha_name_changed(self, value):
        if value == 'all':
            value = 0
            self.timeSlider.setEnabled(True)
            self.doubleTimeSlider.setEnabled(True)
            self.timeSlider.blockSignals(True)
            self.doubleTimeSlider.blockSignals(True)

            self.doubleTimeSlider.setValue([self.doubleTimeSlider.value()[0], self.doubleTimeSlider.maximum()])
            # tm = self.doubleTimeSlider.value()[0]
            # tn = self.doubleTimeSlider.maximum()
            self.canvas.plot_3D_alpha_map(self.canvas.world_model, value)
            # self.canvas.show_heatmap_in_time(np.clip(tn - tm, 0, self.timeSlider.maximum()))
            #
            # tn = int(tn * self.canvas.heatmap_in_time_discr)
            # tm = int(tm * self.canvas.heatmap_in_time_discr)
            # self.timeLabel.setText(self.timeLabelText.format(tm, tn))
            self.time_slider_released()

            self.timeSlider.blockSignals(False)
            self.doubleTimeSlider.blockSignals(False)
        else:
            self.timeSlider.setEnabled(False)
            self.doubleTimeSlider.setEnabled(False)
            value = int(float(value) * 10) + 1
            self.canvas.plot_3D_alpha_map(self.canvas.world_model, value)

    def _time_slider_changed(self, value):
        if value != self.timeSlider.maximum():
            self.heatmapCheck.setEnabled(False)
        else:
            self.heatmapCheck.setEnabled(True)
        self.canvas.show_heatmap_in_time(np.clip(value, 0, self.timeSlider.maximum()))
        value = value * self.canvas.heatmap_in_time_discr
        self.timeLabel.setText(self.timeLabelText.format(value))

    def time_slider_changed(self):
        if self.canvas.heatmap is None:
            return
        values = self.doubleTimeSlider.value()
        tm = values[0]
        tn = values[1]

        if tn != self.timeSlider.maximum() or tm != 0:
            self.heatmapCheck.setEnabled(False)
        else:
            self.heatmapCheck.setEnabled(True)

        if tm != self.canvas.tm:
            return

        self.canvas.show_heatmap_in_time(np.clip(tn - tm, 0, self.timeSlider.maximum()))
        tn = tn * self.canvas.heatmap_in_time_discr
        tm = tm * self.canvas.heatmap_in_time_discr
        self.timeLabel.setText(self.timeLabelText.format(tm, tn))

    def time_slider_released(self):
        if self.canvas.heatmap is None:
            return
        values = self.doubleTimeSlider.value()
        tm = values[0]
        tn = values[1]
        if tm != self.canvas.tm:
            self.canvas.heatmap_in_time_given_tm(tm)
            self.canvas.tm = tm
        self.canvas.show_heatmap_in_time(np.clip(tn - tm, 0, self.timeSlider.maximum()))
        tn = tn * self.canvas.heatmap_in_time_discr
        tm = tm * self.canvas.heatmap_in_time_discr
        self.timeLabel.setText(self.timeLabelText.format(tm, tn))

    def disable_inputs(self):
        self.modelNameCombo.setEnabled(False)
        self.heatmapCheck.setEnabled(False)
        self.clusterSize.setEnabled(False)
        self.filteringMean.setEnabled(False)
        self.filteringButton.setEnabled(False)
        self.timeSlider.setEnabled(False)
        self.alphaCombo.setEnabled(False)
        self.doubleTimeSlider.setEnabled(False)
        self.smoothSlider.setEnabled(False)
        self.clear_tree_view()
        self.demTrajView.setEnabled(False)

    def enable_inputs(self):
        self.modelNameCombo.setEnabled(True)
        self.heatmapCheck.setEnabled(True)
        self.clusterSize.setEnabled(True)
        self.filteringMean.setEnabled(True)
        self.filteringButton.setEnabled(True)
        self.timeSlider.setEnabled(True)
        self.alphaCombo.setEnabled(True)
        self.doubleTimeSlider.setEnabled(True)
        self.smoothSlider.setEnabled(True)
        self.demTrajView.setEnabled(True)

    def de_normalize(self, value, max, min=0.01):
        return int(((value - 0.01) / (max - min)) * (100 - 0) + 0)

    def normalize_value(self, value, max, min=0.01):
       return round(((value - 0) / (100 - 0)) * (max - min) + min, 3)

    def clear_tree_view(self):
        self.trajTreeModel.clear()
        self.treeRootNode = self.trajTreeModel.invisibleRootItem()

    def change_thr_filtering(self):
        # The value of the mean threshold in percentage
        max = np.max(list(self.canvas.mean_moti_rews_dict.values()))
        min = 0.01
        value = self.filteringMean.value()
        value = ((value - 0) / (100 - 0)) * (max - min) + min
        self.clear_tree_view()
        canvas.mean_moti_thr = value
        canvas.cluster_size = self.clusterSize.value()
        canvas.remove_lines()
        canvas.filtering_trajectory()

if __name__ == '__main__':
    if sys.flags.interactive != 1:
        import matplotlib
        matplotlib.use('TKAgg', force=True)

        if plt.get_backend() == 'Qt5Agg':
            from matplotlib.backends.qt_compat import QtWidgets

            qApp = QtWidgets.QApplication(sys.argv)
            plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()

        # build canvas
        canvas = WorlModelCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.distance = 500
        view.camera.translate_speed = 100

        canvas.set_camera(view.camera)
        canvas.set_view(view)

        label_grid = canvas.central_widget.add_grid(margin=0)
        loading_grid = canvas.central_widget.add_grid(margin=0)
        label_grid.spacing = 0

        label = scene.Label("", color='white', anchor_x='left',
                            anchor_y='bottom', font_size=8)
        label.width_max = 20
        label.height_max = 20
        label_grid.add_widget(label, row=0, col=0)
        canvas.set_label(label)

        loading_label = scene.Label("Loading...", color='white', font_size=8)
        loading_label.visible = False
        loading_grid.add_widget(loading_label, row=0, col=0)

        canvas.set_loading(loading_label)

        # Build application and pass it the canvas just created
        app = QApplication(sys.argv)
        gallery = WorldModelApplication(canvas)
        gallery.show()
        gallery.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowType_Mask)
        gallery.showFullScreen()
        sys.exit(app.exec_())