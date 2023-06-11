
"""
Simple tractography visualisation
"""
import os
import signal
import sys

import nibabel as nib
import numpy as np

from PySide2.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PySide2 import QtWidgets, QtGui, QtCore

from vispy import io, plot as vp, visuals

from vispy import app, scene
from vispy.util import transforms
from vispy.color import Color, colormap
from vispy.visuals.filters import ShadingFilter

LOCAL_FILE_PATH = ""

class TrkView:

    def __init__(self, view, tractogram, cmap="viridis", radius=0.5):
        cmap = colormap.get_colormap(cmap)
        self._radius = radius
        densities = tractogram.data_per_point['densities']
        densities_norm = [(d - np.min(d)) / (np.max(d) - np.min(d)) for d in densities]
        all_vertex_colors = [cmap.map(d) for d in densities_norm]

        for sl in range(len(tractogram.streamlines[::10])):
            sldata = tractogram.streamlines[sl]
            vertex_colors=np.repeat(all_vertex_colors[sl], 8, axis=0)
            trk_obj = scene.visuals.Tube(sldata, radius=self._radius, vertex_colors=vertex_colors, parent=view.scene)

class OrthoView(scene.SceneCanvas):

    def __init__(self, axes, **kwargs):
        """
        :param axes: Sequence of x, y, z axes for the view using RAS convention (0=R, 1=A, 2=S)"""
        scene.SceneCanvas.__init__(self, keys='interactive', size=(600, 600))
        self.unfreeze()

        #if sorted(list(np.abs(axes[:3]))) != [0, 1, 2]:
        #    raise ValueError(f"Invalid axes: {axes}")
        
        self._display2ras = axes[:3]
        self._flip_display = axes[3]
        self._view = self.central_widget.add_view()
        self._scene = self._view.scene
        self._shape = [1, 1, 1]
        self._pos = [0, 0, 0]
        self._bgvis = None
        self._tractograms = {}
        self.widget = self.native
        self._miscvis = []
        self._continuous_light_update = kwargs.get("continuous_light_update", True)
        cam = scene.ArcballCamera(center=self._pos)
        self._view.camera = cam

        self._fixed_light_dir = [-1 if  idx == 1 and self.zaxis_ras == 0 else 1 for idx in range(3)] # Total hack
        self._initial_light_dir = self._view.camera.transform.imap(self._fixed_light_dir)[:3]
        self._current_light_dir = self._fixed_light_dir
        self.set_affine(np.identity(4))

    def set_affine(self, affine):
        """
        Get the grid axes which best correspond to the RAS axes

        :return: List of four integers giving the axes indices of the R, A and S axes.
                 The fourth integer is always 3 indicating the volume axis
        """
        self._ras2display = [0, 0, 0]
        for display_axis, ras_axis in enumerate(self._display2ras):
            self._ras2display[ras_axis] = display_axis

        self._v2w = affine
        transform = self._v2w[:3, :3]
        # Sequence defining which RAS axis is identified with each data axis
        self._data2ras = [np.argmax(np.abs(transform[:, axis])) for axis in range(3)]
        print("Data->RAS axis sequence: ", self._data2ras)
        
        # Sequence defining which data axis is identified with each RAS axis
        # and which RAS axes are flipped in the data
        self._ras2data = [self._data2ras.index(axis) for axis in range(3)]
        self._flip_ras = []
        for ras_axis, data_axis in enumerate(self._ras2data):
            if transform[data_axis, ras_axis] < 0:
                self._flip_ras.append(ras_axis)
        print("RAS->data axis sequence: ", self._ras2data)
        print("RAS axes that are flipped in data: ", self._flip_ras)

        zaxis_data = self._ras2data[self.zaxis_ras]
        slices = [slice(None)] * 3
        slices[zaxis_data] = self.zpos
        print("Slicing data on axis: ", zaxis_data)

        # Identify which data axis is identified with each display axis
        # and which data axes need to be flipped
        self._display2data = [self._ras2data[ax] for ax in self._display2ras]
        self._flip_display = []
        for display_axis, data_axis in enumerate(self._display2data):
            if self._data2ras[data_axis] in self._flip_ras:
                self._flip_display.append(display_axis)
        print("Axis transposition/flip for display: ", self._display2ras, self._display2data, self._flip_display)

        w2v = np.linalg.inv(self._v2w)
        i = np.identity(4)
        v2d = np.identity(4)
        for ras_axis, display_axis in enumerate(self._ras2display):
            v2d[display_axis, :] = i[ras_axis, :]
        for d in self._flip_display:
            v2d[d, :] = -v2d[d, :]
            v2d[d, 3] = self._shape[d]-1
        self._w2d= np.dot(v2d, w2v)
        tract_transform = scene.transforms.MatrixTransform(np.dot(self._w2d.T, transforms.rotate(90, (1, 0, 0))))
        self.set_prop("transform", tract_transform)

    def showEvent(self, event):
        print('in showEvent')
        scene.SceneCanvas.showEvent(event)

    def _process_mouse_event(self, event):
        if event.type == "mouse_wheel":
            self.zpos = self.zpos + event.delta[1]
            self._update_bgvol()
        else:
            update_light = (
                event.type == "mouse_release" or
                (event.type == "mouse_move" and event.button == 1 and self._continuous_light_update)
            )
            if update_light:
                transform = self._view.camera.transform
                dir = np.concatenate((self._initial_light_dir, [0]))
                self._current_light_dir = transform.map(dir)[:3]
                self.set_prop(["shading_filter", "light_dir"], self._current_light_dir)
                self.update()

            from vispy.scene.events import SceneMouseEvent
            scene_event = SceneMouseEvent(event=event, visual=self._view)
            getattr(self._view.events, event.type)(scene_event)

    @property
    def zaxis_ras(self):
        return self._display2ras[2]

    @property
    def zaxis_data(self):
        return self._ras2data[self.zaxis_ras]

    @property
    def zpos(self):
        return self._pos[self.zaxis_data]

    @zpos.setter
    def zpos(self, zpos):
        self._pos[self.zaxis_data] = int(zpos)

    def set_bgvol(self, data, affine, clim=None, texture_format="auto"):
        self._bgvol = data
        self._shape = data.shape
        self._pos = [s//2 for s in self._shape[:3]]
        self.set_affine(affine)
        self._texture_format = texture_format
        self._clim = clim

        data_range = [(0, self._shape[d]) for d in range(3)]
        self._view.camera.set_range(*data_range)
        cam_centre = list(self._pos)
        cam_centre[1] = -cam_centre[1]
        self._view.camera.center = cam_centre
        # Left = Right for radiological perspective
        self._view.camera.flip = [True if self._ras2display[idx] == 0 else False for idx in range(3)]
        self._update_bgvol()

    def _update_bgvol(self):     
        self._data_local = np.transpose(self._bgvol, self._display2data)
        for dim in self._flip_display:
            self._data_local = np.flip(self._data_local, dim)
 
        if self._bgvis is not None:
            self._bgvis.parent = None

        for vis in self._miscvis:
            vis.parent = None
        self._miscvis = []

        self._bgvis = scene.visuals.Image(
            self._data_local[:, :, self.zpos].T,
            cmap='grays', clim=self._clim,
            fg_color=(0.5, 0.5, 0.5, 1), 
            texture_format=self._texture_format, 
            parent=self._scene
        )
        tvec = [0, 0, 0]
        tvec[self.zaxis_data] = self.zpos
        img2axis = transforms.translate((0, 0, self.zpos))
        img2axis = np.dot(img2axis, transforms.rotate(90, (1, 0, 0)))
        self._bgvis.transform = scene.transforms.MatrixTransform(img2axis)

    def set_prop(self, props, value):
        for t in self._tractograms.values():
            for vis in t:
                obj = vis
                for prop in props[:-1]:
                    obj = getattr(obj, prop)
                setattr(obj, props[-1], value)

    def set_tractogram(self, name, streamlines, vertex_data, options, updated=()):

        if name not in self._tractograms or any([v in updated for v in ("style", "width", "subset")]):
            self.remove_tractogram(name)
            streamlines = streamlines[::options.get("subset", 20)]
            visuals = []
            for idx, sldata in enumerate(streamlines):
                print(f"Vis: {idx+1}/{len(streamlines)}")
                if options.get("style", "line") == "tube":
                    vis = scene.visuals.Tube(sldata, radius=options.get("width", 1), shading='smooth')
                    vis.shading_filter.light_dir = self._current_light_dir
                    vis.shading_filter.ambient_light = 0.1
                    vis.shading_filter.specular_light = 1.0
                    vis.shading_filter.diffuse_light = 0.8
                    vis.shading_filter.specular_coefficient = 1.0
                    vis.shading_filter.diffuse_coefficient = 1.0
                    vis.shading_filter.ambient_coefficient = 1.0
                    vis.shading_filter.shininess = 50
                    vis.parent = self._scene
                else:
                    vis = scene.visuals.Line(sldata, parent=self._scene) #  FIXME width
                vis.transform = scene.transforms.MatrixTransform(np.dot(self._w2d.T, transforms.rotate(90, (1, 0, 0))))
                visuals.append(vis)
            self._tractograms[name] = visuals

        if vertex_data is not None:
            vertex_data = vertex_data[::options.get("subset", 20)]
            cmap = colormap.get_colormap(options.get("cmap", "viridis"))
            vertex_data_norm = [(d - np.min(d)) / (np.max(d) - np.min(d)) for d in vertex_data]
            vertex_colors = [cmap.map(d) for d in vertex_data_norm]

        for idx, vis in enumerate(self._tractograms[name]):
            if options.get("style", "line") == "tube":
                md = vis.mesh_data
                if vertex_data is not None:
                    vis.set_data(vertices=md.get_vertices(), faces=md.get_faces(), vertex_colors=np.repeat(vertex_colors[idx], 8, axis=0))
                else:
                    vis.set_data(vertices=md.get_vertices(), faces=md.get_faces(), color=options.get("color", "red"))
            else:
                pos = vis.pos
                if vertex_data is not None:
                    vis.set_data(pos=pos, color=vertex_colors[idx])
                else:
                    vis.set_data(pos=pos, color=options.get("color", "red"))

    def remove_tractogram(self, name):
        if name in self._tractograms:
            for vis in self._tractograms[name]:
                vis.parent = None
            del self._tractograms[name]

    def clear_tractograms(self):
        for t in self._tractograms:
            self.remove_tractogram(t)

    def to_png(self, fname):
        img = self.render()
        io.write_png(fname, img)

class VolumeView(OrthoView):

    def _update_bgvol(self):
        self._view.camera.flip = [False,] * 3
        self._data_local = self._bgvol
 
        if self._bgvis is not None:
            self._bgvis.parent = None

        self._bgvis = scene.visuals.Volume(
            self._data_local.T,
            cmap='grays', clim=self._clim,
            texture_format=self._texture_format,
            method="mip",
            parent=self._scene
        )
        img2axis = transforms.rotate(90, (1, 0, 0))
        self._bgvis.transform = scene.transforms.MatrixTransform(img2axis)

class VolumeSelection(QWidget):

    def __init__(self, viewer):
        QWidget.__init__(self)
        self._viewer = viewer

        hbox = QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel("Background volume: "))

        self._edit = QtWidgets.QLineEdit()
        hbox.addWidget(self._edit)

        self._button = QtWidgets.QPushButton("Choose File")
        self._button.clicked.connect(self._choose_file)
        hbox.addWidget(self._button)
        self.setLayout(hbox)

    def _choose_file(self):
        fname, _filter = QtWidgets.QFileDialog.getOpenFileName()
        if fname:
            try:
                nii = nib.load(fname)
                data = nii.get_fdata()
                affine = nii.header.get_best_affine()
                for view in self._viewer.views:
                    view.set_bgvol(data, affine)
                self._edit.setText(fname)
            except:
                import traceback
                traceback.print_exc()

class TractView(QWidget):

    sig_changed = QtCore.Signal(object)

    COLORS = ["red", "green", "blue", "yellow"]
    STYLES = ["none", "line", "tube"]

    def __init__(self):
        QWidget.__init__(self)
        
        self.name = None
        self.tractogram = None
        self._options = {}

        hbox = QHBoxLayout()

        hbox.addWidget(QtWidgets.QLabel("View"))
        self._style_combo = QtWidgets.QComboBox()
        for style in self.STYLES:
            self._style_combo.addItem(style)
        self._style_combo.currentIndexChanged.connect(self._changed)
        hbox.addWidget(self._style_combo, stretch=1)

        hbox.addWidget(QtWidgets.QLabel("Width"))
        self._width_edit = QtWidgets.QLineEdit()
        self._width_edit.setText("1")
        self._width_edit.setValidator(QtGui.QDoubleValidator())
        self._width_edit.editingFinished.connect(self._changed)
        hbox.addWidget(self._width_edit, stretch=1)

        hbox.addWidget(QtWidgets.QLabel("Subset"))
        self._subset_edit = QtWidgets.QLineEdit()
        self._subset_edit.setText("50")
        self._width_edit.setValidator(QtGui.QIntValidator())
        self._subset_edit.editingFinished.connect(self._changed)
        hbox.addWidget(self._subset_edit, stretch=1)

        hbox.addWidget(QtWidgets.QLabel("Colour"))
        self._color_combo = QtWidgets.QComboBox()
        for col in self.COLORS:
            self._color_combo.addItem(col)
        self._color_combo.currentIndexChanged.connect(self._changed)
        hbox.addWidget(self._color_combo, stretch=1)

        hbox.addWidget(QtWidgets.QLabel("Vertex data"))
        self._color_by_combo = QtWidgets.QComboBox()
        self._color_by_combo.addItem("None")
        self._color_by_combo.currentIndexChanged.connect(self._changed)
        hbox.addWidget(self._color_by_combo, stretch=1)

        hbox.addWidget(QtWidgets.QLabel("Colour map"))
        self._cmap_combo = QtWidgets.QComboBox()
        for cmap_name in colormap.get_colormaps():
            self._cmap_combo.addItem(cmap_name)
        self._cmap_combo.currentIndexChanged.connect(self._changed)
        hbox.addWidget(self._cmap_combo, stretch=1)
        
        self.setLayout(hbox)
        self._changed()

    def _changed(self):
        style = self._style_combo.itemText(self._style_combo.currentIndex())
        enabled = style != "none"
        color_by_idx = self._color_by_combo.currentIndex()
        color_by = None if color_by_idx == 0 else self._color_by_combo.itemText(color_by_idx)
        self._options = {
            "style" : self._style_combo.itemText(self._style_combo.currentIndex()),
            "width" : float(self._width_edit.text()),
            "subset" : int(self._subset_edit.text()),
        }
        if color_by is not None:
            self._options["color_by"] = color_by
            self._options["cmap"] = self._cmap_combo.itemText(self._cmap_combo.currentIndex())
        else:
            self._options["color"] = self._color_combo.itemText(self._color_combo.currentIndex())

        self._color_combo.setEnabled(enabled and color_by is None)
        self._cmap_combo.setEnabled(enabled and color_by is not None)
        self._color_by_combo.setEnabled(enabled)
        self.sig_changed.emit(self)

    def set_tract(self, name, tractogram, options):
        try:
            self.blockSignals(True)
            self.name = name
            self.tractogram = tractogram
            self._style_combo.setCurrentIndex(self._style_combo.findText(options.get("style", "none")))
            self._width_edit.setText(str(options.get("width", 1)))
            self._subset_edit.setText(str(options.get("subset", 50)))
            self._color_combo.setCurrentIndex(self._color_combo.findText(options.get("color", "red")))
            self._color_by_combo.clear()
            self._color_by_combo.addItem("None")
            for data_name in tractogram.data_per_point.keys():
                self._color_by_combo.addItem(data_name)
            self._color_by_combo.setCurrentIndex(self._color_by_combo.findText(options.get("color_by", "None")))
            self._cmap_combo.setCurrentIndex(self._cmap_combo.findText(options.get("cmap", "viridis")))
        finally:
            self.blockSignals(False)
        self._changed()

    @property
    def streamlines(self):
        return self.tractogram.streamlines

    @property
    def vertex_data(self):
        color_by = self._options.get("color_by", None)
        print("vertex data: ", self.name, color_by)
        return None if color_by is None else self.tractogram.data_per_point[color_by]

    @property
    def options(self):
        return self._options

class TractSelection(QWidget):

    def __init__(self, viewer):
        QWidget.__init__(self)
        self._viewer = viewer
        self._tract_dir = ""

        self._vbox = QVBoxLayout()
        self._vbox.setSpacing(1)

        hbox = QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel("XTRACT/Probtrackx2 output: "))
        self._edit = QtWidgets.QLineEdit()
        hbox.addWidget(self._edit)
        self._button = QtWidgets.QPushButton("Choose directory")
        self._button.clicked.connect(self._choose_file)
        hbox.addWidget(self._button)
        self._vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel("Tracts: "), stretch=0)
        self._tract_combo = QtWidgets.QComboBox()
        self._tract_combo.currentIndexChanged.connect(self._tract_changed)
        hbox.addWidget(self._tract_combo, stretch=0)

        self._tract_view = TractView()
        self._tract_view.sig_changed.connect(self._tract_view_changed)
        hbox.addWidget(self._tract_view, stretch=2)
        self._vbox.addLayout(hbox)

        self.setLayout(self._vbox)

    def _set_xtract_dir(self, xtract_dir):
        self._edit.setText(xtract_dir)
        self._tract_dir = os.path.join(xtract_dir, "tracts")
        self._tract_combo.clear()
        for dname in os.listdir(self._tract_dir):
            if os.path.isdir(os.path.join(self._tract_dir, dname)) and os.path.exists(os.path.join(self._tract_dir, dname, "streamlines.trk")):
                trk = nib.streamlines.load(os.path.join(self._tract_dir, dname, "streamlines.trk"))
                self._tract_combo.addItem(dname, (trk.tractogram, {}))

        self._vbox.addStretch()

    def _choose_file(self):
        xtract_dir = QtWidgets.QFileDialog.getExistingDirectory()
        if xtract_dir:
            for view in self._viewer.views:
                view.clear_tractograms()
            
            self._set_xtract_dir(xtract_dir)

    def _tract_changed(self, idx):
        name = self._tract_combo.itemText(idx)
        tractogram, options = self._tract_combo.itemData(idx)
        self._tract_view.set_tract(name, tractogram, options)

    def _tract_view_changed(self):
        new_options = self._tract_view.options
        cur_tract_idx = self._tract_combo.currentIndex()
        name = self._tract_combo.itemText(cur_tract_idx)
        cur_tractogram, cur_options = self._tract_combo.itemData(cur_tract_idx)
        updated = [k for k, v in new_options.items() if cur_options.get(k, None) != v]
        self._tract_combo.setItemData(cur_tract_idx, (cur_tractogram, new_options))

        if updated:
            for view in self._viewer.views:
                if new_options["style"] == "none":
                    view.remove_tractogram(name)
                else:
                    view.set_tractogram(name, self._tract_view.streamlines, self._tract_view.vertex_data, new_options, updated)

class DataSelection(QtWidgets.QWidget):
    def __init__(self, viewer):
        QtWidgets.QWidget.__init__(self)

        vbox = QVBoxLayout()
        vbox.addWidget(VolumeSelection(viewer))
        vbox.addWidget(TractSelection(viewer))
        self.setLayout(vbox)

class LightControl(QWidget):
    def __init__(self, viewer):
        QWidget.__init__(self)
        self.viewer = viewer

        hbox = QHBoxLayout()
        self.setLayout(hbox)

        hbox.addWidget(QtWidgets.QLabel("Ambient"))
        self._aedit = QtWidgets.QLineEdit()
        self._aedit.setText("1.0")
        self._aedit.setValidator(QtGui.QDoubleValidator())
        hbox.addWidget(self._aedit)
        self._aedit.editingFinished.connect(self._aedit_changed)
        hbox.addWidget(QtWidgets.QLabel("Diffuse"))
        self._dedit = QtWidgets.QLineEdit()
        self._dedit.setText("1.0")
        self._dedit.setValidator(QtGui.QDoubleValidator())
        hbox.addWidget(self._dedit)
        self._dedit.editingFinished.connect(self._dedit_changed)
        hbox.addWidget(QtWidgets.QLabel("Specular"))
        self._sedit = QtWidgets.QLineEdit()
        self._sedit.setText("1.0")
        self._sedit.setValidator(QtGui.QDoubleValidator())
        hbox.addWidget(self._sedit)
        self._sedit.editingFinished.connect(self._sedit_changed)
        hbox.addWidget(QtWidgets.QLabel("Shininess"))
        self._shedit = QtWidgets.QLineEdit()
        self._shedit.setText("100")
        self._shedit.setValidator(QtGui.QIntValidator())
        hbox.addWidget(self._shedit)
        self._shedit.editingFinished.connect(self._shedit_changed)

    def _aedit_changed(self):
        value = float(self._aedit.text())
        for view in self.viewer.views:
            view.set_prop(["shading_filter", "ambient_coefficient"], value)

    def _dedit_changed(self):
        value = float(self._dedit.text())
        for view in self.viewer.views:
            view.set_prop(["shading_filter", "diffuse_coefficient"], value)

    def _sedit_changed(self):
        value = float(self._sedit.text())
        for view in self.viewer.views:
            view.set_prop(["shading_filter", "specular_coefficient"], value)

    def _shedit_changed(self):
        value = int(self._shedit.text())
        for view in self.viewer.views:
            view.set_prop(["shading_filter", "shininess"], value)

def get_icon(name):
    return QtGui.QIcon(os.path.join(LOCAL_FILE_PATH, "icons", name + ".png"))

class TrackVis(QWidget):

    def __init__(self, voldata, xtract_dir):
        QWidget.__init__(self)
        vbox = QVBoxLayout()
        vbox.addWidget(DataSelection(self))
        vbox.addWidget(LightControl(self))

        hbox = QHBoxLayout()

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("axial"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Show/hide axial view")
        btn.clicked.connect(self._ax_btn_clicked)
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("saggital"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Show/hide saggital view")
        btn.clicked.connect(self._sag_btn_clicked)
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("coronal"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Show/hide coronal view")
        btn.clicked.connect(self._cor_btn_clicked)
        hbox.addWidget(btn)

        hbox.addStretch()
        vbox.addLayout(hbox)
        
        hbox = QHBoxLayout()
        vbox.addLayout(hbox, stretch=2)
        
        self.views = []
        for axis_mapping in [
            (0, 1, 2, [2]),
            (1, 2, 0, []),
            (0, 2, 1, []),
        ]:
            view = OrthoView(axis_mapping)
            hbox.addWidget(view.widget, stretch=1)
            self.views.append(view)

        view = VolumeView((0, 1, 2, [2]))
        hbox.addWidget(view.widget, stretch=1)
        self.views.append(view)

        self.setLayout(vbox)

    def _cor_btn_clicked(self):
        self.views[1].widget.setVisible(not self.views[1].widget.isVisible())
    def _sag_btn_clicked(self):
        self.views[2].widget.setVisible(not self.views[2].widget.isVisible())
    def _ax_btn_clicked(self):
        self.views[0].widget.setVisible(not self.views[0].widget.isVisible())

def main():
    global LOCAL_FILE_PATH
    LOCAL_FILE_PATH = os.path.dirname(__file__)

    # Handle CTRL-C 
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    win = TrackVis("MNI152_T1_2mm.nii.gz", "xtract_results")
    win.show()
    sys.exit(app.exec_())
