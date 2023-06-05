
"""
Simple tractography visualisation
"""
import os
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

    def __init__(self, axes):
        scene.SceneCanvas.__init__(self, keys='interactive', size=(600, 600))
        self.unfreeze()

        if sorted(list(np.abs(axes))) != [0, 1, 2]:
            raise ValueError(f"Invalid axes: {axes}")
        
        self._axes = axes
        self._view = self.central_widget.add_view()
        self._scene = self._view.scene
        self._shape = [1, 1, 1]
        self._pos = [0, 0, 0]
        self._affine = np.identity(4)
        self._slices = tuple([slice(None)] * 3)
        self._bgvis = None
        self._tractograms = {}
        self.widget = self.native

        self._transpose = []
        self._flip = []
        for idx, axis in enumerate(axes):
            self._transpose.append(np.abs(axis))
            if axis < 0:
                self._flip.append(np.abs(axis))

        cam = scene.ArcballCamera(center=self._pos)
        self._view.camera = cam

    @property
    def zpos(self):
        return self._pos[self._axes[2]]

    @zpos.setter
    def zpos(self, zpos):
        self._pos[self._axes[2]] = int(zpos)

    #def on_mouse_press(self, event):
    #    print("mouse_press", event.button, event.pos)
    #    pass

    #def on_mouse_move(self, event):
    #    print("mouse_move", event.button, event.pos)
    #    tr = self.scene.node_transform(self._bgvis)
    #    pos = tr.map(event.pos)
    #    print("bg_pos", pos)

    #def on_mouse_wheel(self, event):
    #    print("wheel", event.button, event.pos, event.delta)
    #    self.zpos = self.zpos + event.delta[1]
    #    print(self._pos)
    #    self._update_bgvol()

    def set_bgvol(self, data, affine, clim=None, texture_format="auto"):
        self._bgvol = data
        self._shape = data.shape
        self._pos = [s//2 for s in self._shape[:3]]
        self._affine = affine
        self._texture_format = texture_format
        self._clim = clim
        
        data_range = [(0, self._shape[d]) for d in range(3)]
        self._view.camera.set_range(*data_range)
        self._view.camera.center = self._pos
        self._update_bgvol()

    def _update_bgvol(self):
        slices = [slice(None)] * 3
        for idx, axis in enumerate(self._axes):
            if idx < 2:
                if axis < 0:
                    slices[idx] = slice(0, -1, -1)
            else:
                slices[idx] = self._pos[idx]
        self._slices = tuple(slices)

        data_local = np.transpose(self._bgvol, self._transpose)
        for dim in self._flip:
            data_local = np.flip(data_local, dim)

        if self._bgvis is not None:
            self._bgvis.parent = None

        self._bgvis = scene.visuals.Image(
            data_local[:, :, self.zpos].T,
            cmap='grays', clim=self._clim,
            fg_color=(0.5, 0.5, 0.5, 1), 
            texture_format=self._texture_format, 
            parent=self._scene
        )
        self._bgvis.transform = scene.transforms.MatrixTransform(transforms.rotate(90, (1, 0, 0)))

        w2v = np.linalg.inv(self._affine)
        i = np.identity(4)
        v2s = np.identity(4)
        for idx, d in enumerate(self._transpose):
            v2s[idx, :] = i[d, :]
        v2s[2, 3] = -self.zpos
        for d in self._flip:
            v2s[:, d] = -v2s[:, d]
        self._w2s = np.dot(v2s, w2v)
        print("v2w\n", self._affine)
        print("v2s\n", v2s)
        print("w2v\n", w2v)
        print("w2s\n", self._w2s)
        print(self._transpose)
        print(self._flip)
        print(self.zpos)

    def set_tractogram(self, name, streamlines, vertex_data, options, updated=()):

        if name not in self._tractograms or any([v in updated for v in ("style", "width", "subset")]):
            self.remove_tractogram(name)
            streamlines = streamlines[::options.get("subset", 20)]
            visuals = []
            for idx, sldata in enumerate(streamlines):
                print(f"Vis: {idx+1}/{len(streamlines)}")
                if options.get("style", "line") == "tube":
                    vis = scene.visuals.Tube(sldata, radius=options.get("width", 1), parent=self._scene, shading='smooth')
                    vis.shading_filter.light_dir = (1, 1, 0)
                    vis.shading_filter.ambient_light = (1, 1, 1, 0.5)
                else:
                    vis = scene.visuals.Line(sldata, parent=self._scene) #  FIXME width
                vis.transform = scene.transforms.MatrixTransform(np.dot(self._w2s.T, transforms.rotate(90, (1, 0, 0))))
                visuals.append(vis)
            self._tractograms[name] = visuals

        for idx, vis in enumerate(self._tractograms[name]):
            print(f"Props: {idx+1}/{len(self._tractograms[name])}")
            if vertex_data is not None:
                vertex_data = vertex_data[::options.get("subset", 20)]
                cmap = colormap.get_colormap(options.get("cmap", "viridis"))
                vertex_data_norm = [(d - np.min(d)) / (np.max(d) - np.min(d)) for d in vertex_data]
                vertex_colors = [cmap.map(d) for d in vertex_data_norm]

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
        print(fname)
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

    COLORS = ["red", "green", "blue"]
    STYLES = ["line", "tube"]

    def __init__(self):
        QWidget.__init__(self)
        
        self.name = None
        self.tractogram = None
        self._options = {}

        hbox = QHBoxLayout()
        self._select_cb = QtWidgets.QCheckBox()
        self._select_cb.stateChanged.connect(self._changed)
        hbox.addWidget(self._select_cb)
        self._name_edit = QtWidgets.QLineEdit()
        hbox.addWidget(self._name_edit)

        hbox.addWidget(QtWidgets.QLabel("Style"))
        self._style_combo = QtWidgets.QComboBox()
        for style in self.STYLES:
            self._style_combo.addItem(style)
        self._style_combo.currentIndexChanged.connect(self._changed)
        hbox.addWidget(self._style_combo)

        hbox.addWidget(QtWidgets.QLabel("Width"))
        self._width_spin = QtWidgets.QSpinBox()
        self._width_spin.setMinimum(1)
        self._width_spin.setMaximum(25)
        self._width_spin.setValue(1)
        self._width_spin.valueChanged.connect(self._changed)
        hbox.addWidget(self._width_spin)

        hbox.addWidget(QtWidgets.QLabel("Subset"))
        self._subset_spin = QtWidgets.QSpinBox()
        self._subset_spin.setMinimum(1)
        self._subset_spin.setMaximum(100)
        self._subset_spin.setValue(20)
        self._subset_spin.valueChanged.connect(self._changed)
        hbox.addWidget(self._subset_spin)

        hbox.addWidget(QtWidgets.QLabel("Colour"))
        self._color_combo = QtWidgets.QComboBox()
        for col in self.COLORS:
            self._color_combo.addItem(col)
        self._color_combo.currentIndexChanged.connect(self._changed)
        hbox.addWidget(self._color_combo)

        hbox.addWidget(QtWidgets.QLabel("Vertex data"))
        self._color_by_combo = QtWidgets.QComboBox()
        self._color_by_combo.addItem("None")
        self._color_by_combo.currentIndexChanged.connect(self._changed)
        hbox.addWidget(self._color_by_combo)

        hbox.addWidget(QtWidgets.QLabel("Colour map"))
        self._cmap_combo = QtWidgets.QComboBox()
        for cmap_name in colormap.get_colormaps():
            self._cmap_combo.addItem(cmap_name)
        self._cmap_combo.currentIndexChanged.connect(self._changed)
        hbox.addWidget(self._cmap_combo)
        
        self.setLayout(hbox)
        self._changed()

    def _changed(self):
        enabled = self._select_cb.isChecked()
        color_by_idx = self._color_by_combo.currentIndex()
        color_by = None if color_by_idx == 0 else self._color_by_combo.itemText(color_by_idx)
        self._options = {
            "enabled" : enabled,
            "style" : self._style_combo.itemText(self._style_combo.currentIndex()),
            "width" : self._width_spin.value(),
            "subset" : self._subset_spin.value(),
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
        self.name = name
        self.tractogram = tractogram
        self._name_edit.setText(name)
        self._select_cb.setChecked(options.get("enabled", False))
        self._style_combo.setCurrentIndex(self._style_combo.findText(options.get("style", "line")))
        self._width_spin.setValue(options.get("width", 1))
        self._subset_spin.setValue(options.get("subset", 20))
        self._color_combo.setCurrentIndex(self._color_combo.findText(options.get("color", "red")))
        self._color_by_combo.clear()
        self._color_by_combo.addItem("None")
        for data_name in tractogram.data_per_point.keys():
            self._color_by_combo.addItem(data_name)
        self._color_by_combo.setCurrentIndex(self._color_by_combo.findText(options.get("color_by", "None")))
        self._cmap_combo.setCurrentIndex(self._cmap_combo.findText(options.get("cmap", "viridis")))
        self._changed()

    @property
    def streamlines(self):
        return self.tractogram.streamlines

    @property
    def vertex_data(self):
        color_by = self._options.get("color_by", None)
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
        hbox.addWidget(QtWidgets.QLabel("Tracts: "))
        self._tract_combo = QtWidgets.QComboBox()
        self._tract_combo.currentIndexChanged.connect(self._tract_changed)
        hbox.addWidget(self._tract_combo)
        self._vbox.addLayout(hbox)

        self._tract_view = TractView()
        self._tract_view.sig_changed.connect(self._tract_view_changed)
        self._vbox.addWidget(self._tract_view)

        self.setLayout(self._vbox)

    def _set_xtract_dir(self, xtract_dir):
        self._edit.setText(xtract_dir)
        self._tract_dir = os.path.join(xtract_dir, "tracts")
        self._tract_combo.clear()
        for dname in os.listdir(self._tract_dir):
            if os.path.isdir(os.path.join(self._tract_dir, dname)) and os.path.exists(os.path.join(self._tract_dir, dname, "streamlines.trk")):
                trk = nib.streamlines.load(os.path.join(self._tract_dir, dname, "streamlines.trk"))
                print(f"Found tract {dname}")
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
        print(f"tract selected: {name}")
        tractogram, options = self._tract_combo.itemData(idx)
        self._tract_view.set_tract(name, tractogram, options)

    def _tract_view_changed(self):
        print(f"tract view changed")
        new_options = self._tract_view.options
        cur_tract_idx = self._tract_combo.currentIndex()
        name = self._tract_combo.itemText(cur_tract_idx)
        cur_tractogram, cur_options = self._tract_combo.itemData(cur_tract_idx)
        updated = [k for k, v in new_options.items() if cur_options.get(k, None) != v]
        print(new_options, cur_options)
        print(updated)
        self._tract_combo.setItemData(cur_tract_idx, (cur_tractogram, new_options))

        if updated:
            for view in self._viewer.views:
                if not new_options["enabled"]:
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

class TrackVis(QWidget):

    def __init__(self, voldata, xtract_dir):
        QWidget.__init__(self)
        vbox = QVBoxLayout()
        vbox.addWidget(DataSelection(self))

        hbox = QHBoxLayout()
        vbox.addLayout(hbox)
        
        self.views = []
        for axis_mapping in [
            (0, 1, 2),
            (1, 2, 0),
            (0, 2, 1),
        ]:
            view = OrthoView(axis_mapping)
            hbox.addWidget(view.widget)
            self.views.append(view)

        self.setLayout(vbox)

app = QApplication(sys.argv)
#scene.backends.use("Pyside2")
win = TrackVis("MNI152_T1_2mm.nii.gz", "xtract_results")

#nii_vol = nib.load("MNI152_T1_2mm.nii.gz")
#nii = nib.load("densityNorm.nii.gz")
#vol_data = nii_vol.get_fdata()[:,:,:]
#vol = scene.visuals.Volume(np.transpose(vol_data, (2, 1, 0)), parent=view.scene, method="mip")
#vol.transform = scene.transforms.MatrixTransform(nii_vol.header.get_best_affine().T)

win.show()
for idx, view in enumerate(win.views):
    view.to_png(f"view_{idx}.png")
sys.exit(app.exec_())
