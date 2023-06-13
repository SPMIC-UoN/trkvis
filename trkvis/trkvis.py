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

import vispy.io
from vispy import scene
from vispy.util import transforms
from vispy.color import Color, colormap
from vispy.visuals.filters import Alpha

from .connected_tube import ConnectedTubeVisual
Tube = scene.visuals.create_visual_node(ConnectedTubeVisual)

LOCAL_FILE_PATH = ""

class View(scene.SceneCanvas):

    CAM_PARAMS = [
        [(90, 0, 0), (-90, 0, 0)],
        [(0, 0, 0), (180, 0, 0)],
        [(0, 90, 0), (180, -90, 0)],
    ]

    def __init__(self, **kwargs):
        """
        Viewer for background volume / tractography
        """
        scene.SceneCanvas.__init__(self, show=False, size=(600, 600))
        self.unfreeze()

        self._zaxis = 2
        self._view = self.central_widget.add_view()
        self._scene = self._view.scene
        self._pos = [0, 0, 0]
        self._invert = False
        self._bgvol = np.zeros((1, 1, 1), dtype=np.float32)
        self._clim = None
        self._bgvis = None
        self._tractograms = {}
        self._texture_format = kwargs.get("texture_format", "auto")

        self._view.camera = scene.TurntableCamera()
        self._fixed_light_dir = [1, 1, 1]
        self._initial_light_dir = self._view.camera.transform.imap(self._fixed_light_dir)[:3]
        self._current_light_dir = self._fixed_light_dir
        self._continuous_light_update = kwargs.get("continuous_light_update", True)
        self._ambient = 0.1
        self._specular = 1.0
        self._diffuse = 0.8
        self._shininess = 50

        self._set_affine(np.identity(4))
        self.zaxis = 2

    @property
    def widget(self):
        return self.native

    @property
    def shape(self):
        return list(self._bgvol.shape[:3])

    @property
    def zaxis(self):
        return self._zaxis

    @zaxis.setter
    def zaxis(self, axis):
        self._zaxis = axis
        self._update_bgvol()
        if axis >= 0:
            cam_params = self.CAM_PARAMS[self._zaxis][1 if self._invert else 0]
            self._view.camera.azimuth = cam_params[0]
            self._view.camera.elevation = cam_params[1]
            self._view.camera.roll = cam_params[2]
            self._update_light()

    @property
    def invert(self):
        return self._invert
    
    @invert.setter
    def invert(self, invert):
        self._invert = invert
        self.zaxis = self._zaxis
    
    @property
    def zpos(self):
        return self._pos[self._ras2data[self.zaxis]]
    
    @zpos.setter
    def zpos(self, zpos):
        self._pos[self._ras2data[self.zaxis]] = int(zpos)

    def _process_mouse_event(self, event):
        if event.type == "mouse_wheel":
            self.zpos = max(0, self.zpos + event.delta[1])
            self._update_bgvol()
        else:
            update_light = (
                event.type == "mouse_release" or
                (event.type == "mouse_move" and event.button == 1 and self._continuous_light_update)
            )
            if update_light:
                self._update_light()
               
            from vispy.scene.events import SceneMouseEvent
            scene_event = SceneMouseEvent(event=event, visual=self._view)
            getattr(self._view.events, event.type)(scene_event)

    def _update_light(self):
        """
        Update light direction so it always seems to come from the same place
        relative to camera
        """
        transform = self._view.camera.transform
        dir = np.concatenate((self._initial_light_dir, [0]))
        self._current_light_dir = transform.map(dir)[:3]
        self.set_prop(["shading_filter", "light_dir"], self._current_light_dir)
        self.update()

    def _set_affine(self, affine):
        """
        Get the grid axes which best correspond to the RAS axes

        :return: List of four integers giving the axes indices of the R, A and S axes.
                 The fourth integer is always 3 indicating the volume axis
        """
        self._v2w = affine
        transform = self._v2w[:3, :3]
        # Sequence defining which RAS axis is identified with each data axis
        self._data2ras = [np.argmax(np.abs(transform[:, axis])) for axis in range(3)]
        #print("Data->RAS axis sequence: ", self._data2ras)
        
        # Sequence defining which data axis is identified with each RAS axis
        # and which RAS axes are flipped in the data
        self._ras2data = [self._data2ras.index(axis) for axis in range(3)]
        self._flip_ras = []
        for ras_axis, data_axis in enumerate(self._ras2data):
            if transform[data_axis, ras_axis] < 0:
                self._flip_ras.append(ras_axis)
        #print("RAS->data axis sequence: ", self._ras2data)
        #print("RAS axes that are flipped in data: ", self._flip_ras)

    def set_bgvol(self, data, affine, clim=None):
        self._bgvol = data.astype(np.float32)
        self._set_affine(affine)
        self._clim = clim
        self._pos = [s // 2 for s in self.shape]

        world_min = np.dot(self._v2w, [0, 0, 0, 1])[:3]
        world_max = np.dot(self._v2w, np.array(self.shape + [1]))[:3]
        world_origin = self._v2w[:, 3][:3]
        #print("World space extent: ", world_min, world_max)
        self._extent = list(zip(world_min, world_max))
        self._view.camera.set_range(*self._extent)
        self._view.camera.centre = world_origin
        self._update_bgvol()

    def _update_bgvol(self):
        if self._bgvis is not None:
            self._bgvis.parent = None

        if self.zaxis < 0:
            self._bgvis = scene.visuals.Volume(
                self._bgvol.T,
                cmap='grays', clim=self._clim,
                texture_format=self._texture_format,
                method="mip",
                parent=self._scene
            )
            self._bgvis.transform = scene.transforms.MatrixTransform(self._v2w.T)
            self._bgvis.attach(Alpha(0.5))
        else:
            zaxis_data = self._ras2data[self.zaxis]
            self._data_slices = [slice(None)] * 3
            slxy = []
            for data_axis in range(3):
                ras_axis = self._data2ras[data_axis]
                if ras_axis == self.zaxis:
                    self._data_slices[data_axis] = self.zpos
                else:
                    slxy.append(ras_axis)
            self._data_slices = tuple(self._data_slices)
            #print("Slicing data on axis: ", zaxis_data, self._data_slices)
            self._data_slice = self._bgvol[self._data_slices]
            
            self._bgvis = scene.visuals.Image(
                self._data_slice,
                cmap='grays', clim=self._clim,
                fg_color=(0.5, 0.5, 0.5, 1), 
                texture_format=self._texture_format, 
                parent=self._scene
            )
            # Image seems to default to XZ plane so need to transform it to right plane
            # Ought to be possible to derive this from zaxis but not found a way so far...
            base_transforms = [
                np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
                np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
                np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            ]
            img2ras = base_transforms[self.zaxis].T
            #print(img2ras.T)
            t = transforms.translate((0, 0, self.zpos))
            #print(t)
            v2ras = np.dot(np.dot(t, img2ras), self._v2w.T)
            #print("Slice->world transform:\n", v2ras)
            self._bgvis.transform = scene.transforms.MatrixTransform(v2ras)

    def set_prop(self, props, value):
        for t in self._tractograms.values():
            for vis in t:
                try:
                    obj = vis
                    for prop in props[:-1]:
                        obj = getattr(obj, prop)
                    setattr(obj, props[-1], value)
                except AttributeError:
                    continue
    
    def _get_connections(self, streamlines):
        connections = []
        for sldata in streamlines:
            connect = np.ones(sldata.shape[0], dtype=bool)
            connect[-1] = False
            connections.append(connect)
        return np.concatenate(streamlines, axis=0), np.concatenate(connections)

    def _create_tubes(self, streamlines, options):
        streamlines = streamlines[::options.get("subset", 1)]
        coords, connections = self._get_connections(streamlines)
        vis = Tube(coords, connect=connections, parent=self._scene, radius=options.get("width", 1), shading='smooth')
        vis.shading_filter.light_dir = self._current_light_dir
        vis.shading_filter.ambient_light = self._ambient
        vis.shading_filter.specular_light = self._specular
        vis.shading_filter.diffuse_light = self._diffuse
        vis.shading_filter.shininess = self._shininess
        return [vis]

    def _create_lines(self, streamlines, options):
        coords, connections = self._get_connections(streamlines)
        vis = scene.visuals.Line(coords, connect=connections, parent=self._scene, method="gl", antialias=True) #  FIXME width
        return [vis]

    def _update_tubes(self, visuals, vertex_colors, options):
        vis = visuals[0]
        md = vis.mesh_data
        if vertex_colors is not None:
            vertex_colors = np.concatenate(vertex_colors, axis=0)
            vis.set_data(vertices=md.get_vertices(), faces=md.get_faces(), vertex_colors=np.repeat(vertex_colors, 8, axis=0))
        else:
            vis.set_data(vertices=md.get_vertices(), faces=md.get_faces(), color=options.get("color", "red"))

    def _update_lines(self, visuals, vertex_colors, options):
        vis = visuals[0]
        pos = vis.pos
        if vertex_colors is not None:
            vertex_colors = np.concatenate(vertex_colors, axis=0)
            vis.set_data(pos=pos, color=vertex_colors)
        else:
            vis.set_data(pos=pos, color=options.get("color", "red"))

    def _get_vertex_colors(self, vertex_data, options):
        if vertex_data is not None:
            vertex_data = vertex_data[::options.get("subset", 1)]
            cmap = colormap.get_colormap(options.get("cmap", "viridis"))
            vertex_data_norm = [(d - np.min(d)) / (np.max(d) - np.min(d)) for d in vertex_data]
            return [cmap.map(d) for d in vertex_data_norm]

    def set_tractogram(self, name, streamlines, vertex_data, options, updated=()):
        """
        Add or update a tractogram view
        """
        if name not in self._tractograms or any([v in updated for v in ("style", "width", "subset")]):
            self.remove_tractogram(name)
            if options.get("style", "line") == "tube":
                visuals = self._create_tubes(streamlines, options)
            else:
                visuals = self._create_lines(streamlines, options)

            self._tractograms[name] = visuals

        vertex_colors = self._get_vertex_colors(vertex_data, options)
        visuals = self._tractograms[name]
        if options.get("style", "line") == "tube":
            self._update_tubes(visuals, vertex_colors, options)
        else:
            self._update_lines(visuals, vertex_colors, options)

    def remove_tractogram(self, name):
        if name in self._tractograms:
            for vis in self._tractograms[name]:
                vis.parent = None
            del self._tractograms[name]

    def clear_tractograms(self):
        for t in list(self._tractograms.keys()):
            self.remove_tractogram(t)

    def to_png(self, fname, size=None):
        # Not working yet
        from vispy.gloo.util import _screenshot
        from vispy import gloo
        #img = _screenshot()
        size = (2048, 2048)
        rendertex = gloo.Texture2D(shape=size + (4,))
        fbo = gloo.FrameBuffer(rendertex, gloo.RenderBuffer(size))
        
        with fbo:
            gloo.clear(depth=True)
            gloo.set_viewport(0, 0, *size)
            self.render()
            self.update()
            screenshot = gloo.read_pixels((0, 0, *size), True)

        #img = self.render(size=size, alpha=False)
        vispy.io.write_png(fname, screenshot)

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
                self._viewer.view.set_bgvol(data, affine)
                self._edit.setText(fname)
            except:
                import traceback
                traceback.print_exc()

class TractViewSelection(QWidget):

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
        self._subset_edit.setValidator(QtGui.QIntValidator())
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
        }
        if color_by is not None:
            self._options["color_by"] = color_by
            self._options["cmap"] = self._cmap_combo.itemText(self._cmap_combo.currentIndex())
        else:
            self._options["color"] = self._color_combo.itemText(self._color_combo.currentIndex())

        if style == "tube":
            self._options["subset"] = int(self._subset_edit.text())
            self._options["width"] = float(self._width_edit.text())

        self._subset_edit.setEnabled(style == "tube")
        self._width_edit.setEnabled(style == "tube")
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
            self._subset_edit.setText(str(options.get("subset", int(round(float(len(tractogram.streamlines)) / 100) * 10))))
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
        return None if color_by is None else self.tractogram.data_per_point[color_by]

    @property
    def options(self):
        return self._options

class TractInputSelection(QWidget):

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

        self._tract_view = TractViewSelection()
        self._tract_view.sig_changed.connect(self._tract_view_changed)
        hbox.addWidget(self._tract_view, stretch=2)
        self._vbox.addLayout(hbox)

        self.setLayout(self._vbox)

    def _get_tract_dir(self, indir):
        xtract_tract_dir = os.path.join(indir, "tracts")
        if os.path.isdir(xtract_tract_dir):
            return xtract_tract_dir
        else:
            return indir

    def _set_input_dir(self, indir):
        self._edit.setText(indir)
        self._tract_dir = self._get_tract_dir(indir)
        self._tract_combo.clear()
        for dname in os.listdir(self._tract_dir):
            if os.path.isdir(os.path.join(self._tract_dir, dname)) and os.path.exists(os.path.join(self._tract_dir, dname, "streamlines.trk")):
                trk = nib.streamlines.load(os.path.join(self._tract_dir, dname, "streamlines.trk"))
                self._tract_combo.addItem(dname, (trk.tractogram, {}))

        self._vbox.addStretch()

    def _choose_file(self):
        input_dir = QtWidgets.QFileDialog.getExistingDirectory()
        if input_dir:
            self._viewer.view.clear_tractograms()
            self._set_input_dir(input_dir)

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
            if new_options["style"] == "none":
                self._viewer.view.remove_tractogram(name)
            else:
                self._viewer.view.set_tractogram(name, self._tract_view.streamlines, self._tract_view.vertex_data, new_options, updated)

class DataSelection(QtWidgets.QWidget):
    def __init__(self, viewer):
        QtWidgets.QWidget.__init__(self)

        vbox = QVBoxLayout()
        vbox.addWidget(VolumeSelection(viewer))
        vbox.addWidget(TractInputSelection(viewer))
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
        self.viewer.view.set_prop(["shading_filter", "ambient_coefficient"], value)

    def _dedit_changed(self):
        value = float(self._dedit.text())
        self.viewer.view.set_prop(["shading_filter", "diffuse_coefficient"], value)

    def _sedit_changed(self):
        value = float(self._sedit.text())
        self.viewer.view.set_prop(["shading_filter", "specular_coefficient"], value)

    def _shedit_changed(self):
        value = int(self._shedit.text())
        self.viewer.view.set_prop(["shading_filter", "shininess"], value)

def get_icon(name):
    return QtGui.QIcon(os.path.join(LOCAL_FILE_PATH, "icons", name + ".png"))

class TrackVis(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        vbox = QVBoxLayout()
        vbox.addWidget(DataSelection(self))
        #vbox.addWidget(LightControl(self))

        hbox = QHBoxLayout()

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("axial"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Axial view")
        btn.clicked.connect(self._ax_btn_clicked)
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("saggital"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Saggital view")
        btn.clicked.connect(self._sag_btn_clicked)
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("coronal"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Coronal view")
        btn.clicked.connect(self._cor_btn_clicked)
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("3d"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("3D volume view")
        btn.clicked.connect(self._vol_btn_clicked)
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("flip"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Flip")
        btn.clicked.connect(self._flip)
        hbox.addWidget(btn)

        #btn = QtWidgets.QPushButton("Save")
        #btn.setIcon(get_icon("3d"))
        #btn.setFixedSize(32, 32)
        #btn.setIconSize(QtCore.QSize(30, 30))
        #btn.setToolTip("Save")
        #btn.clicked.connect(self._save)
        #hbox.addWidget(btn)

        hbox.addStretch()
        vbox.addLayout(hbox)
        
        hbox = QHBoxLayout()
        vbox.addLayout(hbox, stretch=2)
        
        self.view = View()
        hbox.addWidget(self.view.widget, stretch=1)

        self.setLayout(vbox)

    def _cor_btn_clicked(self):
        self.view.zaxis = 0

    def _sag_btn_clicked(self):
        self.view.zaxis = 1

    def _ax_btn_clicked(self):
        self.view.zaxis = 2

    def _vol_btn_clicked(self):
        self.view.zaxis = -1

    def _flip(self):
        self.view.invert = not self.view.invert

    def _save(self):
        self.view.to_png("trkvis.png")

def main():
    global LOCAL_FILE_PATH
    LOCAL_FILE_PATH = os.path.dirname(__file__)

    # Handle CTRL-C 
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    win = TrackVis()
    win.show()
    
    #nii = nib.load("MNI152_T1_2mm.nii.gz")
    #win.view.set_bgvol(nii.get_fdata(), nii.affine)
    #win._save()
    sys.exit(app.exec_())
