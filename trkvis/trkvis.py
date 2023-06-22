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

class ViewerOptions(QtWidgets.QDialog):
    """
    Dialog box which controls viewer options
    """

    def __init__(self, parent, view):
        super(ViewerOptions, self).__init__(parent)
        self.setWindowTitle("View Options")
        self._view = view

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        vbox.addWidget(QtWidgets.QLabel('<font size="5">View Options</font>'))

        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)

        grid.addWidget(QtWidgets.QLabel("Background colour"), 0, 0)

        self._bgcol_label = QtWidgets.QLabel()
        self._bgcol_label.setAutoFillBackground(True)
        self._bgcol_label.setFixedSize(50, 20)
        palette = self._bgcol_label.palette()
        palette.setColor(QtGui.QPalette.Background, QtGui.QColor(*self._view.bgcolor))
        self._bgcol_label.setPalette(palette)
        grid.addWidget(self._bgcol_label, 0, 1)
        
        self._bgcol_picker = QtWidgets.QPushButton("Choose")
        self._bgcol_picker.clicked.connect(self._choose_bg_color)
        grid.addWidget(self._bgcol_picker, 0, 2)

        grid.addWidget(QtWidgets.QLabel("Ambient light"), 1, 0)
        self._ambient_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._ambient_slider.valueChanged.connect(self._ambient_slider_changed)
        grid.addWidget(self._ambient_slider, 1, 1)
        self._ambient_edit = QtWidgets.QLineEdit()
        grid.addWidget(self._ambient_edit, 1, 2)
        self._ambient_slider.setValue(view.ambient * 100)

        grid.addWidget(QtWidgets.QLabel("Diffuse light"), 2, 0)
        self._diffuse_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._diffuse_slider.valueChanged.connect(self._diffuse_slider_changed)
        grid.addWidget(self._diffuse_slider, 2, 1)
        self._diffuse_edit = QtWidgets.QLineEdit()
        grid.addWidget(self._diffuse_edit, 2, 2)
        self._diffuse_slider.setValue(view.diffuse * 100)

        grid.addWidget(QtWidgets.QLabel("Specular light"), 3, 0)
        self._specular_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._specular_slider.valueChanged.connect(self._specular_slider_changed)
        grid.addWidget(self._specular_slider, 3, 1)
        self._specular_edit = QtWidgets.QLineEdit()
        grid.addWidget(self._specular_edit, 3, 2)
        self._specular_slider.setValue(view.specular * 100)

        grid.addWidget(QtWidgets.QLabel("Shininess"), 4, 0)
        self._shininess_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._shininess_slider.valueChanged.connect(self._shininess_slider_changed)
        grid.addWidget(self._shininess_slider, 4, 1)
        self._shininess_edit = QtWidgets.QLineEdit()
        grid.addWidget(self._shininess_edit, 4, 2)
        self._shininess_slider.setValue(view.shininess)

        grid.addWidget(QtWidgets.QLabel("Rotation behaviour"), 5, 0)
        self._rot_combo = QtWidgets.QComboBox()
        self._rot_combo.addItem("Update light continuously")
        self._rot_combo.addItem("Update light on mouse release")
        self._rot_combo.setCurrentIndex(0 if view.continuous_light_update else 1)
        self._rot_combo.currentIndexChanged.connect(self._rot_combo_changed)
        grid.addWidget(self._rot_combo, 5, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Projection"), 6, 0)
        self._ortho_combo = QtWidgets.QComboBox()
        self._ortho_combo.addItem("3D Perspective")
        self._ortho_combo.addItem("2D no perspective - background always behind")
        self._ortho_combo.setCurrentIndex(1 if view.ortho else 0)
        self._ortho_combo.currentIndexChanged.connect(self._ortho_combo_changed)
        grid.addWidget(self._ortho_combo, 6, 1, 1, 2)

        vbox.addStretch(1)

    def _choose_bg_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            palette = self._bgcol_label.palette()
            palette.setColor(QtGui.QPalette.Background, color)
            self._bgcol_label.setPalette(palette)
            print(color.getRgb())
            self._view.bgcolor = [float(v)/255 for v in color.getRgb()[:3]]

    def _ambient_slider_changed(self):
        value = float(self._ambient_slider.value()) / 99
        self._ambient_edit.setText("%.2f" % value)
        self._view.ambient = value

    def _diffuse_slider_changed(self):
        value = float(self._diffuse_slider.value()) / 99
        self._diffuse_edit.setText("%.2f" % value)
        self._view.diffuse = value

    def _specular_slider_changed(self):
        value = float(self._specular_slider.value()) / 99
        self._specular_edit.setText("%.2f" % value)
        self._view.specular = value

    def _shininess_slider_changed(self):
        value = int(float(self._shininess_slider.value())*100/99)
        self._shininess_edit.setText(str(value))
        self._view.shininess = value

    def _rot_combo_changed(self):
        idx = self._rot_combo.currentIndex()
        self._view.continuous_light_update = (idx == 0)

    def _ortho_combo_changed(self):
        idx = self._ortho_combo.currentIndex()
        self._view.ortho = (idx == 1)

class View(scene.SceneCanvas):

    CAM_PARAMS = [
        [(90, 0, 0), (-90, 0, 0)],
        [(0, 0, 0), (180, 0, 0)],
        [(0, 90, 0), (180, -90, 0)],
    ]

    def __init__(self, size=(800, 600), **kwargs):
        """
        Viewer for background volume / tractography
        """
        scene.SceneCanvas.__init__(self, show=False, size=size, **kwargs)
        self.unfreeze()

        self._zaxis = 2
        self._view = self.central_widget.add_view()
        self._scene = self._view.scene
        self._pos = [0, 0, 0]
        self._invert = False
        self._bgvis = None
        self._tractograms = {}
        self._labels = []
        self._texture_format = kwargs.get("texture_format", "auto")
        self._ortho = False
        self._bgcolor = [0, 0, 0]

        self._view.camera = scene.TurntableCamera(fov=45)
        self._fixed_light_dir = [0, 1, 0]
        self._initial_light_dir = self._view.camera.transform.imap(self._fixed_light_dir)[:3]
        self._current_light_dir = self._fixed_light_dir
        self._continuous_light_update = kwargs.get("continuous_light_update", False)
        self._shading="smooth"
        self._ambient_light=0.2
        self._specular_light=0.8
        self._diffuse_light=0.8
        self._shininess=50

        self.set_bgvol(np.zeros((1, 1, 1), dtype=np.float32), np.identity(4), cmap="grays", clim=None)
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
        self._update_labels()
        if axis >= 0:
            cam_params = self.CAM_PARAMS[self._zaxis][1 if self._invert else 0]
            self._view.camera.azimuth = cam_params[0]
            self._view.camera.elevation = cam_params[1]
            self._view.camera.roll = cam_params[2]
            self._view.camera.fov = 0 if self._ortho else 45
            self._update_light()
        else:
            # Disable orthographic projection in 3D mode
            self._view.camera.fov = 45

    @property
    def ortho(self):
        return self._ortho

    @ortho.setter
    def ortho(self, ortho):
        self._ortho = ortho
        if ortho:
            self._view.camera.fov = 0
        else:
            self._view.camera.fov = 45
        self.zaxis = self.zaxis

    @property
    def bg_cmap(self):
        return self._bg_cmap
    
    @bg_cmap.setter
    def bg_cmap(self, cmap):
        self._bg_cmap = cmap
        if self._bgvis is not None:
            self._bgvis.cmap = cmap

    @property
    def bg_clim(self):
        return self._bg_clim

    @bg_clim.setter
    def bg_clim(self, clim):
        self._bg_clim = clim
        if self._bgvis is not None:
            self._bgvis.clim = clim

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
    
    @property
    def bgcolor(self):
        print(self._bgcolor)
        return self._bgcolor
    
    @bgcolor.setter
    def bgcolor(self, col):
        self._bgcolor = col

    @property
    def ambient(self):
        return self._ambient_light
    
    @ambient.setter
    def ambient(self, value):
        self.set_lightval("ambient", value)
    
    @property
    def specular(self):
        return self._specular_light
    
    @specular.setter
    def specular(self, value):
        self.set_lightval("specular", value)

    @property
    def diffuse(self):
        return self._diffuse_light
    
    @diffuse.setter
    def diffuse(self, value):
        self.set_lightval("diffuse", value)

    @property
    def shininess(self):
        return self._shininess
    
    @shininess.setter
    def shininess(self, value):
        value = int(value)
        self._shininess = value
        self.set_prop(["shading_filter", "shininess"], value)
        self.update()

    @property
    def continuous_light_update(self):
        return self._continuous_light_update
    
    @continuous_light_update.setter
    def continuous_light_update(self, value):
        value = bool(value)
        self._continuous_light_update = value

    def set_lightval(self, name, value):
        value = float(value)
        if value > 1 or value < 0:
            raise ValueError("Light must be between 0 and 1")
        self.set_prop(["shading_filter", f"{name}_light"], value)
        #setattr(self._shading_filter, f"{name}_light", value)
        self.update()

    @zpos.setter
    def zpos(self, zpos):
        data_zaxis = self._ras2data[self.zaxis]
        self._pos[data_zaxis] = min(self.shape[data_zaxis]-1, max(0, int(zpos)))
        self._update_bgvol()

    def _process_mouse_event(self, event):
        if event.type == "mouse_wheel":
            self.zpos = max(0, self.zpos + event.delta[1])
        else:
            if self._ortho and self._zaxis >= 0 and event.button == 1:
                # Don't allow rotation in 2D ortho mode
                return
            update_light = (
                event.type == "mouse_release" or
                (event.type == "mouse_move" and event.button == 1 and self.continuous_light_update)
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
        #self._shading_filter.light_dir = self._current_light_dir
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

    def set_bgvol(self, data, affine, cmap="grays", clim=None):
        self._bgvol = data.astype(np.float32)
        self._set_affine(affine)
        self._bg_clim = clim
        self._bg_cmap = cmap
        self._pos = [s // 2 for s in self.shape]

        world_min = np.dot(self._v2w, [0, 0, 0, 1])[:3]
        world_max = np.dot(self._v2w, np.array(self.shape + [1]))[:3]
        world_origin = self._v2w[:, 3][:3]
        self._extent = [(min(wmin, wmax), max(wmin, wmax)) for wmin, wmax in zip(world_min, world_max)]
        print("World space extent: ", world_min, world_max, self._extent)
        self._view.camera.set_range(*self._extent)
        self._view.camera.centre = world_origin
        self._update_bgvol()
        self._update_labels()

    def _update_bgvol(self):
        if self._bgvis is not None:
            self._bgvis.parent = None

        if self.zaxis < 0:
            self._bgvis = scene.visuals.Volume(
                self._bgvol.T,
                cmap=self._bg_cmap, clim=self._bg_clim,
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
                cmap=self._bg_cmap, clim=self._bg_clim,
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
            if self._ortho and self._zaxis >= 0:
                trans = [0, 0, 0]
            else:
                trans = [0, 0, self.zpos]
            t = transforms.translate(trans)
            v2ras = np.dot(np.dot(t, img2ras), self._v2w.T)
            #print("Slice->world transform:\n", v2ras)
            self._bgvis.transform = scene.transforms.MatrixTransform(v2ras)

    def _update_labels(self):
        labels = ["LR", "PA", "IS"]
        for l in self._labels:
            l.parent = None
        data_size = max([e[1] - e[0] for e in self._extent])
        for idx, label in enumerate(labels):
            if idx == self.zaxis:
                continue
            pos = [0, 0, 0]
            pos[idx] = self._extent[idx][0] - 20
            self._labels.append(scene.visuals.Text(label[0], pos=pos, parent=self._scene, color="white", font_size=20 * (1 if self._ortho and self._zaxis >= 0 else data_size)))
            pos[idx] = self._extent[idx][1] + 20
            self._labels.append(scene.visuals.Text(label[1], pos=pos, parent=self._scene, color="white", font_size=20 * (1 if self._ortho and self._zaxis >= 0 else data_size)))

    def set_prop(self, props, value):
        for t in self._tractograms.values():
            for vis in t:
                try:
                    obj = vis
                    for prop in props[:-1]:
                        obj = getattr(obj, prop)
                    setattr(obj, props[-1], value)
                except AttributeError:
                    print(obj, props, value)
                    continue
    
    def _get_connections(self, streamlines):
        connections = []
        for sldata in streamlines:
            connect = np.ones(sldata.shape[0], dtype=bool)
            connect[-1] = False
            connections.append(connect)
        return np.concatenate(streamlines, axis=0), np.concatenate(connections)

    def _create_tubes(self, streamlines, vertex_colors, options):
        streamlines = streamlines[::options.get("subset", 1)]
        if vertex_colors is not None:        
            vertex_colors = np.concatenate(vertex_colors, axis=0)
            vertex_colors = np.repeat(vertex_colors, 8, axis=0)

        coords, connections = self._get_connections(streamlines)
        shading_filter = vispy.visuals.filters.mesh.ShadingFilter(
            shading=self._shading,
            light_dir=self._current_light_dir,
            ambient_light=self._ambient_light,
            specular_light=self._specular_light,
            diffuse_light=self._diffuse_light,
            shininess=self._shininess,
        )
        vis = Tube(
            coords, connect=connections, 
            radius=options.get("width", 1), 
            vertex_colors=vertex_colors,
            color=options.get("color", "red") if vertex_colors is None else None,
            shading=None,
        )
        print("setting up lighting")
        vis.attach(shading_filter)
        vis.shading_filter = shading_filter # why???
        print("DONE setting up lighting")
        vis.parent = self._scene
        return [vis]

    def _create_lines(self, streamlines, vertex_colors, options):
        streamlines = streamlines[::options.get("subset", 1)]
        if vertex_colors is not None:        
            vertex_colors = np.concatenate(vertex_colors, axis=0)

        coords, connections = self._get_connections(streamlines)
        vis = scene.visuals.Line(
            coords, connect=connections, parent=self._scene, 
            method="gl", antialias=True,
            color=options.get("color", "red") if vertex_colors is None else vertex_colors,
        ) #  FIXME width
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
            dmin, dmax = np.min([np.min(d) for d in vertex_data]), np.max([np.max(d) for d in vertex_data])
            print("drange", dmin, dmax)
            cmin, cmax = options.get("clim", (dmin, dmax))
            print("crange", cmin, cmax)
            vertex_data_norm = [(d - cmin) / (cmax-cmin) for d in vertex_data]
            return [cmap.map(d) for d in vertex_data_norm]

    def set_tractogram(self, name, streamlines, vertex_data, options, updated=()):
        """
        Add or update a tractogram view
        """
        vertex_colors = self._get_vertex_colors(vertex_data, options)

        if name not in self._tractograms or any([v in updated for v in ("style", "width", "subset")]):
            self.remove_tractogram(name)
            if options.get("style", "line") == "tube":
                visuals = self._create_tubes(streamlines, vertex_colors, options)
            else:
                visuals = self._create_lines(streamlines, vertex_colors, options)

            self._tractograms[name] = visuals
        else:
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
        #size = (2048, 2048)
        #rendertex = gloo.Texture2D(shape=size + (4,))
        #fbo = gloo.FrameBuffer(rendertex, gloo.RenderBuffer(size))
        
        #with fbo:
        #    gloo.clear(depth=True)
        #    gloo.set_viewport(0, 0, *size)
        #    self.render()
        #    self.update()
        #    img = gloo.read_pixels((0, 0, *size), True)

        #vispy.gloo.set_viewport(0, 0, 100, 100)
        #print(self)
        #print(self._scene)
        #print(self._view)
        #img = self.render(size=size, alpha=True)
        temp_canvas = View(size=(2048, 2048))
        temp_canvas.set_bgvol(self._bgvol, self._v2w)
        temp_canvas.show()
        img = _screenshot()
        vispy.io.write_png(fname, img)

class VolumeSelection(QWidget):

    def __init__(self, viewer):
        QWidget.__init__(self)
        self._viewer = viewer
        self._data_range = (0, 1)
        
        hbox = QHBoxLayout()
        self.setLayout(hbox)

        hbox.addWidget(QtWidgets.QLabel("Background volume: "))

        self._edit = QtWidgets.QLineEdit()
        hbox.addWidget(self._edit, stretch=5)

        self._button = QtWidgets.QPushButton("Choose File")
        self._button.clicked.connect(self._choose_file)
        hbox.addWidget(self._button)

        hbox.addWidget(QtWidgets.QLabel("Colour map"))
        self._cmap_combo = QtWidgets.QComboBox()
        for cmap_name in colormap.get_colormaps():
            self._cmap_combo.addItem(cmap_name)
        self._cmap_combo.setCurrentIndex(self._cmap_combo.findText("grays"))
        self._cmap_combo.currentIndexChanged.connect(self._cmap_changed)
        hbox.addWidget(self._cmap_combo, stretch=1)

        hbox.addWidget(QtWidgets.QLabel("Min"))
        self._cmap_min = QtWidgets.QLineEdit("0.0")
        self._cmap_min.setValidator(QtGui.QDoubleValidator())
        self._cmap_min.editingFinished.connect(self._clim_changed)
        hbox.addWidget(self._cmap_min, stretch=1)

        hbox.addWidget(QtWidgets.QLabel("Max"))
        self._cmap_max = QtWidgets.QLineEdit("1.0")
        self._cmap_max.setValidator(QtGui.QDoubleValidator())
        self._cmap_max.editingFinished.connect(self._clim_changed)
        hbox.addWidget(self._cmap_max, stretch=1)

        self._reset_button = QtWidgets.QPushButton("Reset")
        self._reset_button.clicked.connect(self._reset_clim)
        hbox.addWidget(self._reset_button)

    def _choose_file(self):
        fname, _filter = QtWidgets.QFileDialog.getOpenFileName()
        if fname:
            try:
                nii = nib.load(fname)
                data = nii.get_fdata()
                affine = nii.header.get_best_affine()
                self._data_range = (np.min(data), np.max(data))
                self._cmap_min.setText(str(self._data_range[0]))
                self._cmap_max.setText(str(self._data_range[1]))
                self._viewer.view.set_bgvol(
                    data, affine, 
                    cmap=self._cmap_combo.itemText(self._cmap_combo.currentIndex()),
                    clim=self._data_range,
                )
                self._edit.setText(fname)
            except:
                import traceback
                traceback.print_exc()
    
    def _cmap_changed(self):
        self._viewer.view.bg_cmap = self._cmap_combo.itemText(self._cmap_combo.currentIndex())
        
    def _clim_changed(self):
        self._viewer.view.bg_clim = (float(self._cmap_min.text()), float(self._cmap_max.text()))

    def _reset_clim(self):
        self._viewer.view.bg_clim = self._data_range

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
        self._style_combo.currentIndexChanged.connect(self._style_changed)
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
        self._color_by_combo.currentIndexChanged.connect(self._vertex_data_changed)
        hbox.addWidget(self._color_by_combo, stretch=1)

        hbox.addWidget(QtWidgets.QLabel("Colour map"))
        self._cmap_combo = QtWidgets.QComboBox()
        for cmap_name in colormap.get_colormaps():
            self._cmap_combo.addItem(cmap_name)
        self._cmap_combo.currentIndexChanged.connect(self._changed)
        hbox.addWidget(self._cmap_combo, stretch=1)
        
        hbox.addWidget(QtWidgets.QLabel("Min"))
        self._cmap_min = QtWidgets.QLineEdit("0.0")
        self._cmap_min.setValidator(QtGui.QDoubleValidator())
        self._cmap_min.editingFinished.connect(self._changed)
        hbox.addWidget(self._cmap_min, stretch=1)

        hbox.addWidget(QtWidgets.QLabel("Max"))
        self._cmap_max = QtWidgets.QLineEdit("1.0")
        self._cmap_max.setValidator(QtGui.QDoubleValidator())
        self._cmap_max.editingFinished.connect(self._changed)
        hbox.addWidget(self._cmap_max, stretch=1)

        self._clim_reset_button = QtWidgets.QPushButton("Reset")
        self._clim_reset_button.clicked.connect(self._reset_clim)
        hbox.addWidget(self._clim_reset_button)
    
        self.setLayout(hbox)
        self._changed()

    def _vertex_data_changed(self):
        self._reset_clim()

    def _style_changed(self):
        default_subset = self._default_subset(self._style_combo.itemText(self._style_combo.currentIndex()))
        self._subset_edit.setText(str(default_subset))
        self._changed()

    def _changed(self):
        style = self._style_combo.itemText(self._style_combo.currentIndex())
        enabled = style != "none"
        color_by_idx = self._color_by_combo.currentIndex()
        color_by = None if color_by_idx == 0 else self._color_by_combo.itemText(color_by_idx)
        self._options = {
            "style" : self._style_combo.itemText(self._style_combo.currentIndex()),
            "subset" : int(self._subset_edit.text())
        }
        if color_by is not None:
            self._options["color_by"] = color_by
            self._options["cmap"] = self._cmap_combo.itemText(self._cmap_combo.currentIndex())
            self._options["clim"] = (float(self._cmap_min.text()), float(self._cmap_max.text()))
        else:
            self._options["color"] = self._color_combo.itemText(self._color_combo.currentIndex())

        if style == "tube":
            self._options["width"] = float(self._width_edit.text())

        self._width_edit.setEnabled(style == "tube")
        self._subset_edit.setEnabled(enabled)
        self._color_combo.setEnabled(enabled and color_by is None)
        self._cmap_combo.setEnabled(enabled and color_by is not None)
        self._cmap_min.setEnabled(enabled and color_by is not None)
        self._cmap_max.setEnabled(enabled and color_by is not None)
        self._color_by_combo.setEnabled(enabled)
        self.sig_changed.emit(self)

    def _reset_clim(self):
        data_range = self._color_by_combo.itemData(self._color_by_combo.currentIndex())
        if data_range is not None:
            self._cmap_min.setText(str(data_range[0]))
            self._cmap_max.setText(str(data_range[1]))   
            self._changed()

    def _default_subset(self, style):
        if style == "tube":
            return max(1, int(round(float(len(self.tractogram.streamlines)) / 2500) * 10))
        else:
            return max(1, int(round(float(len(self.tractogram.streamlines)) / 25000) * 10))

    def set_tract(self, name, tractogram, options):
        try:
            self.blockSignals(True)
            self.name = name
            self.tractogram = tractogram
            self._style_combo.setCurrentIndex(self._style_combo.findText(options.get("style", "none")))
            self._width_edit.setText(str(options.get("width", 1)))
            default_subset = self._default_subset(options.get("style", "none"))
            self._subset_edit.setText(str(options.get("subset", default_subset)))
            self._color_combo.setCurrentIndex(self._color_combo.findText(options.get("color", "red")))
            self._color_by_combo.clear()
            self._color_by_combo.addItem("None")
            for data_name, data in tractogram.data_per_point.items():
                data = np.concatenate(data)
                print(f"range for {data_name} is {np.min(data)}, {np.max(data)}")
                self._color_by_combo.addItem(data_name, (np.min(data), np.max(data)))
            self._color_by_combo.setCurrentIndex(self._color_by_combo.findText(options.get("color_by", "None")))
            self._cmap_combo.setCurrentIndex(self._cmap_combo.findText(options.get("cmap", "viridis")))
            data_range = self._color_by_combo.itemData(self._color_by_combo.findText(options.get("color_by", "None")))
            print(f"range for selected vdata={data_range}")
            clim = options.get("clim", data_range)
            print(f"clim={clim}")
            if clim is not None:
                self._cmap_min.setText(str(clim[0]))
                self._cmap_max.setText(str(clim[1]))   
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
        data = self._tract_combo.itemData(idx)
        if data is not None:
            tractogram, options = data
            self._tract_view.set_tract(name, tractogram, options)

    def _tract_view_changed(self):
        new_options = self._tract_view.options
        cur_tract_idx = self._tract_combo.currentIndex()
        name = self._tract_combo.itemText(cur_tract_idx)
        data = self._tract_combo.itemData(cur_tract_idx)
        if data is not None:
            cur_tractogram, cur_options = data
            updated = [k for k, v in new_options.items() if cur_options.get(k, None) != v]
            self._tract_combo.setItemData(cur_tract_idx, (cur_tractogram, new_options))
            print(new_options, updated)
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
        btn.setToolTip("Axial slice view")
        btn.clicked.connect(self._ax_btn_clicked)
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("saggital"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Saggital slice view")
        btn.clicked.connect(self._sag_btn_clicked)
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("coronal"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Coronal slice view")
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
        btn.setToolTip("Flip to see the other side")
        btn.clicked.connect(self._flip)
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton("Save")
        btn.setIcon(get_icon("3d"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("Save")
        btn.clicked.connect(self._save)
        hbox.addWidget(btn)

        hbox.addStretch(1)

        btn = QtWidgets.QPushButton()
        btn.setIcon(get_icon("settings"))
        btn.setFixedSize(32, 32)
        btn.setIconSize(QtCore.QSize(30, 30))
        btn.setToolTip("View settings")
        btn.clicked.connect(self._settings)
        hbox.addWidget(btn)

        hbox.addStretch()
        vbox.addLayout(hbox)
        
        hbox = QHBoxLayout()
        vbox.addLayout(hbox, stretch=2)
        
        self.view = View()
        hbox.addWidget(self.view.widget, stretch=1)
        self._settings_dialog = ViewerOptions(self, self.view)

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

    def _settings(self):
        self._settings_dialog.show()

def main():
    global LOCAL_FILE_PATH
    LOCAL_FILE_PATH = os.path.dirname(__file__)

    # Handle CTRL-C 
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    win = TrackVis()
    #nii = nib.load("MNI152_T1_1mm.nii.gz")
    #data = nii.get_fdata()
    #affine = nii.header.get_best_affine()
    #win.view.set_bgvol(
    #    data, affine
    #)
    
    win.show()
    #win._save()
    
    #nii = nib.load("MNI152_T1_2mm.nii.gz")
    #win.view.set_bgvol(nii.get_fdata(), nii.affine)
    #win._save()
    sys.exit(app.exec_())
