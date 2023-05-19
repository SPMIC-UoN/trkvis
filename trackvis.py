
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

class OrthoView:

    def __init__(self, axes):

        if sorted(list(np.abs(axes))) != [0, 1, 2]:
            raise ValueError(f"Invalid axes: {axes}")
        
        self._axes = axes
        self._canvas = scene.SceneCanvas(keys='interactive', size=(600, 600))
        self._view = self._canvas.central_widget.add_view()
        self._scene = self._view.scene
        self._shape = [1, 1, 1]
        self._pos = [0, 0, 0]
        self._affine = np.identity(4)
        self._slices = tuple([slice(None)] * 3)
        self._bgvis = None
        self._tractograms = {}
        self.widget = self._canvas.native

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

    def set_bgvol(self, data, affine, clim=None, texture_format="auto"):
        self._shape = data.shape
        self._pos = [s//2 for s in self._shape[:3]]
        self._affine = affine
        
        data_range = [(0, self._shape[d]) for d in range(3)]
        self._view.camera.set_range(*data_range)
        self._view.camera.center = self._pos

        slices = [slice(None)] * 3
        for idx, axis in enumerate(self._axes):
            if idx < 2:
                if axis < 0:
                    slices[idx] = slice(0, -1, -1)
            else:
                slices[idx] = self._pos[idx]
        self._slices = tuple(slices)

        if clim is None:
            clim = (np.min(data), np.max(data))

        data_local = np.transpose(data, self._transpose)
        for dim in self._flip:
            data_local = np.flip(data_local, dim)

        if self._bgvis is not None:
            self._bgvis.parent = None

        self._bgvis = scene.visuals.Image(
            data_local[:, :, self.zpos].T,
            cmap='grays', clim=clim,
            fg_color=(0.5, 0.5, 0.5, 1), 
            texture_format=texture_format, 
            parent=self._scene
        )
        self._bgvis.transform = scene.transforms.MatrixTransform(transforms.rotate(90, (1, 0, 0)))

    def add_tractogram(self, tractogram, name, cmap="viridis", radius=0.5):
        cmap = colormap.get_colormap(cmap)
        densities = tractogram.data_per_point['densities']
        densities_norm = [(d - np.min(d)) / (np.max(d) - np.min(d)) for d in densities]
        all_vertex_colors = [cmap.map(d) for d in densities_norm]
        w2v = np.linalg.inv(self._affine)
        i = np.identity(4)
        v2s = np.identity(4)
        for idx, d in enumerate(self._transpose):
            v2s[idx, :] = i[d, :]
        v2s[2, 3] = -self.zpos
        for d in self._flip:
            v2s[:, d] = -v2s[:, d]
        w2s = np.dot(v2s, w2v)
        print("v2w\n", self._affine)
        print("v2s\n", v2s)
        print("w2v\n", w2v)
        print("w2s\n", w2s)
        print(self._transpose)
        print(self._flip)
        print(self.zpos)

        self.remove_tractogram(name)
        visuals = []
        for sldata, vertex_colors in zip(tractogram.streamlines[::10], all_vertex_colors[::10]):
            vertex_colors=np.repeat(vertex_colors, 8, axis=0)
            vis = scene.visuals.Tube(sldata, radius=radius, vertex_colors=vertex_colors, parent=self._scene, shading='smooth')
            vis.transform = scene.transforms.MatrixTransform(np.dot(w2s.T, transforms.rotate(90, (1, 0, 0))))
            vis.shading_filter.light_dir = (0, 1, 0)
            vis.shading_filter.ambient_light = (1, 1, 1, 0.5)
            visuals.append(vis)
        self._tractograms[name] = visuals

    def remove_tractogram(self, name):
        if name in self._tractograms:
            for vis in self._tractograms[name]:
                vis.parent = None
            del self._tractograms[name]

    def clear_tractograms(self):
        for t in self._tractograms:
            self.remove_tractogram(t)

    def to_png(self, fname):
        img = self._canvas.render()
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

class MultiSelectCombo(QtWidgets.QComboBox):
    """
    A combo box which allows multiple items to be selected
    """
    sig_changed = QtCore.Signal()

    def __init__(self, parent=None, **kwargs):
        super(MultiSelectCombo, self).__init__(parent)
        #self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self._changed = False
        self._all_items = []

        self.set_choices([])

        self.currentIndexChanged.connect(self._index_changed)
        self.view().pressed.connect(self._item_pressed)
        delegate = QtWidgets.QStyledItemDelegate(self.view())
        self.view().setItemDelegate(delegate)

        # Need to intercept the default resize event
        # FIXME why can't call superclass method normally?
        self.resizeEventOrig = self.resizeEvent
        self.resizeEvent = self._resized

    @property
    def selected(self):
        """
        Get the names of the selected item(s)
        """
        ret = []
        for idx in range(1, self.count()):
            item = self.model().item(idx, 0)
            if item.checkState() == QtCore.Qt.Checked:
                ret.append(self.itemData(idx))
        return ret

    @selected.setter
    def selected(self, val):
        """
        Set the selected item(s)
        """
        for name in val:
            idx = self.findData(name)
            item = self.model().item(idx, 0)
            item.setCheckState(QtCore.Qt.Checked)

    def set_choices(self, items):
        if items and items == self._all_items:
            return

        self._all_items = items
        current = self.selected
        self.blockSignals(True)
        try:
            self.clear()

            text, tooltip = self._list_text(current)
            self._add(text, tooltip)

            idx = 1
            for name in sorted(items):
                self._add(name)
                item = self.model().item(idx, 0)
                if name in current:
                    item.setCheckState(QtCore.Qt.Checked)
                else:
                    item.setCheckState(QtCore.Qt.Unchecked)
                idx += 1

            # Make sure names are visible even with drop down arrow
            width = self.minimumSizeHint().width()
            self.setMinimumWidth(width+50)
        finally:
            self.blockSignals(False)

        self.setCurrentIndex(0)

    def _item_pressed(self, idx):
        """ One of the checkboxes has been pressed """
        item = self.model().itemFromIndex(idx)
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)
        selected_items = self.selected
        text, tooltip = self._list_text(selected_items)
        self.setItemText(0, text)
        self.setItemData(0, text)
        self.setItemData(0, tooltip, QtCore.Qt.ToolTipRole)
        self.sig_changed.emit()
        self._changed = True

    def hidePopup(self):
        """
        Overridden from QtWidgets.QComboBox

        To allow multi-select, don't hide the popup when it's clicked on to
        select/deselect items, so we can check and uncheck
        them in one go. However if nothing has changed as
        a result of the click (e.g. we click outside the popup
        window), this will close the popup
        """
        if not self._changed:
            QtWidgets.QComboBox.hidePopup(self)
        self._changed = False

    def _list_text(self, selected_items):
        """:return: Text to be used when list not visible"""
        if selected_items:
            return "%i items" % len(selected_items), ",".join(selected_items)
        else:
            return "<Select items>", ""

    def _index_changed(self, _idx):
        self.setCurrentIndex(0)

    def _add(self, name, tooltip=None):
        """
        Add an item, with shortened display name and full name as data
        """
        self.addItem(self._elided_name(name), name)
        if not tooltip:
            tooltip = name
        self.setItemData(self.count()-1, tooltip, QtCore.Qt.ToolTipRole)

    def _elided_name(self, name):
        """
        Put ellipsis between truncated name to make it fit in the combo
        """
        if name.startswith("<"):
            # Special name, not a data set name
            return name
        width = self.fontMetrics().boundingRect(name).width()
        combo_width = self.geometry().width() - 70
        elided_name = name
        left_chars = len(elided_name)//2
        right_chars = left_chars-1
        while width > combo_width:
            elided_name = elided_name[:left_chars] + "..." + elided_name[-right_chars:]
            width = self.fontMetrics().boundingRect(elided_name).width()
            if width < combo_width:
                break
            elif left_chars < 2 or right_chars < 2:
                break
            if left_chars < right_chars:
                right_chars -= 1
            else:
                left_chars -= 1
        return elided_name

    def _resized(self, event):
        self.resizeEventOrig(event)
        for idx in range(self.count()):
            data_name = self.itemData(idx)
            self.setItemText(idx, self._elided_name(data_name))

class TractView:

    def __init__(self, tractogram):
        self.streamlines = tractogram.streamlines
        self.vertex_data = tractogram.vertex_data
        self._color = "red"
        self._color_by = None
        self._cmap = "viridis"
        self._visuals = []

    @property
    def vertex_data_items(self):
        return list(self.vertex_data.keys())
    
    @property
    def color(self):
        return self._color
    
    @color.setter
    def set_color(self, color):
        self._color = color

    @property
    def color_by(self):
        return self._color_by
    
    @color_by.setter
    def set_color_by(self, color_by):
        self._color_by = color_by

    @property
    def cmap(self):
        return self._cmap
    
    @cmap.setter
    def set_cmap(self, cmap):
        self._cmap = cmap
    
    def add_visual(self):
        pass

class TractSelection(QWidget):

    def __init__(self, viewer):
        QWidget.__init__(self)
        self._viewer = viewer
        self._tract_dir = ""
        self._selected_tracts = []

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel("XTRACT/Probtrackx2 output: "))
        self._edit = QtWidgets.QLineEdit()
        hbox.addWidget(self._edit)
        self._button = QtWidgets.QPushButton("Choose directory")
        self._button.clicked.connect(self._choose_file)
        hbox.addWidget(self._button)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel("Visible Tracts "))
        self._tracts_combo = MultiSelectCombo()
        self._tracts_combo.sig_changed.connect(self._tracts_changed)
        hbox.addWidget(self._tracts_combo)
        hbox.addWidget(QtWidgets.QLabel("Colour"))
        self._colour_combo = QtWidgets.QComboBox()
        #self._colour_combo.sig_changed.connect(self._colour_changed)
        hbox.addWidget(self._colour_combo)
        hbox.addWidget(QtWidgets.QLabel("Vertex data"))
        self._data_combo = QtWidgets.QComboBox()
        #self._data_combo.sig_changed.connect(self._vertex_data_changed)
        hbox.addWidget(self._data_combo)
        hbox.addWidget(QtWidgets.QLabel("Colour map"))
        self._trk_cmap_combo = QtWidgets.QComboBox()
        #self._trk_cmap_combo.sig_changed.connect(self._trk_cmap_data_changed)
        hbox.addWidget(self._data_combo)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def _get_tracts(self, xtract_dir):
        self._tract_dir = os.path.join(xtract_dir, "tracts")
        tracts = []
        for dname in os.listdir(self._tract_dir):
            if os.path.isdir(os.path.join(self._tract_dir, dname)) and os.path.exists(os.path.join(self._tract_dir, dname, "streamlines.trk")):
                tracts.append(dname)
        self._tracts_combo.set_choices(tracts)
        if tracts:
            self._tracts_combo.selected = [tracts[0]]

    def _choose_file(self):
        xtract_dir = QtWidgets.QFileDialog.getExistingDirectory()
        if xtract_dir:
            for view in self._viewer.views:
                view.clear_tractograms()
            self._get_tracts(xtract_dir)
            self._edit.setText(xtract_dir)
    
    def _tracts_changed(self):
        selected_now = self._tracts_combo.selected
        print("tracts changed: ", selected_now)
        for tract in self._selected_tracts:
            if tract not in selected_now:
                for view in self._viewer.views:
                    view.remove_tractogram(tract)
        for tract in selected_now:
            if tract not in self._selected_tracts:
                self._select_tract(tract)
        self._selected_tracts = selected_now

    def _select_tract(self, name):
        trk = nib.streamlines.load(os.path.join(self._tract_dir, name, "streamlines.trk"))
        self._vertex_data = trk.tractogram.data_per_point.keys()
        for view in self._viewer.views:
            view.add_tractogram(trk.tractogram, name)

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
