"""Non-destructive visual repair layers; no changes to scan or collision geometry."""
import hashlib
from pathlib import Path

import numpy as np
from scene_io.splat import quat_wxyz_to_mat, covariances
from viewer.contact import at_capture


def brush_indices(points, origin, direction, radius):
    """Front depth band of a ray brush, avoiding the back face of an object."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    delta = points - origin
    depth = delta @ d
    distance = np.linalg.norm(delta - depth[:, None] * d, axis=1)
    hit = (depth > 0) & (distance < radius)
    if not hit.any():
        return np.zeros(0, int)
    front = np.percentile(depth[hit], 5)
    return np.flatnonzero(hit & (depth <= front + radius))


class RepairPanel:
    def __init__(self, server, sc, world, ui, lock, nodes, contacts, add_splat, run_dir):
        self.server, self.sc, self.world, self.ui, self.lock = server, sc, world, ui, lock
        self.nodes, self.contacts, self.add_splat = nodes, contacts, add_splat
        self.path = Path(run_dir) / 'visual_repairs.npz'
        self.layers, self.undo, self.redo = {}, [], []
        self.selection = None
        self.highlight = None
        self.applied = {}
        self.overlays = {}
        self.overlay_scales = {}
        for o in sc.objects:
            for part in ('splat', 'contact'):
                a = o.get(part)
                if a is not None:
                    key = (o['id'], part)
                    self.layers[key] = dict(a=a, state=np.zeros(len(a['means']), np.int8),
                                            base=o['base'], fingerprint=hashlib.sha256(a['means'].tobytes()).hexdigest())
        gui = server.gui
        gui.add_markdown('Select an object, enable the brush, then **click its surface** to select a patch. '
                         'Apply an action after checking the yellow preview. Repairs affect visuals only.')
        self.pick = gui.add_dropdown('repair object', [f"{o['id']}" for o in sc.objects])
        self.active = gui.add_checkbox('enable repair brush', False)
        self.isolate = gui.add_checkbox('isolate repair object', False)
        self.source = gui.add_dropdown('repair layer', ['object', 'contact'])
        self.radius = gui.add_slider('brush radius (scene units)', .001, .08, .001, .015)
        actions = gui.add_button_group('apply to selection', ['erase', 'restore', 'keep with object', 'keep on table'])
        history = gui.add_button_group('repair history', ['undo', 'redo', 'clear selection'])
        disk = gui.add_button_group('repair file', ['save repairs', 'load repairs'])
        self.info = gui.add_markdown('Repairs have their own undo and save file. Collision shapes stay unchanged.')
        self.active.on_update(lambda _: self.enable())
        self.pick.on_update(lambda _: self.clear())
        self.source.on_update(lambda _: self.clear())
        actions.on_click(lambda _: self.edit(actions.value))
        history.on_click(lambda _: self.history(history.value))
        disk.on_click(lambda _: self.disk(disk.value))

    def enable(self):
        if self.active.value:
            self.ui['paused'] = True
        else:
            self.clear()

    def clear(self):
        self.selection = None
        if self.highlight is not None:
            self.highlight.remove()
            self.highlight = None

    def key(self):
        return int(self.pick.value), 'splat' if self.source.value == 'object' else 'contact'

    def click(self, event):
        with self.lock:
            self.ui['paused'] = True
            key = self.key()
            if key not in self.layers or not self.world.present(key[0]):
                self.info.content = 'This object has no available layer.'
                return
            layer = self.layers[key]
            p, q = self.world.pose(key[0]) if key[1] == 'splat' else (layer['base'], np.array([1., 0, 0, 0]))
            points = layer['a']['means'] @ quat_wxyz_to_mat(q[None])[0].T + p
            ids = brush_indices(points, event.ray_origin, event.ray_direction, self.radius.value)
            self.clear()
            self.selection = key, ids
            if len(ids):
                self.highlight = self.server.scene.add_point_cloud('/repair/selection', points=points[ids],
                    colors=np.tile(np.array([255, 210, 30], np.uint8), (len(ids), 1)), point_size=.002)
            self.info.content = f'{len(ids):,} splats selected. Front depth band only; orbit to edit another side.'

    def edit(self, action):
        with self.lock:
            if self.selection is None:
                self.info.content = 'Click a surface first.'
                return
            key, ids = self.selection
            if not len(ids):
                return
            obj = next(o for o in self.sc.objects if o['id'] == key[0])
            if action in ('keep with object', 'keep on table') and not at_capture(
                    self.world.pose(key[0]), obj['base'], self.sc.radius, obj['scale']):
                self.info.content = 'Restore the original object position and size in Inspect before reassigning splats.'
                return
            state = self.layers[key]['state']
            value = {'erase': -1, 'restore': 0, 'keep with object': 1, 'keep on table': 2}[action]
            self.undo.append((key, ids.copy(), state[ids].copy(), np.full(len(ids), value, np.int8)))
            self.undo = self.undo[-30:]
            self.redo.clear()
            state[ids] = value
            self.rebuild(key)
            self.clear()
            self.info.content = f'{action}: {len(ids):,} splats. Undo available; save repairs to keep them.'

    def rebuild(self, key):
        self.applied.pop(key, None)
        layer = self.layers[key]
        for target in (1, 2):
            name = (*key, target)
            if name in self.overlays:
                self.overlays.pop(name).remove()
            self.overlay_scales.pop(name, None)
            mask = layer['state'] == target
            if mask.any():
                self.overlays[name] = self.add_splat(self.server, '/repair/' + '-'.join(map(str, name)),
                                                   {k: v[mask] for k, v in layer['a'].items()})

    def history(self, action):
        with self.lock:
            self.clear()
            if action == 'clear selection':
                return
            src, dst = (self.undo, self.redo) if action == 'undo' else (self.redo, self.undo)
            if src:
                item = src.pop()
                key, ids, before, after = item
                self.layers[key]['state'][ids] = before if action == 'undo' else after
                dst.append(item)
                self.rebuild(key)
                self.info.content = f'Repair {action} applied.'

    def disk(self, action):
        with self.lock:
            try:
                if action == 'save repairs':
                    payload = {}
                    for key, layer in self.layers.items():
                        name = f'{key[0]}_{key[1]}'
                        payload[name] = layer['state']
                        payload[name+'_hash'] = layer['fingerprint']
                    tmp = self.path.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp, **payload)
                    tmp.replace(self.path)
                else:
                    with np.load(self.path, allow_pickle=False) as data:
                        pending = {}
                        for key, layer in self.layers.items():
                            name = f'{key[0]}_{key[1]}'
                            state = data[name]
                            if (str(data[name+'_hash']) != layer['fingerprint'] or state.shape != layer['state'].shape
                                    or not np.isin(state, [-1, 0, 1, 2]).all()):
                                raise ValueError('Repair file belongs to different geometry')
                            pending[key] = state.copy()
                    for key, state in pending.items():
                        self.layers[key]['state'] = state
                        self.rebuild(key)
                    self.undo.clear()
                    self.redo.clear()
                    self.clear()
                self.info.content = f'{action}: visual_repairs.npz'
            except (OSError, ValueError, KeyError) as e:
                self.info.content = f'Repair file error: {e}'

    def tick(self):
        for key, layer in self.layers.items():
            node = (self.nodes if key[1] == 'splat' else self.contacts).get(key[0])
            if node is not None and self.applied.get(key) is not node:
                node.opacities = (layer['a']['opacity'] * (layer['state'] == 0))[:, None]
                self.applied[key] = node
        for (k, part, target), node in self.overlays.items():
            layer = self.layers[(k, part)]
            node.visible = not self.ui['sim'] and (not self.isolate.value or k == int(self.pick.value))
            if target == 1:
                node.visible &= self.world.present(k)
                if self.world.present(k):
                    p, q = self.world.pose(k)
                    node.position, node.wxyz = tuple(p), tuple(q)
                scale = next(o['scale'] for o in self.sc.objects if o['id'] == k)
                name = (k, part, target)
                if self.overlay_scales.get(name) != scale:
                    mask = layer['state'] == target
                    a = {field: v[mask] for field, v in layer['a'].items()}
                    factor = scale if part == 'contact' else 1.
                    node.centers = a['means'] * factor
                    node.covariances = covariances(a) * factor**2
                    self.overlay_scales[name] = scale
            else:
                node.position = tuple(layer['base'])
