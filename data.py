"""Strict source-group splits and temporally identical crop/augmentation."""
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def read_rgb(path):
    # imdecode supports Windows paths containing Chinese characters.
    array = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError(f'Cannot decode image: {path}')
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def write_rgb(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f'Cannot encode {path}')
    buffer.tofile(str(path))


def load_manifest(path):
    path = Path(path).resolve()
    records = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ('source_id', 'group_id', 'frames'):
                if key not in row:
                    raise ValueError(f'{path}: missing {key}')
            row['frames'] = [str((path.parent / p).resolve()) for p in row['frames']]
            records.append(row)
    if not records:
        raise ValueError(f'Empty manifest: {path}')
    return records


def validate_splits(manifests):
    """Reject shared source/group/frame paths among all supplied partitions."""
    seen = {k: {} for k in ('source_id', 'group_id', 'frames')}
    summary = {}
    for split, manifest in manifests.items():
        records = load_manifest(manifest)
        for row in records:
            for key in seen:
                values = row[key] if key == 'frames' else [str(row[key])]
                for value in values:
                    previous = seen[key].get(value)
                    if previous is not None and previous != split:
                        raise ValueError(f'Data leakage: {key}={value} shared by {previous} and {split}')
                    seen[key][value] = split
        summary[split] = {'sequences': len(records), 'sources': len({r['source_id'] for r in records}),
                          'groups': len({r['group_id'] for r in records})}
    return summary


def spatial(images, size, rng, augment):
    if len({im.shape for im in images}) != 1:
        raise ValueError('All frames of one sequence must have the same dimensions')
    h, w = images[0].shape[:2]
    scale = max(1., size / min(h, w))
    if scale > 1:
        images = [cv2.resize(im, (int(np.ceil(w * scale)), int(np.ceil(h * scale)))) for im in images]
        h, w = images[0].shape[:2]
    y = rng.randint(0, h - size) if augment else (h - size) // 2
    x = rng.randint(0, w - size) if augment else (w - size) // 2
    flip = augment and rng.random() < .5
    ims = [im[y:y+size, x:x+size, :][:, ::-1 if flip else 1, :].copy() for im in images]
    return torch.from_numpy(np.stack(ims).transpose(0, 3, 1, 2)).float() / 255.


class SequenceDataset(Dataset):
    def __init__(self, manifest, clean_dir, frames=5, crop=256, training=True, seed=42):
        self.records = load_manifest(manifest)
        self.clean = sorted(p for p in Path(clean_dir).rglob('*') if p.suffix.lower() in IMAGE_EXTS)
        if not self.clean:
            raise ValueError(f'No clean-domain images: {clean_dir}')
        self.frames, self.crop, self.training, self.seed = frames, crop, training, seed
        for row in self.records:
            if len(row['frames']) != frames:
                raise ValueError(f'Expected {frames} consecutive frames: {row["source_id"]}')
            for p in row['frames']:
                if not Path(p).is_file():
                    raise FileNotFoundError(p)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rng = random if self.training else random.Random(self.seed + index)
        row = self.records[index]
        x = spatial([read_rgb(p) for p in row['frames']], self.crop, rng, self.training)
        clean = spatial([read_rgb(rng.choice(self.clean))], self.crop, rng, self.training)[0]
        return {'input': x, 'clean': clean, 'source_id': row['source_id']}
