"""Public analytic renderer dispatch and result serialization."""
import json
import numpy as np
from PIL import Image
from .renderer_cpu import camera_ground_truth, camera_rays, render_cpu


def render(config, device=None):
    """Render with the configured analytic device.

    CUDA is the default for configs without an explicit device. Importing this
    module remains safe in CPU-only environments because PyTorch is loaded only
    when the CUDA path is selected.
    """
    selected = device or config.get('render', {}).get('device', 'cuda:0')
    if str(selected).lower() == 'cpu':
        return render_cpu(config)
    from .renderer_gpu import render_host
    return render_host(config, str(selected))


def encode_display(irradiance, config):
    """Fixed exposure across all frames; linear response matrix then sRGB display."""
    response = np.asarray(config['camera']['response_rgb_3x3'])
    rgb = irradiance @ response.T * config['render']['exposure_scale']
    rgb = np.maximum(rgb, 0)
    if config['render']['tone_map'] == 'reinhard':
        rgb = rgb / (1 + rgb)
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
    return np.uint8(np.round(255 * srgb))


def save_result(out, sensor, config, backend, gt=None, radiance=None, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encode_display(sensor, config)).save(out / 'image.png')
    np.save(out / 'sensor_irradiance_rgb_w_m2.npy', sensor.astype(np.float32))
    if gt is not None:
        np.savez_compressed(out / 'geometry_gt.npz', **gt)
    if radiance is not None:
        np.save(out / 'radiance_rgb_w_m2_sr.npy', radiance.astype(np.float32))
    meta = {'backend': backend, 'sensor_array_order': 'height,width,RGB',
            'sensor_unit': 'W/m^2 per independent RGB band',
            'image_encoding': 'fixed exposure + camera response matrix + sRGB',
            'config': config, **(extra or {})}
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
