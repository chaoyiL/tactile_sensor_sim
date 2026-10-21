"""ZOS-API / Python.NET backend, tested with OpticStudio 2024 R1.

Creates its own standalone application and never uses an existing open lens.
The square pinhole is an explicit absorbing screen; the detector is behind it.
"""
from pathlib import Path
import math
import shutil
import time
import uuid
import numpy as np
from .geometry import camera_frame, strip_geometry, export_stl


def rotation_to_tilts(r):
    """OpticStudio convention R = Rx @ Ry @ Rz, angles in degrees."""
    y = math.asin(float(np.clip(r[0, 2], -1, 1)))
    if abs(math.cos(y)) > 1e-9:
        x, z = math.atan2(-r[1, 2], r[2, 2]), math.atan2(-r[0, 1], r[0, 0])
    else:
        x, z = math.atan2(r[2, 1], r[1, 1]), 0.
    return np.degrees([x, y, z])


class ZemaxSession:
    def __init__(self, config):
        try:
            import clr
        except ImportError as e:
            raise RuntimeError('Run uv sync --locked, then use uv run python on Windows to load pythonnet==3.0.5.') from e
        z = config['zemax']
        candidates = []
        if z.get('nethelper_dll'):
            candidates.append(Path(z['nethelper_dll']))
        if z.get('install_dir'):
            candidates.append(Path(z['install_dir']) / 'ZOSAPI_NetHelper.dll')
        candidates.append(Path.home() / 'Documents/Zemax/ZOS-API/Libraries/ZOSAPI_NetHelper.dll')
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Zemax') as k:
                candidates.append(Path(winreg.QueryValueEx(k, 'ZemaxRoot')[0]) / 'ZOS-API/Libraries/ZOSAPI_NetHelper.dll')
        except (OSError, ImportError):
            pass
        helper = next((p for p in candidates if p.is_file()), None)
        if helper is None:
            raise RuntimeError('Cannot find ZOSAPI_NetHelper.dll; set zemax.install_dir or nethelper_dll.')
        clr.AddReference(str(helper))
        import ZOSAPI_NetHelper
        initializer = ZOSAPI_NetHelper.ZOSAPI_Initializer
        ok = initializer.Initialize(z['install_dir']) if z.get('install_dir') else initializer.Initialize()
        if not ok:
            raise RuntimeError('OpticStudio initialization failed; check zemax.install_dir.')
        root = Path(initializer.GetZemaxDirectory())
        clr.AddReference(str(root / 'ZOSAPI.dll'))
        clr.AddReference(str(root / 'ZOSAPI_Interfaces.dll'))
        import ZOSAPI
        self.api, self.app = ZOSAPI, None
        self.connection = ZOSAPI.ZOSAPI_Connection()
        try:
            self.app = self.connection.CreateNewApplication()
            if self.app is None or not self.app.IsValidLicenseForAPI:
                raise RuntimeError('No valid ZOS-API license available.')
            self.system = self.app.CreateNewSystem(ZOSAPI.SystemType.NonSequential)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.app is not None:
            self.app.CloseApplication()
            self.app = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _pose(row, position, rotation):
    row.XPosition, row.YPosition, row.ZPosition = map(float, position)
    row.TiltAboutX, row.TiltAboutY, row.TiltAboutZ = map(float, rotation_to_tilts(rotation))


def _array2d(net_array):
    return np.fromiter(net_array, dtype=float).reshape(net_array.GetLength(0), net_array.GetLength(1))


def _archive(system, out):
    archive = system.Tools.OpenCreateZAR()
    try:
        archive.UseDataCompression = True
        if not archive.SetArchiveFileName(str(out / 'scene.zar')):
            raise RuntimeError('Could not select Zemax archive output')
        if not archive.RunAndWaitForCompletion() or not archive.Succeeded:
            raise RuntimeError('Zemax archive export failed: ' + str(archive.ErrorMessage))
    finally:
        archive.Close()
    if not (out / 'scene.zar').is_file() or (out / 'scene.zar').stat().st_size == 0:
        raise RuntimeError('Zemax did not write a valid archive')


def render(config, out, build_only=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    a, c, z = config['surface'], config['camera'], config['zemax']
    with ZemaxSession(config) as session:
        api, app, system = session.api, session.app, session.system
        nce = system.NCE
        system.SystemData.Units.LensUnits = api.SystemData.ZemaxSystemUnits.Millimeters
        system.SystemData.Units.SourceUnits = api.SystemData.ZemaxSourceUnits.Watts
        system.SystemData.Units.SourceUnitPrefix = getattr(api.SystemData.ZemaxUnitPrefix, 'None')
        system.SystemData.Units.AnalysisUnits = api.SystemData.ZemaxAnalysisUnits.WattsPerMMSq
        system.SystemData.Units.AnalysisUnitPrefix = getattr(api.SystemData.ZemaxUnitPrefix, 'None')
        nonseq = system.SystemData.NonSequentialData
        nonseq.MinimumRelativeRayIntensity = 1e-14
        nonseq.MinimumAbsoluteRayIntensity = 1e-30
        nonseq.MaximumIntersectionsPerRay = 100
        nonseq.MaximumSegmentsPerRay = 100

        token = uuid.uuid4().hex[:12]
        mesh_name = 'tactile_' + token + '.STL'
        triangle_count = export_stl(out / mesh_name, config)
        cad_dir = Path(app.ObjectsDir) / 'CAD Files'
        cad_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / mesh_name, cad_dir / mesh_name)

        # A private coating file avoids editing the user's COATING.DAT.
        # Spectral bands are traced independently; the table supplies true reflectance.
        coating_name = 'tactile_' + token + '.DAT'
        spectrum = sorted(zip(z['wavelengths_rgb_um'], a['reflectance_rgb']))
        lines = ['TABLE TACTILE_SKIN']
        for angle in (0, 90):
            lines.append('ANGL ' + str(angle))
            lines.extend(f'WAVE {wave:.10g} {rho:.10g} {rho:.10g} 0 0' for wave, rho in spectrum)
        (out / coating_name).write_text('\n'.join(lines) + '\n', encoding='ascii')
        shutil.copy2(out / coating_name, Path(app.CoatingDir) / coating_name)
        system.SystemData.Files.CoatingFile = coating_name

        def new_object(kind):
            row = nce.AddObject()
            settings = row.GetObjectTypeSettings(getattr(api.Editors.NCE.ObjectType, kind))
            if not row.ChangeType(settings):
                raise RuntimeError('Cannot create NSC object: ' + kind)
            return row

        # The initial default row can be removed once another row exists.
        skin = new_object('NullObject')
        nce.RemoveObjectAt(1)
        skin = nce.GetObjectAt(1)
        settings = skin.GetObjectTypeSettings(api.Editors.NCE.ObjectType.CADPartSTL)
        settings.FileName1 = mesh_name
        if not skin.ChangeType(settings):
            raise RuntimeError('Cannot import the surface STL')
        data = api.Editors.NCE.IObjectCADPartSTL(skin.ObjectData)
        data.Scale, data.IsVolume = 1., False
        skin.Material = 'MIRROR'
        face = skin.CoatScatterData.GetFaceData(0)
        face.Coating = 'TACTILE_SKIN'
        scatter = face.CreateScatterModelSettings(api.Editors.NCE.ObjectScatteringTypes.Lambertian)
        scatter._S_Lambertian.ScatterFraction = 1.
        if not face.ChangeScatterModelSettings(scatter):
            raise RuntimeError('Cannot configure Lambertian surface scattering')
        face.NumberOfRays = 1

        r = camera_frame(c)
        right, down, forward = r[:, 0], r[:, 1], r[:, 2]
        detector_frame = np.column_stack((right, -down, -forward))
        camera_center = np.asarray(c['position_mm'], float)
        target = new_object('NullObject')
        target.Comment = 'Pinhole center / importance-sampling target'
        _pose(target, camera_center, detector_frame)
        importance = skin.ScatterToData
        importance.ScatterToMethod = api.Editors.NCE.ScatterToType.ImportanceSampling
        ray = importance.GetRayData(1)
        ray.Towards = target.ObjectNumber
        ray.Size = math.sqrt(2) * z['pinhole_half_width_mm'] * 1.01
        ray.Limit = .01
        if not importance.SetRayData(1, ray):
            raise RuntimeError('Cannot configure importance sampling towards camera')

        # Explicit square hole: four large, absorbing coplanar rectangles.
        # No ray may reach the detector without crossing this screen.
        hole = z['pinhole_half_width_mm']
        # Keep the screen finite and well scaled to avoid precision loss for a tiny hole.
        extent = 3 * (max(a['size_mm']) + float(np.linalg.norm(camera_center)) + c['focal_length_mm'])
        screens = [(-(extent + hole) / 2, 0, (extent - hole) / 2, extent),
                   ((extent + hole) / 2, 0, (extent - hole) / 2, extent),
                   (0, -(extent + hole) / 2, hole, (extent - hole) / 2),
                   (0, (extent + hole) / 2, hole, (extent - hole) / 2)]
        for x, y, hx, hy in screens:
            row = new_object('Rectangle')
            row.Material = 'ABSORB'
            row.Comment = 'Camera absorbing pinhole screen'
            _pose(row, camera_center + right * x - down * y, detector_frame)
            data = api.Editors.NCE.IObjectRectangle(row.ObjectData)
            data.XHalfWidth, data.YHalfWidth = hx, hy

        detector = new_object('DetectorRectangle')
        detector.Comment = 'Camera sensor (real image behind square pinhole)'
        f, w, h = c['focal_length_mm'], c['width_px'], c['height_px']
        ox = f * (c['cx_px'] - (w - 1) / 2) / c['fx_px']
        oy = f * ((h - 1) / 2 - c['cy_px']) / c['fy_px']
        _pose(detector, camera_center - f * forward + ox * right - oy * down, detector_frame)
        detector.Material = 'ABSORB'
        data = api.Editors.NCE.IObjectDetectorRectangle(detector.ObjectData)
        data.XHalfWidth, data.YHalfWidth = f * w / (2 * c['fx_px']), f * h / (2 * c['fy_px'])
        data.NumberXPixels, data.NumberYPixels = w, h
        data.DataType, data.Smoothing, data.FrontOnly, data.Mirroring = 0, 0, 0, 0
        detector.TypeData.UsePixelInterpolation = False
        detector_number = detector.ObjectNumber

        waves = system.SystemData.Wavelengths
        while waves.NumberOfWavelengths > 1:
            waves.RemoveWavelength(waves.NumberOfWavelengths)
        waves.GetWavelength(1).Wavelength = z['wavelengths_rgb_um'][0]
        for wave in z['wavelengths_rgb_um'][1:]:
            waves.AddWavelength(wave, 1.)
        sources = []
        for strip in config['lights']:
            center, u, v, normal = strip_geometry(strip, config)
            for channel, power in enumerate(strip['power_rgb_w']):
                if power <= 0:
                    continue
                row = new_object('SourceRectangle')
                row.Comment = 'LED strip ' + strip['side'] + ' ' + 'RGB'[channel]
                _pose(row, center, np.column_stack((u, v, normal)))
                source = api.Editors.NCE.IObjectSources(row.ObjectData)
                source.NumberOfLayoutRays = 8
                source.NumberOfAnalysisRays = int(z['rays_per_strip'])
                source.Power, source.WaveNumber = float(power), channel + 1
                data = api.Editors.NCE.IObjectSourceRectangle(row.ObjectData)
                data.XHalfWidth, data.YHalfWidth = strip['length_mm'] / 2, strip['width_mm'] / 2
                data.SourceDistance, data.CosineExponent = 0., strip['cosine_exponent']
                sources.append((source, channel, float(power)))
        system.SaveAs(str(out / 'scene.zos'))
        if build_only:
            _archive(system, out)
            return None, {'triangles': triangle_count, 'scene': str(out / 'scene.zos')}

        sensor = np.zeros((h, w, 3))
        details = []
        for channel in range(3):
            print('Zemax: tracing ' + 'RGB'[channel] + ' band...', flush=True)
            for source, source_channel, power in sources:
                # OpticStudio replaces exactly zero source power with its default 1 W.
                # A negligible positive value safely disables a band without that reset.
                source.Power = power if source_channel == channel else 1e-30
                source.NumberOfAnalysisRays = int(z['rays_per_strip']) if source_channel == channel else 1
            trace = system.Tools.OpenNSCRayTrace()
            start = time.perf_counter()
            try:
                trace.ScatterNSCRays, trace.SplitNSCRays = True, False
                trace.UsePolarization, trace.IgnoreErrors, trace.SaveRays = True, True, False
                trace.SetRandomSeed(z['random_seed'] + channel)
                trace.ClearDetectors(0)
                completed = trace.RunAndWaitForCompletion()
                error = str(trace.ErrorMessage)
                if not completed or not trace.Succeeded:
                    raise RuntimeError('Zemax ray trace did not complete: ' + error)
                dead_errors = float(trace.DeadRayErrors)
                dead_threshold = float(trace.DeadRayThreshold)
                traced_energy = float(trace.GetTotalRayEnergy())
                if dead_errors > max(1e-20, traced_energy * 1e-6):
                    raise RuntimeError(f'Excess ray-error energy: {dead_errors:g} W of {traced_energy:g} W')
            finally:
                trace.Close()
            raw = _array2d(nce.GetAllDetectorDataSafe(detector_number, 1))  # W/mm^2
            if raw.shape != (h, w):
                raise RuntimeError(f'Unexpected detector grid {raw.shape}; expected {(h, w)}')
            # API grid rows run along +local y, columns along +local x.
            # This real image is inverted in x relative to OpenCV; y already matches.
            sensor[..., channel] = raw[:, ::-1] * 1e6
            system.SaveAs(str(out / ('scene_' + 'RGB'[channel] + '.zos')))
            details.append({'band': 'RGB'[channel], 'seconds': time.perf_counter() - start,
                            'irradiance_sum_w_m2': float(sensor[..., channel].sum()), 'tool_message': error,
                            'ray_error_energy_w': dead_errors, 'threshold_discarded_energy_w': dead_threshold,
                            'source_energy_w': traced_energy})
        # Keep every band enabled in the main scene; individual trace scenes are also saved.
        for source, source_channel, power in sources:
            source.Power = power
            source.NumberOfAnalysisRays = int(z['rays_per_strip'])
        system.SaveAs(str(out / 'scene.zos'))
        np.save(out / 'unmasked_sensor_irradiance_rgb_w_m2.npy', sensor.astype(np.float32))
        _archive(system, out)
        return sensor, {'triangles': triangle_count, 'bands': details,
                        'opticstudio_version': str(app.OpticStudioVersion),
                        'license': str(app.LicenseStatus),
                        'cad_file_installed': str(cad_dir / mesh_name),
                        'coating_file_installed': str(Path(app.CoatingDir) / coating_name),
                        'square_pinhole_half_width_mm': hole}
