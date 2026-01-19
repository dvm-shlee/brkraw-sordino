import numpy as np
from typing import Any, Dict, Tuple
from numpy.typing import NDArray
import logging
from .helper import progressbar
from .typing import Options

logger = logging.getLogger("brkraw.sordino")


def parse_fid_info(recon_info: Dict[str, Any]) -> Tuple[np.ndarray, np.dtype]:
    n_receivers = int(recon_info.get("EncNReceivers") or 0)
    n_points = int(recon_info.get("NPoints") or 0)
    n_pro = int(recon_info.get("NPro") or 0)
    dtype = recon_info['FIDDataType']
    if not all((n_receivers, n_points, n_pro)):
        raise ValueError("Missing reconstruction dimensions in recon_spec output.")
    return np.array([2, n_points, n_receivers, n_pro]), dtype


def get_num_frames(recon_info: Dict[str, Any], options: Options):
    """ Return number of data frames need to be reconstructed
    """
    total_frames = recon_info['NRepetitions']
    offset = getattr(options, 'offset') or 0
    avail_frames = total_frames - offset
    set_frames = getattr(options, 'num_frames') or total_frames
    
    if set_frames > avail_frames:
        diff = set_frames - avail_frames
        set_frames -= diff
    return set_frames


def parse_volume_shape(recon_info: Dict[str, Any], 
                       options: Options) -> NDArray[np.int_]:
    matrix = recon_info.get("Matrix")
    if matrix is None:
        matrix = [int(recon_info.get("NPoints") or 0)] * 3
        logger.warning("Matrix size missing; defaulting to %s.", matrix)
    ext_factors = getattr(options, 'ext_factors', None)
    if ext_factors is None: 
        ext_factors = [1.0, 1.0, 1.0]
    return np.asarray(matrix * np.asarray(ext_factors)).astype(int).tolist()


def get_dataobj_shape(recon_info: Dict[str, Any], 
                      options: Options):
    num_receivers = parse_fid_info(recon_info)[0][2]
    vol_shape = parse_volume_shape(recon_info, options)
    num_frame = get_num_frames(recon_info, options)

    if num_receivers > 1:
        return [num_receivers] + vol_shape + [num_frame]
    else:
        return vol_shape + [num_frame]


def nufft_adjoint(kspace, traj, volume_shape, operator='finufft'):
    """Run nufft and return the reconstucted image"""
    from mrinufft import get_operator
    
    dcf = np.sqrt(np.square(traj).sum(-1)).flatten() ** 2
    dcf /= dcf.max()
    traj = traj.copy() / 0.5 * np.pi
    
    nufft_op = get_operator(operator)(traj, shape=volume_shape, density=dcf)
    complex_img = nufft_op.adj_op(kspace.flatten())
    return complex_img


def recon_dataobj(fid_fobj, 
                  traj, 
                  recon_info: Dict[str, Any],
                  img_fobj,
                  options: Options,
                  override_buffer_size=None, 
                  override_dtype=None):
    img_fobj.seek(0)
    fid_shape, fid_dtype = parse_fid_info(recon_info)
    volume_shape = parse_volume_shape(recon_info, options)
    
    offset = getattr(options, 'offset') or 0
    num_frames = get_num_frames(recon_info, options)
    ignore_samples = getattr(options, 'ignore_samples') or 1

    if all(arg != None for arg in [override_buffer_size, override_buffer_size]):
        fid_fobj.seek(0)
        buffer_size = override_buffer_size
        fid_dtype = override_dtype
    else:
        buffer_size = int(np.prod(fid_shape) * fid_dtype.itemsize)
        buf_offset = offset * buffer_size
        fid_fobj.seek(buf_offset)
    
    trimmed_traj = traj[:, ignore_samples:, ...]
    logger.debug("Reconstruction traj shape: %s", trimmed_traj.shape)
    
    dtype = None
    for n in progressbar(range(num_frames), desc='frames', ncols=100):
        buffer = fid_fobj.read(buffer_size)
        vol = np.frombuffer(buffer, dtype=fid_dtype).reshape(fid_shape, order='F')
        vol = (vol[0] + 1j * vol[1])[np.newaxis, ...]
        k_space = vol.squeeze().T[..., ignore_samples:]
        logger.debug("Reconstruction k-space shape: %s", k_space.shape)
        n_receivers = fid_shape[2]
        
        if n_receivers > 1:
            recon_vol = []
            for ch_id in range(n_receivers):
                _k_space = k_space[:, ch_id, :]
                _vol = nufft_adjoint(_k_space, trimmed_traj, volume_shape)
                recon_vol.append(_vol)
            recon_vol = np.stack(recon_vol, axis=0)
        else:
            recon_vol = nufft_adjoint(k_space, trimmed_traj, volume_shape)
        if n == 0:
            dtype = recon_vol.dtype
        img_fobj.write(recon_vol.T.flatten(order="C").tobytes())
    return dtype

__all__ = [
    'recon_dataobj',
    'get_dataobj_shape',
]