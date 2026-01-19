import os
import logging
from .helper import progressbar
import numpy as np
from typing import Dict, Any
from scipy.interpolate import interp1d
from .recon import get_num_frames, parse_fid_info
from .typing import Options

logger = logging.getLogger("brkraw.sordino")



def get_fid_filesize_in_gb(fid_fobj, 
                           recon_info: Dict[str, Any], 
                           options: Options) -> int:
    fid_fobj.seek(0, os.SEEK_END)
    file_size = fid_fobj.tell()
    fid_fobj.seek(0)

    if options.num_frames is not None:
        file_size *= get_num_frames(recon_info, options) / options.num_frames
    file_size_gb = file_size / (1024 ** 3)
    return file_size_gb


def get_num_segment(file_size_gb, recon_info: Dict[str, Any], options: Options):
    npro = recon_info['NPro']
    num_segs = int(np.ceil(file_size_gb / options.mem_limit)) if options.mem_limit > 0 else 1
    npro_per_seg = int(npro / num_segs)

    if residual_pro := npro % npro_per_seg:
        segs = [npro_per_seg for _ in range(num_segs - 1)] + [residual_pro]
    else:
        segs = [npro_per_seg for _ in range(num_segs)]
    segs = np.asarray(segs, dtype=int)
    return segs


def prep_fid_segmentation(fid_fobj, 
                          recon_info: Dict[str, Any], 
                          options: Options):
    file_size = get_fid_filesize_in_gb(fid_fobj, recon_info, options)
    segs = get_num_segment(file_size, recon_info, options)
    return segs


def get_timestamps(repetition_time, npro, num_frames):
    vol_scantime = float(repetition_time) * npro
    base_timestamps = np.arange(num_frames) * vol_scantime
    target_timestamps = base_timestamps + (vol_scantime / 2)
    return {'base': base_timestamps, 
            'target': target_timestamps}


def get_segmented_data(fid_f, fid_shape, fid_dtype, 
                       buff_size, num_frames, 
                       seg_size, pro_loc,
                       options):
    
    seg = []
    
    stc_buffer_size = int(buff_size / fid_shape[3])
    pro_offset = pro_loc * stc_buffer_size
    seg_buffer_size = stc_buffer_size * seg_size # total buffer size for current segment

    for t in range(num_frames):
        frame_offset = t * buff_size
        seek_loc = int(options.offset or 0) * buff_size + frame_offset + pro_offset
        fid_f.seek(seek_loc)
        seg.append(fid_f.read(seg_buffer_size))

    seg_data = np.frombuffer(b''.join(seg), dtype=fid_dtype)
    seg_data = seg_data.reshape([2, np.prod(fid_shape[1:3]), seg_size, num_frames], order='F')
    return {
        'data': seg_data,
        'offset': pro_offset,
    }


def interpolate_spoketiming(complex_feed, sample_id, pro_id, ref_timestamps, timestamps, corrected_data):
    """ interpolation step
    each projection interpolated the timing at the middle of projection
    number of spokes * receivers processed together
    therefore, the spoke timing within a projection will not be corrected
    instead this actually corrected the spoke timing for each projection
    as single TR represent time of one projection
    
    :param complex_feed: Description
    :param sample_id: Description
    :param pro_id: Description
    :param ref_timestamps: Description
    :param timestamps: Description
    :param corrected_data: Description
    """
    mag = np.abs(complex_feed)
    phase = np.angle(complex_feed)
    phase_unw = np.unwrap(phase)
    interp_mag   = interp1d(ref_timestamps, mag, kind='linear',
                            bounds_error=False, fill_value='extrapolate') # pyright: ignore[reportArgumentType]
    interp_phase = interp1d(ref_timestamps, phase_unw, kind='linear',
                            bounds_error=False, fill_value='extrapolate') # pyright: ignore[reportArgumentType]
    
    mag_t   = interp_mag(timestamps['target'])
    phase_t = interp_phase(timestamps['target'])
    
    phase_wrap = (phase_t + np.pi) % (2 * np.pi) - np.pi
    
    z_t = mag_t * np.exp(1j * phase_wrap)
    corrected_data[0, sample_id, pro_id, :] = z_t.real
    corrected_data[1, sample_id, pro_id, :] = z_t.imag
    return corrected_data


def correct_spoketiming(segs, fid_f, stc_f, recon_info, options):
    """ Correct timing of each spoke to align center of scan time
    (Same concept as slice timing correction, but applied to FID signal)
    """
    logger.debug("+ Spoketiming correction")
    fid_shape, fid_dtype = parse_fid_info(recon_info)
    buff_size = int(np.prod(fid_shape) * fid_dtype.itemsize)
    num_frames = get_num_frames(recon_info, options)
    repetition_time = recon_info['RepetitionTime_ms'] / 1000.0
    timestamps = get_timestamps(repetition_time, fid_shape[3], num_frames)
    
    pro_loc = 0
    results = {
        'buffer_size': 0,
        'dtype': None,
    }
    for seg_size in progressbar(segs, desc=' - Segments', ncols=100):
        # load data
        segmented = get_segmented_data(
            fid_f, fid_shape, fid_dtype, buff_size, num_frames, seg_size, pro_loc, options
            )
        
        seg_data = segmented['data']
        corrected_seg_data = np.empty_like(seg_data)
        for pro_id in range(seg_size):
            cur_pro = pro_loc + pro_id
            ref_timestamps = timestamps['base'] + (cur_pro * repetition_time)
            for sample_id in range(np.prod(fid_shape[1:3])):
                complex_feed = seg_data[0, sample_id, pro_id, :] + 1j * seg_data[1, sample_id, pro_id, :]
                try:
                    corrected_seg_data = interpolate_spoketiming(complex_feed, sample_id, pro_id, 
                                                                 ref_timestamps, timestamps, 
                                                                 corrected_seg_data)
                except Exception as e:
                    logger.debug("+ Exception occured during spoketiming correction")
                    logger.debug(f" - RefTimeStamps: {ref_timestamps}")
                    logger.debug(f" - DataFeed: {complex_feed}")
                    raise e
                
        # Store data
        for t in range(num_frames):
            frame_offset = t * buff_size
            stc_f.seek(frame_offset + segmented['offset'])
            stc_f.write(corrected_seg_data[:,:,:, t].flatten(order='F').tobytes())

        if results['dtype'] == None:
            results['dtype'] = corrected_seg_data.dtype
        else:
            assert results['dtype'] == corrected_seg_data.dtype
        
        pro_loc += seg_size

    results['buffer_size'] = np.prod([fid_shape]) * results['dtype'].itemsize
    return results
