"""Measured reconstruction memory vs the estimate (WI-0095).

Peak RSS increase measured on macOS arm64 (ru_maxrss in bytes), one fresh
process per row. "recon": recon_dataobj only, synthetic trajectory (WI-0071).
"total": end-to-end get_dataobj with an empty recon cache (WI-0071 real v1
run; WI-0095 160^3 fixture geometry with a synthetic point-source FID).
For every row the estimate is at or above the measurement and at most
MAX_OVER times it.
"""
import pytest

from brkraw_sordino import memguard

MAX_OVER = 2.0
#: (kind, n, n_pro, n_points, n_rx, estimate_k0, read_nbytes, measured_nbytes, source)
MEASURED = [
    ("recon", 32, 3200, 32, 1, False, 0, 42729472, "WI-0071 recon_mem_all.json"),
    ("recon", 64, 12800, 64, 1, False, 0, 277168128, "WI-0071 recon_mem_all.json"),
    ("recon", 64, 12800, 64, 1, False, 0, 284377088, "WI-0071 recon_mem_all.json"),
    ("recon", 64, 12800, 64, 2, False, 0, 439336960, "WI-0071 recon_mem_all.json"),
    ("recon", 96, 28800, 96, 1, False, 0, 856555520, "WI-0071 recon_mem_all.json"),
    ("recon", 128, 51200, 128, 1, False, 0, 1928740864, "WI-0071 recon_mem_all.json"),
    ("recon", 64, 25600, 64, 1, False, 0, 527876096, "WI-0071 recon_mem_all.json"),
    ("recon", 128, 12800, 64, 1, False, 0, 530038784, "WI-0071 recon_mem_all.json"),
    ("recon", 64, 12800, 64, 1, False, 0, 312279040, "WI-0071 recon_mem_phase_all.json"),
    ("recon", 64, 12800, 64, 2, False, 0, 471842816, "WI-0071 recon_mem_phase_all.json"),
    ("recon", 96, 28800, 96, 1, False, 0, 944521216, "WI-0071 recon_mem_phase_all.json"),
    ("recon", 128, 51200, 128, 1, False, 0, 2130624512, "WI-0071 recon_mem_phase_all.json"),
    ("recon", 64, 12800, 64, 4, False, 0, 779616256, "WI-0071 recon_mem_phase_all.json"),
    ("recon", 64, 12800, 64, 8, False, 0, 1366081536, "WI-0071 recon_mem_channels_all.json"),
    ("recon", 64, 12800, 64, 16, False, 0, 2076377088, "WI-0071 recon_mem_channels_all.json"),
    ("recon", 96, 28800, 96, 4, False, 0, 2523447296, "WI-0071 recon_mem_channels_all.json"),
    ("recon", 64, 12800, 64, 1, True, 0, 512802816, "WI-0071 recon_mem_k0_all.json"),
    ("recon", 64, 12800, 64, 2, True, 0, 655851520, "WI-0071 recon_mem_k0_all.json"),
    ("recon", 96, 28800, 96, 1, True, 0, 1605959680, "WI-0071 recon_mem_k0_all.json"),
    ("recon", 64, 25600, 64, 1, True, 0, 940359680, "WI-0071 recon_mem_k0_all.json"),
    ("recon", 128, 12800, 64, 1, True, 0, 950763520, "WI-0071 recon_mem_k0_all.json"),
    ("total", 64, 12800, 64, 1, False, 75497472, 334921728, "WI-0071 estimate_final.json:real_30"),
    ("total", 64, 12800, 64, 1, True, 75497472, 523550720, "WI-0071 estimate_final.json:real_30_k0"),
    ("total", 160, 80876, 640, 1, False, 229376000, 13555335168, "WI-0095 run_rx1_k00_tm0.json"),
    ("total", 160, 80876, 640, 1, True, 229376000, 25197248512, "WI-0095 run_rx1_k01_tm0.json"),
    ("total", 160, 80876, 640, 2, False, 425984000, 20255244288, "WI-0095 run_rx2_k00_tm0.json"),
    ("total", 160, 80876, 640, 2, True, 425984000, 29263200256, "WI-0095 run_rx2_k01_tm0.json"),
    ("total", 160, 80876, 640, 4, False, 819200000, 31289573376, "WI-0095 run_rx4_k00_tm0.json"),
]


def _estimate(row):
    kind, n, n_pro, n_points, n_rx, estimate_k0, read_nbytes, measured_nbytes, source = row
    return read_nbytes + memguard.recon_nbytes(n_pro, n_points, n_rx, (n, n, n), estimate_k0=estimate_k0)


def _row_id(row):
    kind, n, n_pro, n_points, n_rx, estimate_k0, read_nbytes, measured_nbytes, source = row
    index = MEASURED.index(row)
    return f"{kind}-{n}cubed-pro{n_pro}-pts{n_points}-rx{n_rx}-k0{int(estimate_k0)}-{index}"


@pytest.mark.parametrize("row", MEASURED, ids=[_row_id(r) for r in MEASURED])
def test_estimate_is_at_or_above_every_measurement(row):
    est = _estimate(row)
    measured = row[7]
    assert est >= measured, (row, est)


@pytest.mark.parametrize("row", MEASURED, ids=[_row_id(r) for r in MEASURED])
def test_estimate_is_at_most_max_over_times_every_measurement(row):
    est = _estimate(row)
    measured = row[7]
    assert est <= MAX_OVER * measured, (row, est)


def test_two_receiver_160_scan_with_estimate_k0_reproduces_the_reported_stop():
    recon = memguard.recon_nbytes(80892, 640, 2, (160, 160, 160), estimate_k0=True)
    read = 32768000 + 3 * 131072000
    assert recon + read == 41303801856
    assert f"{(recon + read) / memguard.GIB:.2f}" == "38.47"
