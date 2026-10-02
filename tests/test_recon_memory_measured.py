"""Measured reconstruction memory vs the estimate (WI-0095, WI-0097).

Peak RSS increase measured on macOS arm64 (ru_maxrss in bytes), one fresh
process per row. "recon": recon_dataobj only (WI-0097 bench "small": synthetic
radial geometry, random int32 FID; "direct": 160^3 fixture geometry with the
WI-0096 phantom FID and a given chunk size). "total": end-to-end get_dataobj
with an empty recon cache (160^3 fixture geometry, phantom FID, default limit).
estimate_k0 rows are the whole-scan solve measured in WI-0071/WI-0095 (stage 1
of WI-0097 keeps it). For every row the estimate is at or above the measurement
and at most MAX_OVER times it.
"""
import pytest

from brkraw_sordino import memguard

MAX_OVER = 2.0
#: (kind, n, n_pro, n_points, n_rx, estimate_k0, chunk_spokes, read_nbytes, measured_nbytes, source)
#: chunk_spokes None: the planner's choice without a budget (the cap).
MEASURED = [
    ("recon", 32, 3200, 32, 1, False, None, 0, 28344320, "WI-0097 small_n32_p3200_s32_rx1_k0_ph0"),
    ("recon", 32, 3200, 32, 1, False, None, 0, 27607040, "WI-0097 small_n32_p3200_s32_rx1_k0_ph1"),
    ("recon", 64, 12800, 64, 1, False, None, 0, 193380352, "WI-0097 small_n64_p12800_s64_rx1_k0_ph0"),
    ("recon", 64, 12800, 64, 1, False, None, 0, 185565184, "WI-0097 small_n64_p12800_s64_rx1_k0_ph1"),
    ("recon", 64, 12800, 64, 2, False, None, 0, 234356736, "WI-0097 small_n64_p12800_s64_rx2_k0_ph0"),
    ("recon", 64, 12800, 64, 2, False, None, 0, 241336320, "WI-0097 small_n64_p12800_s64_rx2_k0_ph1"),
    ("recon", 64, 12800, 64, 4, False, None, 0, 241647616, "WI-0097 small_n64_p12800_s64_rx4_k0_ph0"),
    ("recon", 64, 12800, 64, 4, False, None, 0, 295960576, "WI-0097 small_n64_p12800_s64_rx4_k0_ph1"),
    ("recon", 64, 12800, 64, 8, False, None, 0, 381304832, "WI-0097 small_n64_p12800_s64_rx8_k0_ph0"),
    ("recon", 64, 12800, 64, 8, False, None, 0, 492978176, "WI-0097 small_n64_p12800_s64_rx8_k0_ph1"),
    ("recon", 64, 12800, 64, 16, False, None, 0, 588677120, "WI-0097 small_n64_p12800_s64_rx16_k0_ph0"),
    ("recon", 64, 12800, 64, 16, False, None, 0, 808206336, "WI-0097 small_n64_p12800_s64_rx16_k0_ph1"),
    ("recon", 64, 25640, 64, 1, False, None, 0, 314441728, "WI-0097 small_n64_p25600_s64_rx1_k0_ph0"),
    ("recon", 64, 25640, 64, 1, False, None, 0, 299089920, "WI-0097 small_n64_p25600_s64_rx1_k0_ph1"),
    ("recon", 96, 28796, 96, 1, False, None, 0, 518651904, "WI-0097 small_n96_p28800_s96_rx1_k0_ph0"),
    ("recon", 96, 28796, 96, 1, False, None, 0, 566738944, "WI-0097 small_n96_p28800_s96_rx1_k0_ph1"),
    ("recon", 96, 28796, 96, 4, False, None, 0, 705216512, "WI-0097 small_n96_p28800_s96_rx4_k0_ph0"),
    ("recon", 96, 28796, 96, 4, False, None, 0, 843825152, "WI-0097 small_n96_p28800_s96_rx4_k0_ph1"),
    ("recon", 128, 51128, 128, 1, False, None, 0, 1253752832, "WI-0097 small_n128_p51200_s128_rx1_k0_ph0"),
    ("recon", 128, 51128, 128, 1, False, None, 0, 1190215680, "WI-0097 small_n128_p51200_s128_rx1_k0_ph1"),
    ("recon", 128, 12800, 64, 1, False, None, 0, 432996352, "WI-0097 small_n128_p12800_s64_rx1_k0_ph0"),
    ("recon", 128, 12800, 64, 1, False, None, 0, 424837120, "WI-0097 small_n128_p12800_s64_rx1_k0_ph1"),
    ("recon", 160, 80876, 640, 1, False, None, 0, 2672852992, "WI-0097 direct_rx1_k0_c0"),
    ("recon", 160, 80876, 640, 1, False, 10110, 0, 1454161920, "WI-0097 direct_rx1_k0_c10110"),
    ("recon", 160, 80876, 640, 1, False, 5055, 0, 943947776, "WI-0097 direct_rx1_k0_c5055"),
    ("recon", 160, 80876, 640, 2, False, None, 0, 2743173120, "WI-0097 direct_rx2_k0_c0"),
    ("recon", 160, 80876, 640, 2, False, 10110, 0, 1575157760, "WI-0097 direct_rx2_k0_c10110"),
    ("recon", 160, 80876, 640, 2, False, 5055, 0, 1294434304, "WI-0097 direct_rx2_k0_c5055"),
    ("recon", 160, 80876, 640, 4, False, None, 0, 4375511040, "WI-0097 direct_rx4_k0_c0"),
    ("recon", 160, 80876, 640, 4, False, 10110, 0, 2310668288, "WI-0097 direct_rx4_k0_c10110"),
    ("recon", 160, 80876, 640, 4, False, 5055, 0, 1883848704, "WI-0097 direct_rx4_k0_c5055"),
    ("total", 160, 80876, 640, 1, False, None, 229376000, 2905899008, "WI-0097 total_rx1_k0_default"),
    ("total", 160, 80876, 640, 2, False, None, 425984000, 3246669824, "WI-0097 total_rx2_k0_default"),
    ("total", 160, 80876, 640, 4, False, None, 819200000, 4623335424, "WI-0097 total_rx4_k0_default"),
    ("recon", 64, 12800, 64, 1, True, None, 0, 512802816, "WI-0071 recon_mem_k0_all.json"),
    ("recon", 64, 12800, 64, 2, True, None, 0, 655851520, "WI-0071 recon_mem_k0_all.json"),
    ("recon", 96, 28800, 96, 1, True, None, 0, 1605959680, "WI-0071 recon_mem_k0_all.json"),
    ("recon", 64, 25600, 64, 1, True, None, 0, 940359680, "WI-0071 recon_mem_k0_all.json"),
    ("recon", 128, 12800, 64, 1, True, None, 0, 950763520, "WI-0071 recon_mem_k0_all.json"),
    ("total", 64, 12800, 64, 1, True, None, 75497472, 523550720, "WI-0071 estimate_final.json:real_30_k0"),
    ("total", 160, 80876, 640, 1, True, None, 229376000, 25197248512, "WI-0095 run_rx1_k01_tm0.json"),
    ("total", 160, 80876, 640, 2, True, None, 425984000, 29263200256, "WI-0095 run_rx2_k01_tm0.json"),
]


def _estimate(row):
    kind, n, n_pro, n_points, n_rx, estimate_k0, chunk, read_nbytes, measured_nbytes, source = row
    if chunk is None:
        return read_nbytes + memguard.recon_nbytes(n_pro, n_points, n_rx, (n, n, n),
                                                   estimate_k0=estimate_k0)
    n_chunks = -(-n_pro // chunk)
    equal = -(-n_pro // n_chunks)
    return (read_nbytes + memguard.serial_fixed_nbytes(n_rx, (n, n, n))
            + memguard.chunk_nbytes(equal, n_points, n_rx))


def _row_id(row):
    kind, n, n_pro, n_points, n_rx, estimate_k0, chunk, read_nbytes, measured_nbytes, source = row
    index = MEASURED.index(row)
    return f"{kind}-{n}cubed-pro{n_pro}-pts{n_points}-rx{n_rx}-k0{int(estimate_k0)}-c{chunk}-{index}"


@pytest.mark.parametrize("row", MEASURED, ids=[_row_id(r) for r in MEASURED])
def test_estimate_is_at_or_above_every_measurement(row):
    est = _estimate(row)
    measured = row[8]
    assert est >= measured, (row, est)


@pytest.mark.parametrize("row", MEASURED, ids=[_row_id(r) for r in MEASURED])
def test_estimate_is_at_most_max_over_times_every_measurement(row):
    est = _estimate(row)
    measured = row[8]
    assert est <= MAX_OVER * measured, (row, est)


def test_two_receiver_160_scan_without_estimate_k0_fits_an_8_gb_laptop():
    """Plain 160^3, 2 receivers: 18.9 GiB peak before WI-0097 (WI-0096 product row); the
    serial plan takes the 4 GiB limit of an 8 GB computer as its budget and fits."""
    read = 32768000 + 3 * 131072000
    plan = memguard.recon_plan(80892, 640, 2, (160, 160, 160), budget_nbytes=4 * memguard.GIB - read)
    assert plan["fits"] is True and plan["n_chunks"] > 4
    assert plan["recon_nbytes"] + read <= 4 * memguard.GIB


def test_two_receiver_160_scan_with_estimate_k0_reproduces_the_reported_stop():
    recon = memguard.recon_nbytes(80892, 640, 2, (160, 160, 160), estimate_k0=True)
    read = 32768000 + 3 * 131072000
    assert recon + read == 41303801856
    assert f"{(recon + read) / memguard.GIB:.2f}" == "38.47"
