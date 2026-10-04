"""Measured reconstruction memory vs the estimate (WI-0095, WI-0097).

Peak RSS increase measured on macOS arm64 (ru_maxrss in bytes), one fresh
process per row. "recon": recon_dataobj only (WI-0097 bench "small": synthetic
radial geometry, random int32 FID; "direct": 160^3 fixture geometry with the
WI-0096 phantom FID and a given chunk size). "total": end-to-end get_dataobj
with an empty recon cache (160^3 fixture geometry, phantom FID, default limit).
estimate_k0 rows name their solve: "toeplitz" (WI-0097 stage 2) or "samples"
(the sample-based normal operator, WI-0099; bench .cache/scratch/wi-0099/bench,
random int32 FID, 160^3 rows with the planner's chunk); each row is held to the
estimate of the solve it ran, so both estimates are pinned whichever one
``memguard.k0_method`` would pick (the whole-scan solve rows of WI-0071/WI-0095
no longer apply). For every row the estimate is at or above the measurement and
at most MAX_OVER times it.
"""
import pytest

from brkraw_sordino import memguard

MAX_OVER = 2.0
#: (kind, n, n_pro, n_points, n_rx, k0_solve, chunk_spokes, read_nbytes, measured_nbytes, source)
#: k0_solve: False (no estimate_k0), "toeplitz" or "samples"
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
    ("recon", 32, 3200, 32, 1, "toeplitz", None, 0, 77037568, "WI-0097 small_n32_p3200_s32_rx1_k1_ph0"),
    ("recon", 32, 3200, 32, 1, "toeplitz", None, 0, 76972032, "WI-0097 small_n32_p3200_s32_rx1_k1_ph1"),
    ("recon", 64, 12800, 64, 1, "toeplitz", None, 0, 444956672, "WI-0097 small_n64_p12800_s64_rx1_k1_ph0"),
    ("recon", 64, 12800, 64, 1, "toeplitz", None, 0, 437338112, "WI-0097 small_n64_p12800_s64_rx1_k1_ph1"),
    ("recon", 64, 25640, 64, 1, "toeplitz", None, 0, 556498944, "WI-0097 small_n64_p25600_s64_rx1_k1_ph0"),
    ("recon", 64, 25640, 64, 1, "toeplitz", None, 0, 562839552, "WI-0097 small_n64_p25600_s64_rx1_k1_ph1"),
    ("recon", 64, 12800, 64, 2, "toeplitz", None, 0, 420298752, "WI-0097 small_n64_p12800_s64_rx2_k1_ph0"),
    ("recon", 64, 12800, 64, 2, "toeplitz", None, 0, 421609472, "WI-0097 small_n64_p12800_s64_rx2_k1_ph1"),
    ("recon", 64, 12800, 64, 4, "toeplitz", None, 0, 423067648, "WI-0097 small_n64_p12800_s64_rx4_k1_ph0"),
    ("recon", 64, 12800, 64, 4, "toeplitz", None, 0, 477626368, "WI-0097 small_n64_p12800_s64_rx4_k1_ph1"),
    ("recon", 64, 12800, 64, 8, "toeplitz", None, 0, 507379712, "WI-0097 small_n64_p12800_s64_rx8_k1_ph0"),
    ("recon", 64, 12800, 64, 8, "toeplitz", None, 0, 610484224, "WI-0097 small_n64_p12800_s64_rx8_k1_ph1"),
    ("recon", 64, 12800, 64, 16, "toeplitz", None, 0, 794722304, "WI-0097 small_n64_p12800_s64_rx16_k1_ph0"),
    ("recon", 64, 12800, 64, 16, "toeplitz", None, 0, 994213888, "WI-0097 small_n64_p12800_s64_rx16_k1_ph1"),
    ("recon", 96, 28796, 96, 1, "toeplitz", None, 0, 1240711168, "WI-0097 small_n96_p28800_s96_rx1_k1_ph0"),
    ("recon", 96, 28796, 96, 1, "toeplitz", None, 0, 1251803136, "WI-0097 small_n96_p28800_s96_rx1_k1_ph1"),
    ("recon", 96, 28796, 96, 4, "toeplitz", None, 0, 1296564224, "WI-0097 small_n96_p28800_s96_rx4_k1_ph0"),
    ("recon", 96, 28796, 96, 4, "toeplitz", None, 0, 1477148672, "WI-0097 small_n96_p28800_s96_rx4_k1_ph1"),
    ("recon", 128, 12800, 64, 1, "toeplitz", None, 0, 2235727872, "WI-0097 small_n128_p12800_s64_rx1_k1_ph0"),
    ("recon", 128, 12800, 64, 1, "toeplitz", None, 0, 2253111296, "WI-0097 small_n128_p12800_s64_rx1_k1_ph1"),
    ("recon", 128, 51128, 128, 1, "toeplitz", None, 0, 2759229440, "WI-0097 small_n128_p51200_s128_rx1_k1_ph0"),
    ("recon", 128, 51128, 128, 1, "toeplitz", None, 0, 2755788800, "WI-0097 small_n128_p51200_s128_rx1_k1_ph1"),
    ("recon", 160, 80876, 640, 1, "toeplitz", None, 0, 4989599744, "WI-0097 direct_rx1_k1_c0"),
    ("recon", 160, 80876, 640, 1, "toeplitz", 10110, 0, 4337451008, "WI-0097 direct_rx1_k1_c10110"),
    ("recon", 160, 80876, 640, 1, "toeplitz", 5055, 0, 3596206080, "WI-0097 direct_rx1_k1_c5055"),
    ("recon", 160, 80876, 640, 2, "toeplitz", None, 0, 5142282240, "WI-0097 direct_rx2_k1_c0"),
    ("recon", 160, 80876, 640, 2, "toeplitz", 10110, 0, 4477173760, "WI-0097 direct_rx2_k1_c10110"),
    ("recon", 160, 80876, 640, 2, "toeplitz", 5055, 0, 3984752640, "WI-0097 direct_rx2_k1_c5055"),
    ("recon", 160, 80876, 640, 4, "toeplitz", None, 0, 5807472640, "WI-0097 direct_rx4_k1_c0"),
    ("recon", 160, 80876, 640, 4, "toeplitz", 10110, 0, 4796383232, "WI-0097 direct_rx4_k1_c10110"),
    ("recon", 160, 80876, 640, 4, "toeplitz", 5055, 0, 4350705664, "WI-0097 direct_rx4_k1_c5055"),
    ("total", 160, 80876, 640, 1, "toeplitz", None, 229376000, 5123457024, "WI-0097 total_rx1_k1_default"),
    ("total", 160, 80876, 640, 2, "toeplitz", None, 425984000, 5217959936, "WI-0097 total_rx2_k1_default"),
    ("total", 160, 80876, 640, 4, "toeplitz", None, 819200000, 5916852224, "WI-0097 total_rx4_k1_default"),
    ("recon", 160, 80876, 640, 2, "toeplitz", None, 0, 5147131904, "WI-0099 direct160_rx2_new_auto"),
    ("recon", 160, 80876, 640, 2, "samples", None, 0, 6555271168, "WI-0099 direct160_rx2_new_samples"),
    ("recon", 128, 12800, 64, 1, "samples", None, 0, 605339648, "WI-0099 small_n128_p12800_s64_rx1_auto_ph0"),
    ("recon", 128, 12800, 64, 1, "samples", None, 0, 621576192, "WI-0099 small_n128_p12800_s64_rx1_auto_ph1"),
    ("recon", 128, 12800, 64, 1, "samples", None, 0, 612122624, "WI-0099 small_n128_p12800_s64_rx1_samples_ph0"),
    ("recon", 128, 12800, 64, 1, "samples", None, 0, 603422720, "WI-0099 small_n128_p12800_s64_rx1_samples_ph1"),
    ("recon", 128, 12800, 64, 1, "toeplitz", None, 0, 2235809792, "WI-0099 small_n128_p12800_s64_rx1_toeplitz_ph0"),
    ("recon", 128, 12800, 64, 1, "toeplitz", None, 0, 2240724992, "WI-0099 small_n128_p12800_s64_rx1_toeplitz_ph1"),
    ("recon", 128, 51128, 128, 1, "samples", None, 0, 1730985984, "WI-0099 small_n128_p51200_s128_rx1_samples_ph0"),
    ("recon", 128, 51128, 128, 1, "samples", None, 0, 1754005504, "WI-0099 small_n128_p51200_s128_rx1_samples_ph1"),
    ("recon", 32, 3200, 32, 1, "samples", None, 0, 38027264, "WI-0099 small_n32_p3200_s32_rx1_samples_ph0"),
    ("recon", 32, 3200, 32, 1, "samples", None, 0, 38486016, "WI-0099 small_n32_p3200_s32_rx1_samples_ph1"),
    ("recon", 64, 12800, 64, 16, "samples", None, 0, 684113920, "WI-0099 small_n64_p12800_s64_rx16_samples_ph0"),
    ("recon", 64, 12800, 64, 16, "samples", None, 0, 898170880, "WI-0099 small_n64_p12800_s64_rx16_samples_ph1"),
    ("recon", 64, 12800, 64, 1, "samples", None, 0, 271056896, "WI-0099 small_n64_p12800_s64_rx1_samples_ph0"),
    ("recon", 64, 12800, 64, 1, "samples", None, 0, 266813440, "WI-0099 small_n64_p12800_s64_rx1_samples_ph1"),
    ("recon", 64, 12800, 64, 1, "toeplitz", None, 0, 449036288, "WI-0099 small_n64_p12800_s64_rx1_toeplitz_ph0"),
    ("recon", 64, 12800, 64, 2, "samples", None, 0, 308051968, "WI-0099 small_n64_p12800_s64_rx2_samples_ph0"),
    ("recon", 64, 12800, 64, 2, "samples", None, 0, 324075520, "WI-0099 small_n64_p12800_s64_rx2_samples_ph1"),
    ("recon", 64, 12800, 64, 4, "samples", None, 0, 307937280, "WI-0099 small_n64_p12800_s64_rx4_samples_ph0"),
    ("recon", 64, 12800, 64, 4, "samples", None, 0, 365182976, "WI-0099 small_n64_p12800_s64_rx4_samples_ph1"),
    ("recon", 64, 12800, 64, 8, "samples", None, 0, 468385792, "WI-0099 small_n64_p12800_s64_rx8_samples_ph0"),
    ("recon", 64, 12800, 64, 8, "samples", None, 0, 577536000, "WI-0099 small_n64_p12800_s64_rx8_samples_ph1"),
    ("recon", 64, 25640, 64, 1, "samples", None, 0, 446496768, "WI-0099 small_n64_p25600_s64_rx1_samples_ph0"),
    ("recon", 64, 25640, 64, 1, "samples", None, 0, 430194688, "WI-0099 small_n64_p25600_s64_rx1_samples_ph1"),
    ("recon", 96, 28796, 96, 1, "samples", None, 0, 742162432, "WI-0099 small_n96_p28800_s96_rx1_samples_ph0"),
    ("recon", 96, 28796, 96, 1, "samples", None, 0, 805847040, "WI-0099 small_n96_p28800_s96_rx1_samples_ph1"),
    ("recon", 96, 28796, 96, 4, "samples", None, 0, 944095232, "WI-0099 small_n96_p28800_s96_rx4_samples_ph0"),
    ("recon", 96, 28796, 96, 4, "samples", None, 0, 1141456896, "WI-0099 small_n96_p28800_s96_rx4_samples_ph1"),
    ("recon", 96, 28796, 96, 4, "toeplitz", None, 0, 1505345536, "WI-0099 small_n96_p28800_s96_rx4_toeplitz_ph1"),
]


def _k0_term(solve, n, n_pro, n_points):
    if not solve:
        return 0
    if solve == "toeplitz":
        return memguard.k0_fixed_nbytes((n, n, n))
    assert solve == "samples"
    return memguard.k0_samples_nbytes(n_pro, n_points, (n, n, n))


def _estimate(row):
    kind, n, n_pro, n_points, n_rx, solve, chunk, read_nbytes, measured_nbytes, source = row
    k0 = _k0_term(solve, n, n_pro, n_points)
    if chunk is None:
        return read_nbytes + memguard.recon_nbytes(n_pro, n_points, n_rx, (n, n, n)) + k0
    n_chunks = -(-n_pro // chunk)
    equal = -(-n_pro // n_chunks)
    return (read_nbytes + memguard.serial_fixed_nbytes(n_rx, (n, n, n)) + k0
            + memguard.chunk_nbytes(equal, n_points, n_rx))


def _row_id(row):
    kind, n, n_pro, n_points, n_rx, solve, chunk, read_nbytes, measured_nbytes, source = row
    index = MEASURED.index(row)
    return f"{kind}-{n}cubed-pro{n_pro}-pts{n_points}-rx{n_rx}-k0{solve or 0}-c{chunk}-{index}"


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


def test_two_receiver_160_scan_with_estimate_k0():
    """160^3, 2 receivers, estimate_k0: 38.47 GiB estimated and 27.3 GiB measured before
    WI-0097 (WI-0095); the Toeplitz solve needs a grid part that does not shrink with the
    chunk, so an 8 GB computer (4 GiB limit) still stops and is offered a limit (measured
    3.71 GiB with 5055-spoke chunks plus the read); a 16 GB computer (8 GiB) fits."""
    read = 32768000 + 3 * 131072000
    small = memguard.recon_plan(80892, 640, 2, (160, 160, 160), estimate_k0=True,
                                budget_nbytes=4 * memguard.GIB - read)
    assert small["fits"] is False and small["chunk_spokes"] == memguard.MIN_CHUNK_SPOKES
    assert 4 * memguard.GIB < small["recon_nbytes"] + read < 6 * memguard.GIB
    info = {"peak_nbytes": small["recon_nbytes"] + read, "limit_nbytes": 4 * memguard.GIB,
            "limit_source": "half of physical memory", "frames": 1, "shape": (160, 160, 160, 2),
            "dtype": "float64", "count": 1, "cached": False}
    with pytest.raises(memguard.SordinoResourceError) as err:
        memguard.check(info)
    assert err.value.retry_kwargs == {"max_memory_gb": 5.3}      # README / docs example
    big = memguard.recon_plan(80892, 640, 2, (160, 160, 160), estimate_k0=True,
                              budget_nbytes=8 * memguard.GIB - read)
    assert big["fits"] is True and big["recon_nbytes"] + read <= 8 * memguard.GIB


def test_the_chosen_solve_is_the_planned_k0_term():
    """With chunk None the estimate of a row whose solve is the one ``k0_method`` picks is
    the planner's own estimate (``recon_nbytes(..., estimate_k0=True)``)."""
    checked = 0
    for row in MEASURED:
        kind, n, n_pro, n_points, n_rx, solve, chunk, read_nbytes, _, _ = row
        if solve and chunk is None and memguard.k0_method(n_pro, n_points, n_rx, (n, n, n))["method"] == solve:
            assert _estimate(row) == read_nbytes + memguard.recon_nbytes(
                n_pro, n_points, n_rx, (n, n, n), estimate_k0=True)
            checked += 1
    assert checked >= 20


def test_few_samples_on_a_large_grid_are_back_at_the_whole_scan_level():
    """WI-0097 decision 2 (D-0136, WI-0099 acceptance 3): 128^3 with 12,800 x 64 samples
    took 0.95 GB (906.7 MiB) with the whole-scan solve (WI-0071) and 2.08 GiB with the
    Toeplitz solve; the sample-based solve is picked (it halves the estimate, D-0143) and
    measured below the WI-0071 value."""
    assert memguard.k0_method(12800, 64, 1, (128, 128, 128))["method"] == "samples"
    wi0071 = 950763520                                   # 906.71875 MiB, recon_mem_k0_128_12800_64_1_2
    rows = [r for r in MEASURED if r[1:6] == (128, 12800, 64, 1, "samples")]
    assert len(rows) >= 4 and all(r[8] < wi0071 for r in rows)
    toeplitz = [r for r in MEASURED if r[1:6] == (128, 12800, 64, 1, "toeplitz")]
    assert toeplitz and all(r[8] > 2 * wi0071 for r in toeplitz)
