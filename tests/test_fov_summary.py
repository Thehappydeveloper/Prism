from server_app.fov_summary import summarize_fov_segments


def test_perfect_fov_summary_is_perfect_and_compact() -> None:
    stats = summarize_fov_segments(
        [
            {
                "duration_s": 2.0,
                "predicted": {"T1": 1.0, "T2": 0.25, "T3": 0.0},
                "actual": {"T1": 1.0, "T2": 0.25, "T3": 0.0},
                "actual_frames": [
                    {"frame": 1, "tile_visibility": {"T1": 1.0, "T2": 0.25, "T3": 0.0}},
                ],
                "tile_quality": {"T1": 1.0, "T2": 0.5, "T3": 0.0},
                "tile_bits": {"T1": 2_000_000, "T2": 1_000_000, "T3": 500_000},
            }
        ]
    )

    assert stats is not None
    assert stats["mae"] == [0.0]
    assert stats["rmse"] == [0.0]
    assert stats["iou"] == [1.0]
    assert stats["weightedIou"] == [1.0]
    assert stats["top1Hit"] == [1.0]
    assert 0 < len(stats["calibration"]) <= 20


def test_fov_summary_tracks_missed_and_wasted_tiles() -> None:
    stats = summarize_fov_segments(
        [
            {
                "duration_s": 1.0,
                "predicted": {"visible": 0.0, "hidden": 1.0},
                "actual": {"visible": 1.0, "hidden": 0.0},
                "tile_quality": {"visible": 0.0, "hidden": 1.0},
                "tile_bits": {"visible": 100.0, "hidden": 300.0},
            }
        ]
    )

    assert stats is not None
    assert stats["iou"] == [0.0]
    assert stats["falsePositiveMass"] == [1.0]
    assert stats["falseNegativeMass"] == [1.0]
    assert stats["wastedUnseenBits"] == [300.0]
    assert stats["wastedUnseenBitFraction"] == [0.75]
    assert stats["missedVisibleQuality"] == [1.0]
