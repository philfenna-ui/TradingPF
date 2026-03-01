from __future__ import annotations

from pathlib import Path

from main import run_pipeline


def test_run_pipeline_returns_expected_sections():
    result = run_pipeline("config/default.yaml")
    assert "recommendations" in result
    assert "risk_report" in result
    assert "allocation" in result
    assert "cross_asset_intelligence" in result
    assert len(result["recommendations"]) > 0
    assert Path(result["report_path"]).exists()

