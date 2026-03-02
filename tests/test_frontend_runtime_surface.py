from streamlit.testing.v1 import AppTest


def test_root_surface_shows_diagnostics_and_ocr_fallback(monkeypatch):
    monkeypatch.setenv("DATASTORE", "sqlite")
    app_test = AppTest.from_file("app.py")
    app_test.run(timeout=60)

    assert len(app_test.error) == 0

    diagnostics_found = any(
        "Diagnostics | query.embed=" in str(cap.value) for cap in app_test.caption
    )
    assert diagnostics_found

    ocr_fallback_found = any(
        "OCR Quick Entry" in str(md.value) for md in app_test.markdown
    )
    assert ocr_fallback_found
