import json
import pathlib

from streamlit.testing.v1 import AppTest

import app

ROOT = pathlib.Path(__file__).resolve().parent.parent


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_chembl_compounds_are_fetched_in_batches(monkeypatch):
    responses = {
        "target.json": FakeResponse(200, {"targets": [{"target_chembl_id": "CHEMBL1"}]}),
        "mechanism.json": FakeResponse(200, {
            "mechanisms": [
                {"molecule_chembl_id": "CHEMBL2"},
                {"molecule_chembl_id": "CHEMBL3"},
                {"molecule_chembl_id": "CHEMBL2"},
            ]
        }),
        "molecule.json": FakeResponse(200, {"molecules": [
            {
                "molecule_chembl_id": "CHEMBL2",
                "pref_name": "Drug Two",
                "max_phase": 4,
                "molecule_structures": {"canonical_smiles": "CCO"},
            },
            {
                "molecule_chembl_id": "CHEMBL3",
                "pref_name": "Drug Three",
                "max_phase": 1,
                "molecule_structures": {"canonical_smiles": "CC"},
            },
        ]}),
        "activity.json": FakeResponse(200, {"activities": [
            {"molecule_chembl_id": "CHEMBL2", "pchembl_value": "7.25"},
        ]}),
    }
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        endpoint = url.rsplit("/", 1)[-1]
        return responses[endpoint]

    monkeypatch.setattr(app.requests, "get", fake_get)
    app.fetch_chembl_compounds.clear()

    compounds, source = app.fetch_chembl_compounds("TARG", "P12345")

    assert source == "chembl_api"
    assert compounds == [{
        "name": "Drug Two",
        "smiles": "CCO",
        "max_phase": 4,
        "exp_pkd": 7.25,
    }]
    assert len(calls) == 4
    molecule_params = next(
        kwargs["params"] for url, kwargs in calls if url.endswith("molecule.json")
    )
    assert molecule_params["molecule_chembl_id__in"] == "CHEMBL2,CHEMBL3"


def test_chembl_failure_returns_bundled_compounds(monkeypatch):
    fallback = [{"name": "Demo drug", "smiles": "CCO", "max_phase": 4}]

    def fail_request(*args, **kwargs):
        raise app.requests.ConnectionError("offline")

    monkeypatch.setattr(app.requests, "get", fail_request)
    app.fetch_chembl_compounds.clear()

    compounds, source = app.fetch_chembl_compounds(
        "TARG", "P12345", json.dumps(fallback)
    )

    assert compounds == fallback
    assert source == "bundled_fallback"


def test_demo_screen_uses_bundled_compounds_without_network(monkeypatch):
    fallback = [{"name": "Demo drug", "smiles": "CCO", "max_phase": 4}]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("demo cache should not call ChEMBL")

    monkeypatch.setattr(app.requests, "get", fail_if_called)
    app.fetch_chembl_compounds.clear()

    compounds, source = app.fetch_chembl_compounds(
        "TARG", "P12345", json.dumps(fallback), prefer_fallback=True
    )

    assert compounds == fallback
    assert source == "bundled_cache"


def test_disease_screening_renders_cached_demo_results(monkeypatch):
    def unavailable_request(*args, **kwargs):
        return FakeResponse(503)

    monkeypatch.setattr(app.requests, "get", unavailable_request)
    app.fetch_uniprot_sequence.clear()
    app.fetch_chembl_compounds.clear()

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.button(key="run_disease_btn").click()
    at.run()

    assert not at.exception
    assert any("Drug name" in frame.value.columns for frame in at.dataframe)
