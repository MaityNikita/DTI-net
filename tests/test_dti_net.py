"""Pytest suite for DTI-Net prediction wrapper and Streamlit AppTest smoke check."""

import hashlib
import pathlib
import sys

from rdkit import Chem
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


BCR_ABL_SEQ = app.PRESETS["Preset 1: Imatinib + BCR-ABL Kinase (Leukemia)"]["protein"]
IMATINIB_SMILES = app.PRESETS["Preset 1: Imatinib + BCR-ABL Kinase (Leukemia)"]["smiles"]
CONTROL_SMILES = "CCO"  # Ethanol negative control


def _seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()[:16]


def test_binder_vs_control_ordering():
    """Known binder (Imatinib) scores higher pKd and probability than small-molecule control."""
    emb = app.get_target_esm2_resource(BCR_ABL_SEQ)
    sh = _seq_hash(BCR_ABL_SEQ)

    binder_pkd, binder_prob, binder_status = app.predict_compound_cached(
        IMATINIB_SMILES, sh, BCR_ABL_SEQ, emb
    )
    ctrl_pkd, ctrl_prob, _ = app.predict_compound_cached(
        CONTROL_SMILES, sh, BCR_ABL_SEQ, emb
    )

    assert binder_status == "HIGH"
    assert binder_pkd > ctrl_pkd
    assert binder_prob > ctrl_prob


def test_prediction_determinism():
    """Repeated calls (including after cache clear) return identical predictions and embeddings."""
    emb1 = app.get_target_esm2_resource(BCR_ABL_SEQ)
    emb2 = app.get_target_esm2_resource(BCR_ABL_SEQ)
    assert emb1 == emb2

    sh = _seq_hash(BCR_ABL_SEQ)
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    res1 = app.predict_compound_cached(test_smiles, sh, BCR_ABL_SEQ, emb1)
    app.predict_compound_cached.clear()
    res2 = app.predict_compound_cached(test_smiles, sh, BCR_ABL_SEQ, emb1)

    assert res1 == res2


def test_invalid_input_handling():
    """Malformed and empty SMILES remain safe inputs for the prediction wrapper."""
    emb = app.get_target_esm2_resource(BCR_ABL_SEQ)
    sh = _seq_hash(BCR_ABL_SEQ)

    assert Chem.MolFromSmiles("INVALID_SMILES_123") is None

    for invalid_smiles in ("", "INVALID_SMILES_123", "  "):

        pkd, prob, status = app.predict_compound_cached(
            invalid_smiles, sh, BCR_ABL_SEQ, emb
        )

        assert isinstance(pkd, float)
        assert isinstance(prob, float)
        assert status in {"HIGH", "MODERATE", "NONE"}
        assert 0.0 <= prob <= 100.0


def test_apptest_smoke():
    """Smoke test: app loads and renders comparison table without exceptions."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)
    at.run()

    assert not at.exception
    assert len(at.tabs) == 4
    assert len(at.dataframe) >= 1
