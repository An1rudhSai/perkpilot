import pytest

from perkpilot.extract import extract_card
from perkpilot.offline import ScriptedLLM
from perkpilot.terms import available_cards
from perkpilot.tools import Toolbox
from perkpilot.transactions import generate


@pytest.fixture(scope="session")
def terms():
    return {c: extract_card(c, ScriptedLLM()).terms for c in available_cards()}


@pytest.fixture
def toolbox(terms):
    return Toolbox(terms, generate())
