"""Speaker attribution only; no customer messages, audio or identifiers."""
import pytest

from app.services.phoneburner_roles import _guided_intros, resolve_phoneburner_roles
from app.webhook.phoneburner import _phoneburner_agent_name


@pytest.mark.parametrize("team,offer", [
    ("Ah, ok, nosotros somos los abogados.", "Estamos interesados en ayudarte en caso de tu accidente."),
    ("Yes, we are the legal team.", "We're here to help you with your accident claim."),
])
def test_owns_firm_and_directed_offer_across_an_interruption(mock_call_event, team, offer):
    text = (f"00:07 [Speaker 3]: {team}\n00:17 [Speaker 1]: Hello.\n"
            f"00:30 [Speaker 3]: {offer}")
    result = resolve_phoneburner_roles(text, mock_call_event)
    assert result.roles == {"Speaker 3": "Agent", "Speaker 1": "Unknown"}
    assert result.transcript == text.replace("[Speaker 3]", "[Agent]").replace(
        "[Speaker 1]", "[Unknown (Speaker 1)]")


@pytest.mark.parametrize("team,label,at", [
    ("Nosotros somos los abogados.", "Speaker 1", "00:20"),
    ("Nosotros somos los abogados.", "Speaker 3", "01:12"),
    ("Nosotros somos los abogados de la parte contraria.", "Speaker 3", "00:20"),
    ("Me dijeron: nosotros somos los abogados.", "Speaker 3", "00:20"),
    ("¿Nosotros somos los abogados?", "Speaker 3", "00:20"),
    ("No, nosotros somos los abogados.", "Speaker 3", "00:20"),
])
def test_ambiguous_roles_abstain(team, label, at):
    assert not _guided_intros(f"00:04 [Speaker 3]: {team}\n"
                              f"{at} [{label}]: Estamos aquí para ayudarte con tu accidente.")


@pytest.mark.parametrize("head, label, expected", [
    ("A gente está te ligando da Ajuda", "Speaker 2", True),
    ("Estamos te ligando de l'Ajuda", "Speaker 2", True),
    ("A gente está te ligando da Ajuda", "Speaker 1", False),
    ("Ele disse: a gente está te ligando da Ajuda", "Speaker 2", False),
    ("A gente não está te ligando da Ajuda", "Speaker 2", False),
    ("A gente está te ligando da Clínica", "Speaker 2", False),
])
def test_portuguese_direct_intro_only(head, label, expected):
    raw = (f"00:12 [Speaker 2]: {head}\n"
           f"00:15 [{label}]: Latina para te ajudar com o seu acidente.")
    assert _guided_intros(raw) == ({"Speaker 2"} if expected else set())


def test_post_transfer_department_or_recorded_greeting_not_enough(mock_call_event):
    raw = ("00:00 [Speaker 0]: Thank you for calling American Compensation.\n"
           "00:05 [Speaker 1]: Fui a la terapia.\n"
           "00:12 [Speaker 0]: Soy su compañero de trabajo, te voy a transferir al departamento.")
    assert set(resolve_phoneburner_roles(raw, mock_call_event).roles.values()) == {"Unknown"}


@pytest.mark.parametrize("payload, expected", [
    ({"agent": {"first_name": " Taylor ", "last_name": " Jordan "}}, "Taylor Jordan"),
    ({"from": {"name": "First Choice"}, "agent": {"first_name": "Other", "last_name": "Name"}},
     "First Choice"),
    ({"agent": {"first_name": "Taylor"}}, None),
    ({"contact": {"first_name": "Taylor", "last_name": "Jordan"}}, None),
    ({"agent": {"first_name": 123, "last_name": "Jordan"}}, None),
])
def test_provider_agent_metadata(payload, expected):
    assert _phoneburner_agent_name(payload) == expected
