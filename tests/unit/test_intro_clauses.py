import pytest

from app.services.phoneburner_roles import _guided_intros, resolve_phoneburner_roles


@pytest.mark.parametrize("text", [
    "Te llamo de parte del equipo legal para ayudarte con tu accidente. ¿Cómo ha estado? ¿Cómo se siente hoy?",
    "Llamo de parte del equipo legal para asistirle con el accidente. ¿Cómo ha estado?",
    "I am calling from the legal team to help you with your claim. How are you doing today?",
    "Te llamo de parte del equipo legal para ayudarte con tu caso, ¿me escucha?",
])
def test_standalone_declaration_survives_courtesy_question(text, mock_call_event):
    raw = f"00:05 [Speaker 1]: {text}\n00:11 [Speaker 0]: Hola."
    result = resolve_phoneburner_roles(raw, mock_call_event)
    assert _guided_intros(raw) == {"Speaker 1"}
    assert result.transcript == f"00:05 [Agent]: {text}\n00:11 [Unknown (Speaker 0)]: Hola."


@pytest.mark.parametrize("text", [
    "¿Te llamo de parte del equipo legal para ayudarte con tu caso? ¿Cómo estás?",
    "Te llamo de parte del equipo legal. ¿Cómo puedo ayudarte con tu accidente?",
    "Ella me dijo que te llamo de parte del equipo legal para ayudarte con tu accidente. ¿Cómo está?",
    "¿Cómo ha estado? ¿Cómo puedo ayudarle con su caso?",
    "Te llamo de parte del equipo legal. ¿Quieres que diga que yo voy a ayudarte con tu caso?",
    "You told me I am calling from the legal team to help you with your case. How are you?",
    "Te llamo de parte del equipo legal para ayudarte con tu accidente? How are you doing today?",
])
def test_no_loan_of_evidence_from_question_or_quotation(text):
    assert not _guided_intros(f"00:04 [Speaker 0]: {text}")


def test_firm_and_case_assistance_must_be_same_continuous_voice(mock_call_event):
    start = "00:07 [Speaker 1]: Mucho gusto, te estamos llamando de parte de los abogados."
    help_ = "00:12 [Speaker 1]: Estamos interesados en ayudarte en el caso de tu accidente."
    assert _guided_intros(start + "\n" + help_) == {"Speaker 1"}
    assert not _guided_intros(start + "\n" + help_.replace("Speaker 1", "Speaker 0"))
    assert not _guided_intros(start + "\n" + help_.replace("00:12", "01:20"))
    assert not _guided_intros("00:03 [Speaker 1]: Mi hermano te llamó de parte de los abogados.\n" + help_)
    assert resolve_phoneburner_roles(start + "\n" + help_ + "\n00:17 [Speaker 1]: My car was hit.",
                                     mock_call_event).roles["Speaker 1"] == "Unknown"
