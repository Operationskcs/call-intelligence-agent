import pytest

from app.services.phoneburner_roles import _guided_intros, resolve_phoneburner_roles


@pytest.mark.parametrize("speech", [
    "Te llamo de parte del equipo legal para ayudarte con tu accidente.",
    "Soy la persona encargada de ayudarle con el proceso en nuestra firma.",
    "Soy la persona que le va a ayudar con su caso en nuestro equipo legal.",
    "I'm calling from the legal team to assist you with your accident.",
    "I'll be the person helping you with your claim at our law firm.",
])
def test_opening_identifies_active_voice_not_first_voice(mock_call_event, speech):
    source = ("00:01 [Channel 1 Speaker 2]: Hello.\n"
              f"00:07 [Channel 0 Speaker 3]: {speech}\n"
              "03:21 [Channel 0 Speaker 3]: Gracias.\n"
              "03:23 [Channel 0 Speaker 4]: Gracias.")
    result = resolve_phoneburner_roles(source, mock_call_event)
    assert _guided_intros(source) == {"Channel 0 Speaker 3"}
    assert result.transcript.count("[Agent]") == 2
    assert "03:21 [Agent]: Gracias." in result.transcript
    assert result.roles["Channel 1 Speaker 2"] == result.roles["Channel 0 Speaker 4"] == "Unknown"


@pytest.mark.parametrize("speech", [
    "Me dijeron: te llamo de parte del equipo legal para ayudarte con tu accidente.",
    "Si te llamo de parte del equipo legal para ayudarte con tu accidente.",
    "¿Te llamo de parte del equipo legal para ayudarte con tu accidente?",
    "I can help you with your case.",
    "I'll be helping you with your case. My lawyer told me to.",
    'She said, "I am calling from the law firm to assist you with your accident."',
    "I am a family member and I'm calling from the legal team to help you with your case.",
    "Otra persona te llamó de parte del equipo legal para ayudarte con tu accidente.",
    "Cuando dices te llamo de parte del equipo legal para ayudarte con tu accidente.",
])
def test_unsafe_or_generic_opening_not_new_evidence(speech):
    assert not _guided_intros(f"00:05 [Speaker 3]: {speech}")


def test_consecutive_fragments_do_not_cross_identity_or_time(mock_call_event):
    a = "00:04 [Speaker 1]: Te llamo de parte del equipo legal."
    b = "00:10 [Speaker 1]: Para ayudarte con tu accidente."
    assert _guided_intros(a + "\n" + b) == {"Speaker 1"}
    for other in (b.replace("Speaker 1", "Speaker 2"), b.replace("00:10", "02:10"),
                  b.replace("00:10", "00:02"), b.replace("00:10 ", "")):
        assert not _guided_intros(a + "\n" + other)
    assert not _guided_intros(a + "\n00:08 [Speaker 2]: Hola.\n" + b)
    conflicting = a + "\n" + b + "\n00:44 [Speaker 1]: My car was hit."
    assert resolve_phoneburner_roles(conflicting, mock_call_event).roles["Speaker 1"] == "Unknown"
    unseparated = (a + "\n" + b).replace("Speaker 1", "Unseparated Channel 0")
    assert resolve_phoneburner_roles(unseparated, mock_call_event).roles["Unseparated Channel 0"] == "Unknown"


def test_echo_and_quoted_continuation_are_not_introductions():
    intro = "Te llamo de parte del equipo legal para ayudarte con tu accidente."
    assert not _guided_intros(f"00:05 [Speaker 1]: {intro}\n00:06 [Speaker 2]: {intro}")
    assert not _guided_intros(f"00:03 [Speaker 1]: Ella dijo:\n00:05 [Speaker 1]: {intro}")
