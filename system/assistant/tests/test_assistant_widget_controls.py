from cereal import messaging, log
from openpilot.system.assistant.rev_speechd import AssistantWidgetControl
STTState = log.SpeechToText.State

if __name__ == "__main__":
    awc = AssistantWidgetControl()
    awc.begin()
    awc.set_text("TEST", final=False)

    awc.set_text("ENDING TEST", final=True)
    awc.empty()
    awc.error()
