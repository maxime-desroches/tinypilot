#include <QMap>
#include <QSoundEffect>
#include <QString>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/ui.h"

const std::tuple<AudibleAlert, QString, int> sound_list[] = {
  // AudibleAlert, file name, loop count
  {AudibleAlert::ENGAGE, "engage.wav", 0},
  {AudibleAlert::DISENGAGE, "disengage.wav", 0},

  {AudibleAlert::ERROR_REFUSE, "error.wav", 0},
  {AudibleAlert::PROMPT, "prompt.wav", 0},
  {AudibleAlert::PROMPT_REPEAT, "prompt.wav", 0},

  {AudibleAlert::WARNING_SOFT, "warning_soft.wav", QSoundEffect::Infinite},
  {AudibleAlert::WARNING_IMMEDIATE, "warning_immediate.wav", QSoundEffect::Infinite},
  {AudibleAlert::WARNING_DISTRACTED, "warning_distracted.wav", QSoundEffect::Infinite},
};

class Sound : public QObject {
public:
  explicit Sound(QObject *parent = 0);

protected:
  void update();
  void setAlert(const Alert &alert);

  Alert current_alert = {};
  QMap<AudibleAlert, QPair<QSoundEffect *, int>> sounds;
  SubMaster sm;
  uint64_t started_frame;
};
