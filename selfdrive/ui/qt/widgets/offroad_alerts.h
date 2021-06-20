#pragma once

#include <map>

#include <QFrame>

#include "selfdrive/common/params.h"

class QLabel;
class QPushButton;
class QVBoxLayout;
class ScrollView;

class OffroadAlert : public QFrame {
  Q_OBJECT

public:
  explicit OffroadAlert(QWidget *parent = 0);
  void setCurrentIndex(int id);

  int alertCount = 0;
  bool updateAvailable;

private:
  void updateAlerts();

  Params params;
  std::map<std::string, QLabel*> alerts;

  QLabel *releaseNotes;
  QPushButton *rebootBtn;
  ScrollView *alertsScroll;
  ScrollView *releaseNotesScroll;
  QVBoxLayout *alerts_layout;

signals:
  void closeAlerts();

public slots:
  void refresh();
};
