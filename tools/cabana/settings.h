#pragma once

#include <QDialog>
#include <QList>
#include <QStringList>
#include <QSpinBox>

class Settings : public QObject {
  Q_OBJECT

public:
  Settings();
  void save();
  void load();

  int fps = 10;
  int can_msg_log_size = 100;
  int cached_segment_limit = 3;
  int chart_height = 200;

  // session data
  QString dbc_name;
  QString selected_msg_id;
  QStringList charts;
  QString zoom_range;
  QList<int> splitter_sizes;

signals:
  void changed();
};

class SettingsDlg : public QDialog {
  Q_OBJECT

public:
  SettingsDlg(QWidget *parent);
  void save();
  QSpinBox *fps;
  QSpinBox *log_size ;
  QSpinBox *cached_segment;
  QSpinBox *chart_height;
};

extern Settings settings;
