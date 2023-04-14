#pragma once

#include <QButtonGroup>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QDialog>

class OpenRouteDialog : public QDialog {
  Q_OBJECT

public:
  OpenRouteDialog(QWidget *parent);
  void loadRoute();
  inline bool failedToLoad() const { return failed_to_load; }

private:
  QLineEdit *route_edit;
  QButtonGroup *video_btn_group;
  QDialogButtonBox *btn_box;
  bool failed_to_load = false;
};
