#pragma once

#include <QFrame>
#include <QWidget>

class WiFiPromptWidget : public QFrame {
  Q_OBJECT

public:
  explicit WiFiPromptWidget(QWidget* parent = 0);
};
