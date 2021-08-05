#pragma once

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QStackedLayout>
#include <QWidget>
#include <QPushButton>

#include "cereal/gen/cpp/log.capnp.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/ui.h"


// ***** onroad widgets *****

class ButtonsWindow : public QWidget {
  Q_OBJECT

public:
  ButtonsWindow(QWidget* parent = 0);
  void updateState(const UIState &s);

private:
  QPushButton *dfButton;
//  QPushButton *mlButton;

  int dfStatus = -1;  // always initialize style sheet and send msg
  const QStringList dfButtonColors = {"#044389", "#24a8bc", "#fcff4b", "#37b868"};
  void updateDfButton(int status);
};

class OnroadAlerts : public QWidget {
  Q_OBJECT

public:
  OnroadAlerts(QWidget *parent = 0) : QWidget(parent) {};
  void updateAlert(const Alert &a, const QColor &color);

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QColor bg;
  Alert alert = {};
};

// container window for the NVG UI
class NvgWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit NvgWindow(QWidget* parent = 0);
  ~NvgWindow();

protected:
  void paintGL() override;
  void initializeGL() override;
  void resizeGL(int w, int h) override;

private:
  double prev_draw_t = 0;

public slots:
  void updateState(const UIState &s);
};

// container for all onroad widgets
class OnroadWindow : public QWidget {
  Q_OBJECT

public:
  OnroadWindow(QWidget* parent = 0);
  QWidget *map = nullptr;

private:
  void paintEvent(QPaintEvent *event);

  OnroadAlerts *alerts;
  NvgWindow *nvg;
  ButtonsWindow *buttons;
  QColor bg = bg_colors[STATUS_DISENGAGED];
  QHBoxLayout* split;

signals:
  void updateStateSignal(const UIState &s);
  void offroadTransitionSignal(bool offroad);

private slots:
  void offroadTransition(bool offroad);
  void updateState(const UIState &s);
};
