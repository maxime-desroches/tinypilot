#include <cassert>
#include <iostream>
#include <cmath>
#include <iostream>

#include <QGuiApplication>
#include <QSurfaceFormat>
#include <QOpenGLContext>

#include "qt_window.hpp"

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout;
  GLWindow * glWindow = new GLWindow;

  main_layout->addWidget(glWindow);
  main_layout->setMargin(0);
  setLayout(main_layout);
}

GLWindow::~GLWindow() {
  makeCurrent();
  doneCurrent();
}


void GLWindow::timerEvent(QTimerEvent *) {
  update();
}

void GLWindow::initializeGL()
{

  initializeOpenGLFunctions();

  ui_state = new UIState();
  ui_state->fb_w = 1920;
  ui_state->fb_h = 1080;

  ui_nvg_init(ui_state);

  // Use QBasicTimer because its faster than QTimer
  // timer.start(12, this);
}

void GLWindow::resizeGL(int w, int h) {
  std::cout << "resize " << w << "x" << h << std::endl;

}

void GLWindow::paintGL() {
  ui_draw(ui_state);
}
