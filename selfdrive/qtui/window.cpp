#include <cassert>
#include <QGuiApplication>
#include <QSurfaceFormat>
#include <QOpenGLContext>

#include "glwindow.hpp"
#include "window.hpp"

Window::Window(QWidget *parent) : QWidget(parent) {
  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  QSurfaceFormat::setDefaultFormat(format);

  GLWindow glWindow;
  glWindow.show();
}
