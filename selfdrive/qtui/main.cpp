#include <QApplication>

#include "window.hpp"
#include "glwindow.hpp"

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);

  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  QSurfaceFormat::setDefaultFormat(format);


  MainWindow w;
  w.setFixedSize(1920, 1080);
  w.show();

  // GLWindow glWindow;
  // glWindow.setFixedSize(1920, 1080);
  // glWindow.show();
  return a.exec();
}
