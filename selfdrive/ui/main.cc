#include <sys/resource.h>

#include <QApplication>
#include <QTranslator>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -20);

  qInstallMessageHandler(swagLogMessageHandler);
  initApp(argc, argv);

  int result = 0;

  do {
    QString language_file = QString::fromStdString(Params().get("DeviceLanguage"));
    qDebug() << "Loading language:" << language_file;

    QTranslator translator;
    if (!translator.load(language_file, "/home/batman/openpilot/selfdrive/ui/translations")) {  // TODO: don't hardcode this
      qDebug() << "Failed to load translation file!";
    }
    QApplication a(argc, argv);
    a.installTranslator(&translator);

    MainWindow w;
    setMainWindow(&w);
    a.installEventFilter(&w);
    result = a.exec();
  } while (result == 99);

  return result;
}
