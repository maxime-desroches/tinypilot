#include <csignal>
#include <sys/resource.h>

#include <QApplication>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/navd/map_renderer.h"

int main(int argc, char *argv[]) {
  qInstallMessageHandler(swagLogMessageHandler);
  setpriority(PRIO_PROCESS, 0, -20);

  QApplication app(argc, argv);
  std::signal(SIGINT, sigTermHandler);
  std::signal(SIGTERM, sigTermHandler);

  MapRenderer *m = new MapRenderer(get_mapbox_settings(), true, &app);
  assert(m);

  return app.exec();
}
