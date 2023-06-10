#include "selfdrive/ui/qt/maps/map_panel.h"

#include <QStackedLayout>

#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_settings.h"
#include "selfdrive/ui/ui.h"

MapPanel::MapPanel(const QMapboxGLSettings &mapboxSettings, QWidget *parent) : QFrame(parent) {
  auto stack = new QStackedLayout(this);
  stack->setContentsMargins(0, 0, 0, 0);


  stack->addWidget(new MapSettings(parent));

  auto map = new MapWindow(mapboxSettings);
  QObject::connect(uiState(), &UIState::offroadTransition, map, &MapWindow::offroadTransition);
  stack->addWidget(map);
}
