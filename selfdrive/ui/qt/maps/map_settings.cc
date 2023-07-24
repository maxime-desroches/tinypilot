#include "map_settings.h"

#include <QApplication>
#include <QDebug>

#include "common/util.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

MapSettings::MapSettings(bool closeable, QWidget *parent) : QFrame(parent) {
  setAttribute(Qt::WA_NoMousePropagation);

  auto *frame = new QVBoxLayout(this);
  frame->setContentsMargins(40, 40, 40, 0);
  frame->setSpacing(0);

  auto *heading_frame = new QHBoxLayout;
  heading_frame->setContentsMargins(0, 0, 0, 0);
  heading_frame->setSpacing(32);
  {
    if (closeable) {
      auto *close_btn = new QPushButton("←");
      close_btn->setStyleSheet(R"(
        QPushButton {
          color: #FFFFFF;
          font-size: 100px;
          padding-bottom: 8px;
          border 1px grey solid;
          border-radius: 70px;
          background-color: #292929;
          font-weight: 500;
        }
        QPushButton:pressed {
          background-color: #3B3B3B;
        }
      )");
      close_btn->setFixedSize(140, 140);
      QObject::connect(close_btn, &QPushButton::clicked, [=]() { emit closeSettings(); });
      // TODO: read map_on_left from ui state
      heading_frame->addWidget(close_btn);
    }

    auto *heading = new QVBoxLayout;
    heading->setContentsMargins(0, 0, 0, 0);
    heading->setSpacing(16);
    {
      auto *title = new QLabel(tr("NAVIGATION"), this);
      title->setStyleSheet("color: #FFFFFF; font-size: 54px; font-weight: 600;");
      heading->addWidget(title);

      auto *subtitle = new QLabel(tr("Manage at connect.comma.ai"), this);
      subtitle->setStyleSheet("color: #A0A0A0; font-size: 40px; font-weight: 300;");
      heading->addWidget(subtitle);
    }
    heading_frame->addLayout(heading, 1);
  }
  frame->addLayout(heading_frame);
  frame->addSpacing(32);

  current_widget = new DestinationWidget(DestinationWidget::Current, this);
  QObject::connect(current_widget, &DestinationWidget::actionClicked, [=]() {
    params.remove("NavDestination");
    updateCurrentRoute();
  });
  frame->addWidget(current_widget);
  frame->addSpacing(32);

  QWidget *destinations_container = new QWidget(this);
  destinations_layout = new QVBoxLayout(destinations_container);
  destinations_layout->setContentsMargins(0, 32, 0, 32);
  destinations_layout->setSpacing(20);
  destinations_layout->addWidget(home_widget = new DestinationWidget(DestinationWidget::Home, this));
  destinations_layout->addWidget(work_widget = new DestinationWidget(DestinationWidget::Work, this));
  QObject::connect(home_widget, &DestinationWidget::navigateTo, this, &MapSettings::navigateTo);
  QObject::connect(work_widget, &DestinationWidget::navigateTo, this, &MapSettings::navigateTo);
  destinations_layout->addStretch();

  ScrollView *destinations_scroller = new ScrollView(destinations_container, this);
  destinations_scroller->setFrameShape(QFrame::NoFrame);
  frame->addWidget(destinations_scroller);

  setStyleSheet("MapSettings { background-color: #333333; }");

  QObject::connect(NavigationRequest::instance(), &NavigationRequest::locationsUpdated, this, &MapSettings::updateLocations);
  QObject::connect(NavigationRequest::instance(), &NavigationRequest::nextDestinationUpdated, this, &MapSettings::updateCurrentRoute);

  current_locations = NavigationRequest::instance()->currentLocations();
}

void MapSettings::showEvent(QShowEvent *event) {
  updateCurrentRoute();
}

void MapSettings::updateCurrentRoute() {
  current_destination = {};
  auto dest = QString::fromStdString(params.get("NavDestination"));
  if (dest.size()) {
    QJsonDocument doc = QJsonDocument::fromJson(dest.trimmed().toUtf8());
    if (doc.isNull()) {
      qWarning() << "JSON Parse failed on NavDestination" << dest;
      return;
    }
    current_destination = doc.object();
  }
  current_widget->set(current_destination);
  if (isVisible()) refresh();
}

void MapSettings::updateLocations(const QJsonArray &locations) {
  current_locations = locations;
  refresh();
}

void MapSettings::refresh() {
  setUpdatesEnabled(false);

  auto get_w = [this](int i) {
    auto w = i < widgets.size() ? widgets[i] : widgets.emplace_back(new DestinationWidget);
    if (!w->parentWidget()) {
      destinations_layout->insertWidget(destinations_layout->count() - 1, w);
      QObject::connect(w, &DestinationWidget::navigateTo, this, &MapSettings::navigateTo);
    }
    return w;
  };

  home_widget->set({});
  work_widget->set({});

  int n = 0;
  for (auto location : current_locations) {
    DestinationWidget *w = nullptr;
    auto dest = location.toObject();
    if (dest["save_type"].toString() == NAV_TYPE_FAVORITE) {
      auto label = dest["label"].toString();
      if (label == NAV_FAVORITE_LABEL_HOME) w = home_widget;
      if (label == NAV_FAVORITE_LABEL_WORK) w = work_widget;
    }
    w = w ? w : get_w(n++);
    w->set(dest);
    w->setVisible(dest != current_destination);
  }
  for (; n < widgets.size(); ++n) widgets[n]->setVisible(false);

  setUpdatesEnabled(true);
}

void MapSettings::navigateTo(const QJsonObject &place) {
  QJsonDocument doc(place);
  params.put("NavDestination", doc.toJson().toStdString());
  updateCurrentRoute();
  emit closeSettings();
}

DestinationWidget::DestinationWidget(DestinationWidget::Type type, QWidget *parent) : type(type), QPushButton(parent) {
  auto *frame = new QHBoxLayout(this);
  frame->setContentsMargins(32, 24, 32, 24);
  frame->setSpacing(32);

  icon = new QLabel(this);
  icon->setAlignment(Qt::AlignCenter);
  icon->setFixedSize(96, 96);
  icon->setObjectName("icon");
  icon->setPixmap(typeIcon(type));
  frame->addWidget(icon);

  auto *inner_frame = new QVBoxLayout;
  inner_frame->setContentsMargins(0, 0, 0, 0);
  inner_frame->setSpacing(0);
  {
    title = new ElidedLabel(this);
    title->setAttribute(Qt::WA_TransparentForMouseEvents);
    inner_frame->addWidget(title);

    subtitle = new ElidedLabel(this);
    subtitle->setAttribute(Qt::WA_TransparentForMouseEvents);
    subtitle->setObjectName("subtitle");
    inner_frame->addWidget(subtitle);
  }
  frame->addLayout(inner_frame, 1);

  action = new QPushButton(this);
  action->setFixedSize(96, 96);
  action->setObjectName("action");
  action->setAttribute(Qt::WA_TransparentForMouseEvents, type != Type::Current);
  QObject::connect(action, &QPushButton::clicked, this,  &DestinationWidget::actionClicked);
  frame->addWidget(action);

  setFixedHeight(164);
  setProperty("current", type == Type::Current);
  setStyleSheet(R"(
    DestinationWidget { background-color: #202123; border-radius: 10px; }
    QLabel { color: #FFFFFF; font-size: 48px; font-weight: 400; }
    #subtitle, QLabel:disabled { color: #9BA0A5; }
    #icon { background-color: #3B4356; border-radius: 48px; }
    #action { font-size: 65px; font-weight: 600; color: #FFFFFF; padding-bottom: 4px; }

    [current="true"] { background-color: #E8E8E8; }
    [current="true"] QLabel { color: #000000; }
    [current="true"] QLabel:disabled { color: #A0000000; }
    [current="true"] #icon { background-color: #42906B; }
    [current="true"] #subtitle { color: #333333; }
    [current="true"] #action { color: #202123; }
    [current="false"]:pressed { background-color: #18191B; }
    [current="true"] #action:pressed { background-color: #D6D6D6; }
  )");
  QObject::connect(this, &QPushButton::clicked, [this]() { if (!dest.isEmpty()) emit navigateTo(dest); });
}

void DestinationWidget::set(const QJsonObject &destination) {
  if (type == Type::Home || type == Type::Work) {
    if (!destination.isEmpty()) {
      title->setText(type == Type::Home ? tr("Home") : tr("Work"));
      subtitle->setText(destination["place_name"].toString() + "," + destination["place_details"].toString());
    } else {
      title->setText(tr("No %1 location set").arg(type == Type::Home ? tr("home") : tr("work")));
    }
  } else {
    title->setText(destination.isEmpty() ? tr("No destination set") : destination["place_name"].toString());
    subtitle->setText(destination["place_details"].toString());
    if (type == Type::Recent) {
      icon->setPixmap(typeIcon((destination["save_type"] == NAV_TYPE_FAVORITE ? Type::Favorite : Type::Recent)));
    }
  }
  dest = destination;
  // TODO: use pixmap
  action->setText(type == Type::Current ? "×" : "→");
  action->setVisible(!dest.isEmpty());
  subtitle->setVisible(!dest.isEmpty());
  setEnabled(!dest.isEmpty());
  setVisible(true);
}

// singleton NavigationRequest

NavigationRequest *NavigationRequest::instance() {
  static NavigationRequest *request = new NavigationRequest(qApp);
  return request;
}

NavigationRequest::NavigationRequest(QObject *parent) : QObject(parent) {
  if (auto dongle_id = getDongleId()) {
    {
      // Fetch favorite and recent locations
      QString url = CommaApi::BASE_URL + "/v1/navigation/" + *dongle_id + "/locations";
      RequestRepeater *repeater = new RequestRepeater(this, url, "ApiCache_NavDestinations", 30, true);
      QObject::connect(repeater, &RequestRepeater::requestDone, this, &NavigationRequest::parseLocationsResponse);
    }
    {
      auto param_watcher = new ParamWatcher(this);
      QObject::connect(param_watcher, &ParamWatcher::paramChanged, this, &NavigationRequest::nextDestinationUpdated);

      // Destination set while offline
      QString url = CommaApi::BASE_URL + "/v1/navigation/" + *dongle_id + "/next";
      HttpRequest *deleter = new HttpRequest(this);
      RequestRepeater *repeater = new RequestRepeater(this, url, "", 10, true);
      QObject::connect(repeater, &RequestRepeater::requestDone, [=](const QString &resp, bool success) {
        if (success && resp != "null") {
          if (params.get("NavDestination").empty()) {
            qWarning() << "Setting NavDestination from /next" << resp;
            params.put("NavDestination", resp.toStdString());
          } else {
            qWarning() << "Got location from /next, but NavDestination already set";
          }
          // Send DELETE to clear destination server side
          deleter->sendRequest(url, HttpRequest::Method::DELETE);
        }

        // athena can set destination at any time
        param_watcher->addParam("NavDestination");
      });
    }
  }
}

static void swap(QJsonValueRef v1, QJsonValueRef v2) { std::swap(v1, v2); }

void NavigationRequest::parseLocationsResponse(const QString &response, bool success) {
  if (!success || response == prev_response) return;

  prev_response = response;
  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());
  if (doc.isNull()) {
    qWarning() << "JSON Parse failed on navigation locations" << response;
    return;
  }

  // Sort: alphabetical FAVORITES, and then most recent (as returned by API).
  // We don't need to care about the ordering of HOME and WORK. DestinationWidget always displays them at the top.
  locations = doc.array();
  std::stable_sort(locations.begin(), locations.end(), [](const QJsonValue &a, const QJsonValue &b) {
    bool has_favorite = a["save_type"] == NAV_TYPE_FAVORITE || b["save_type"] == NAV_TYPE_FAVORITE;
    return has_favorite && (std::tuple(a["save_type"].toString(), a["place_name"].toString()) <
                            std::tuple(b["save_type"].toString(), b["place_name"].toString()));
  });
  emit locationsUpdated(locations);
}
