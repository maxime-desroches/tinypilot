#include "selfdrive/ui/navd/route_engine.h"

#include <QDebug>

#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/qt/api.h"

#include "selfdrive/common/params.h"

const qreal REROUTE_DISTANCE = 25;
const float MANEUVER_TRANSITION_THRESHOLD = 10;

static float get_time_typical(const QGeoRouteSegment &segment) {
  auto maneuver = segment.maneuver();
  auto attrs = maneuver.extendedAttributes();
  return attrs.contains("mapbox.duration_typical") ? attrs["mapbox.duration_typical"].toDouble() : segment.travelTime();
}

static void parse_banner(cereal::NavInstruction::Builder &instruction, const QMap<QString, QVariant> &banner, bool full) {
  QString primary_str, secondary_str;

  auto p = banner["primary"].toMap();
  primary_str += p["text"].toString();


  if (p.contains("type")) {
    instruction.setManeuverType(p["type"].toString().toStdString());
  }

  if (p.contains("modifier")) {
    instruction.setManeuverModifier(p["modifier"].toString().toStdString());
  }

  if (banner.contains("secondary") && full) {
    auto s = banner["secondary"].toMap();
    secondary_str += s["text"].toString();
  }

  // TODO: Lanes

  instruction.setManeuverPrimaryText(primary_str.toStdString());
  instruction.setManeuverSecondaryText(secondary_str.toStdString());
}

RouteEngine::RouteEngine() {
  sm = new SubMaster({"liveLocationKalman"});
  pm = new PubMaster({"navInstruction"});

  // Timers
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));
  timer->start(100);

  recompute_timer = new QTimer(this);
  QObject::connect(recompute_timer, SIGNAL(timeout()), this, SLOT(recomputeRoute()));
  recompute_timer->start(1000);

  // Build routing engine
  QVariantMap parameters;
  QString token = MAPBOX_TOKEN.isEmpty() ? CommaApi::create_jwt({}, 4 * 7 * 24 * 3600) : MAPBOX_TOKEN;
  parameters["mapbox.access_token"] = token;
  parameters["mapbox.directions_api_url"] = MAPS_HOST + "/directions/v5/mapbox/";

  geoservice_provider = new QGeoServiceProvider("mapbox", parameters);
  routing_manager = geoservice_provider->routingManager();
  if (routing_manager == nullptr) {
    qWarning() << geoservice_provider->errorString();
    assert(routing_manager);
  }
  QObject::connect(routing_manager, &QGeoRoutingManager::finished, this, &RouteEngine::routeCalculated);

  // Get last gps position from params
  auto last_gps_position = coordinate_from_param("LastGPSPosition");
  if (last_gps_position) {
    last_position = *last_gps_position;
  }
}

void RouteEngine::timerUpdate() {
  sm->update(0);
  if (sm->updated("liveLocationKalman")) {
    auto location = (*sm)["liveLocationKalman"].getLiveLocationKalman();
    gps_ok = location.getGpsOK();

    localizer_valid = location.getStatus() == cereal::LiveLocationKalman::Status::VALID;

    if (localizer_valid) {
      auto pos = location.getPositionGeodetic();
      auto orientation = location.getCalibratedOrientationNED();

      float bearing = RAD2DEG(orientation.getValue()[2]);
      auto coordinate = QMapbox::Coordinate(pos.getValue()[0], pos.getValue()[1]);

      last_position = coordinate;
      last_bearing = bearing;
    }
  }

  MessageBuilder msg;
  cereal::Event::Builder evt = msg.initEvent(segment.isValid());
  cereal::NavInstruction::Builder instruction = evt.initNavInstruction();

  // Show route instructions
  if (segment.isValid()) {
    auto cur_maneuver = segment.maneuver();
    auto attrs = cur_maneuver.extendedAttributes();
    if (cur_maneuver.isValid() && attrs.contains("mapbox.banner_instructions")) {
      float along_geometry = distance_along_geometry(segment.path(), to_QGeoCoordinate(*last_position));
      float distance_to_maneuver_along_geometry = segment.distance() - along_geometry;

      auto banners = attrs["mapbox.banner_instructions"].toList();
      if (banners.size()) {
        auto banner = banners[0].toMap();

        for (auto &b : banners) {
          auto bb = b.toMap();
          if (distance_to_maneuver_along_geometry < bb["distance_along_geometry"].toDouble()) {
            banner = bb;
          }
        }

        instruction.setManeuverDistance(distance_to_maneuver_along_geometry);
        parse_banner(instruction, banner, distance_to_maneuver_along_geometry < banner["distance_along_geometry"].toDouble());

        // ETA
        float progress = distance_along_geometry(segment.path(), to_QGeoCoordinate(*last_position)) / segment.distance();
        float total_distance = segment.distance() * (1.0 - progress);
        float total_time = segment.travelTime() * (1.0 - progress);
        float total_time_typical = get_time_typical(segment) * (1.0 - progress);

        auto s = segment.nextRouteSegment();
        while (s.isValid()) {
          total_distance += s.distance();
          total_time += s.travelTime();
          total_time_typical += get_time_typical(s);

          s = s.nextRouteSegment();
        }
        instruction.setTimeRemaining(total_time);
        instruction.setTimeRemainingTypical(total_time_typical);
        instruction.setDistanceRemaining(total_distance);
      }

      // Transition to next route segment
      if (!shouldRecompute() && (distance_to_maneuver_along_geometry < -MANEUVER_TRANSITION_THRESHOLD)) {
        auto next_segment = segment.nextRouteSegment();
        if (next_segment.isValid()) {
          segment = next_segment;

          recompute_backoff = std::max(0, recompute_backoff - 1);
          recompute_countdown = 0;
        } else {
          qWarning() << "Destination reached";
          Params().remove("NavDestination");

          // Clear route if driving away from destination
          float d = segment.maneuver().position().distanceTo(to_QGeoCoordinate(*last_position));
          if (d > REROUTE_DISTANCE) {
            clearRoute();
          }
        }
      }
    }
  }

  pm->send("navInstruction", msg);
}

void RouteEngine::clearRoute() {
  segment = QGeoRouteSegment();
  nav_destination = QMapbox::Coordinate();
}

bool RouteEngine::shouldRecompute() {
  if (!segment.isValid()) {
    return true;
  }

  // Compute closest distance to all line segments in the current path
  float min_d = REROUTE_DISTANCE + 1;
  auto path = segment.path();
  auto cur = to_QGeoCoordinate(*last_position);
  for (size_t i = 0; i < path.size() - 1; i++) {
    auto a = path[i];
    auto b = path[i+1];
    if (a.distanceTo(b) < 1.0) {
      continue;
    }
    min_d = std::min(min_d, minimum_distance(a, b, cur));
  }
  return min_d > REROUTE_DISTANCE;

  // TODO: Check for going wrong way in segment
}

void RouteEngine::recomputeRoute() {
  if (!last_position) {
    return;
  }

  auto new_destination = coordinate_from_param("NavDestination");
  if (!new_destination) {
    clearRoute();
    return;
  }

  bool should_recompute = shouldRecompute();
  if (*new_destination != nav_destination) {
    qWarning() << "Got new destination from NavDestination param" << *new_destination;
    should_recompute = true;
  }

  if (!gps_ok && segment.isValid()) return; // Don't recompute when gps drifts in tunnels

  if (recompute_countdown == 0 && should_recompute) {
    recompute_countdown = std::pow(2, recompute_backoff);
    recompute_backoff = std::min(7, recompute_backoff + 1);
    calculateRoute(*new_destination);
  } else {
    recompute_countdown = std::max(0, recompute_countdown - 1);
  }
}

void RouteEngine::calculateRoute(QMapbox::Coordinate destination) {
  qWarning() << "Calculating route" << *last_position << "->" << destination;

  nav_destination = destination;
  QGeoRouteRequest request(to_QGeoCoordinate(*last_position), to_QGeoCoordinate(destination));
  request.setFeatureWeight(QGeoRouteRequest::TrafficFeature, QGeoRouteRequest::AvoidFeatureWeight);

  if (last_bearing) {
    QVariantMap params;
    int bearing = ((int)(*last_bearing) + 360) % 360;
    params["bearing"] = bearing;
    request.setWaypointsMetadata({params});
  }

  routing_manager->calculateRoute(request);
}

void RouteEngine::routeCalculated(QGeoRouteReply *reply) {
  bool got_route = false;
  if (reply->error() == QGeoRouteReply::NoError) {
    if (reply->routes().size() != 0) {
      qWarning() << "Got route response";

      route = reply->routes().at(0);
      segment = route.firstRouteSegment();

      // auto route_points = coordinate_list_to_collection(route.path());
      // TODO: send route to UI

      got_route = true;
    } else {
      qWarning() << "Got empty route response";
    }
  } else {
    qWarning() << "Got error in route reply" << reply->errorString();
  }

  reply->deleteLater();
}
