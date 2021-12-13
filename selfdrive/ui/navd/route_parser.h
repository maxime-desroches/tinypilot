#pragma once

#include <QGeoRouteRequest>
#include <QGeoRouteSegment>
#include <QJsonObject>
#include <QUrl>

#include "selfdrive/ui/navd/route_reply.h"

class RouteParser : public QObject {
  Q_OBJECT

public:
  RouteParser() {}

  RouteReply::Error parseReply(QList<Route> &routes, QString &error_string, const QByteArray &reply) const;
  QUrl requestUrl(const QGeoRouteRequest &request, const QString &prefix) const;
};
