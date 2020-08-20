#include <string>
#include <iostream>
#include <sstream>

#include "qt/settings.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QScrollArea>
#include <QCheckBox>
#include <QScroller>
#include <QScrollerProperties>
#include <QtDBus>
#include <QDebug>

typedef QMap<QString, QMap<QString, QVariant> > Connection;
Q_DECLARE_METATYPE(Connection)

SettingsWindow::SettingsWindow(QWidget *parent) : QWidget(parent) {
  qDBusRegisterMetaType<Connection>();

  QWidget *container = new QWidget(this);
  QVBoxLayout *checkbox_layout = new QVBoxLayout();

  for(int i = 0; i < 25; i++){
    QCheckBox *chk = new QCheckBox("Check Box " + QString::number(i+1));
    checkbox_layout->addWidget(chk);
    checkbox_layout->addSpacing(50);
  }
  container->setLayout(checkbox_layout);

  QScrollArea *scrollArea = new QScrollArea;
  scrollArea->setWidget(container);

  QScrollerProperties sp;
  sp.setScrollMetric(QScrollerProperties::DecelerationFactor, 2.0);

  QScroller* qs = QScroller::scroller(scrollArea);
  qs->setScrollerProperties(sp);

  QHBoxLayout *main_layout = new QHBoxLayout;
  main_layout->addWidget(scrollArea);

  QPushButton * button = new QPushButton("Close");
  main_layout->addWidget(button);

  setLayout(main_layout);

  QScroller::grabGesture(scrollArea, QScroller::LeftMouseButtonGesture);
  QObject::connect(button, SIGNAL(clicked()), this, SIGNAL(closeSettings()));


  QString nm_path = "/org/freedesktop/NetworkManager";
  QString nm_settings_path = "/org/freedesktop/NetworkManager/Settings";

  QString nm_iface = "org.freedesktop.NetworkManager";
  QString props_iface = "org.freedesktop.DBus.Properties";
  QString nm_settings_iface = "org.freedesktop.NetworkManager.Settings";

  QString nm_service = "org.freedesktop.NetworkManager";
  QString device_service = "org.freedesktop.NetworkManager.Device";

  QDBusConnection bus = QDBusConnection::systemBus();

  // Get devices
  QDBusInterface nm(nm_service, nm_path, nm_iface, bus);
  QDBusMessage response = nm.call("GetDevices");
  QVariant first =  response.arguments().at(0);

  const QDBusArgument &args = first.value<QDBusArgument>();
  args.beginArray();
  while (!args.atEnd()) {
    QDBusObjectPath path;
    args >> path;

    // Get device type
    QDBusInterface device_props(nm_service, path.path(), props_iface, bus);
    QDBusMessage response = device_props.call("Get", device_service, "DeviceType");
    QVariant first =  response.arguments().at(0);
    QDBusVariant dbvFirst = first.value<QDBusVariant>();
    QVariant vFirst = dbvFirst.variant();
    uint device_type = vFirst.value<uint>();
    qDebug() << path.path() << device_type;
  }
  args.endArray();


  // Add connection
  Connection connection;
  connection["connection"]["type"] = "802-11-wireless";
  connection["connection"]["uuid"] = QUuid::createUuid().toString().remove('{').remove('}');
  connection["connection"]["id"] = "Connection 1";

  connection["802-11-wireless"]["ssid"] = QByteArray("<ssid>");
  connection["802-11-wireless"]["mode"] = "infrastructure";

  connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
  connection["802-11-wireless-security"]["auth-alg"] = "open";
  connection["802-11-wireless-security"]["psk"] = "<password>";

  connection["ipv4"]["method"] = "auto";
  connection["ipv6"]["method"] = "ignore";


  QDBusInterface nm_settings(nm_service, nm_settings_path, nm_settings_iface, bus);
  QDBusReply<QDBusObjectPath> result = nm_settings.call("AddConnection", QVariant::fromValue(connection));
  if (!result.isValid()) {
    qDebug() << result.error().name() << result.error().message();
  } else {
    qDebug() << result.value().path();
  }
}
