#pragma once

#include <QtDBus>
#include <QWidget>

enum class SecurityType {
  OPEN,
  WPA,
  UNSUPPORTED
};
enum class ConnectedType{
  DISCONNECTED,
  CONNECTING,
  CONNECTED
};

typedef QMap<QString, QMap<QString, QVariant>> Connection;
typedef QVector<QMap<QString, QVariant>> IpConfig;

struct Network {
  QString path;
  QByteArray ssid;
  unsigned int strength;
  ConnectedType connected;
  SecurityType security_type;
  bool known;
};

class WifiManager : public QWidget {
  Q_OBJECT
public:
  explicit WifiManager(QWidget* parent);

  void requestScan();
  QVector<Network> seenNetworks;
  QString ip4Address;

  void refreshNetworks();
  void forgetConnection(const QString &ssid);
  bool isKnownConnection(const QString &ssid);

  void connect(const Network &ssid);
  void connect(const Network &ssid, const QString &password);
  void connect(const Network &ssid, const QString &username, const QString &password);
  void disconnect();

  // Tethering functions
  void enableTethering();
  void disableTethering();
  bool tetheringEnabled();

  void addTetheringConnection();
  void activateWifiConnection(const QString &ssid);
  void changeTetheringPassword(const QString &newPassword);

private:
  QMap<QDBusObjectPath, QString> knownConnections;

  QVector<QByteArray> seen_ssids;
  QString adapter;  // Path to network manager wifi-device
  QDBusConnection bus = QDBusConnection::systemBus();
  unsigned int raw_adapter_state;  // Connection status https://developer.gnome.org/NetworkManager/1.26/nm-dbus-types.html#NMDeviceState
  QString connecting_to_network;
  QString tethering_ssid;
  QString tetheringPassword = "swagswagcommma";

  QString get_adapter();
  QString getIp4Address(const QDBusObjectPath &path);
  void connect(const QByteArray &ssid, const QString &username, const QString &password, SecurityType security_type);
  QString get_active_ap();
  void deactivateConnection(const QString &ssid);
  QVector<QDBusObjectPath> get_active_connections();
  uint get_wifi_device_state();
  QByteArray getApProperty(const QString &network_path, const QString &property);
  unsigned int get_ap_strength(const QString &network_path);
  SecurityType getSecurityType(const QString &path);
  QDBusObjectPath getConnectionPath(const QString &ssid);
  QMap<QDBusObjectPath, QString> listConnections();
  QString getConnectionSsid(const QDBusObjectPath &path);

signals:
  void wrongPassword(const QString &ssid);
  void refreshSignal();

private slots:
  void stateChange(unsigned int new_state, unsigned int previous_state, unsigned int change_reason);
  void propertyChange(const QString &interface, const QVariantMap &props, const QStringList &invalidated_props);
  void connectionRemoved(const QDBusObjectPath &path);
  void newConnection(const QDBusObjectPath &path);
  void accessPointAdded(const QDBusObjectPath &path);
  void accessPointRemoved(const QDBusObjectPath &path);
};
