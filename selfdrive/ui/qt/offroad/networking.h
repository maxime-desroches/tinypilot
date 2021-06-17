#pragma once

#include <QButtonGroup>
#include <QPushButton>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QWidget>

#include "selfdrive/ui/qt/offroad/wifiManager.h"
#include "selfdrive/ui/qt/widgets/input.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"
#include "selfdrive/ui/qt/widgets/toggle.h"

class NetworkStrengthWidget : public QWidget {
  Q_OBJECT

public:
  explicit NetworkStrengthWidget(int strength, QWidget* parent = nullptr) : strength_(strength), QWidget(parent) { setFixedSize(100, 15); }

private:
  void paintEvent(QPaintEvent* event) override;
  int strength_ = 0;
};

class WifiUI : public QWidget {
  Q_OBJECT

public:
  explicit WifiUI(QWidget *parent = 0);
  void refresh(QVector<Network> _seen_networks);

private:
  QVBoxLayout* main_layout;

  QButtonGroup *connectButtons;
  bool tetheringEnabled;

  QVector<Network> seen_networks;

signals:
  void connectToNetwork(const Network n, const QString pass);

public slots:
  void handleButton(QAbstractButton* m_button);
};

class AdvancedNetworking : public QWidget {
  Q_OBJECT
public:
  explicit AdvancedNetworking(QWidget* parent = 0);
  void refresh(const QString ipv4_address);
  ToggleControl *tetheringToggle;

private:
  LabelControl* ipLabel;
  ButtonControl* editPasswordButton;

signals:
  void backPress();

  // WifiThread signals
  void toggleTetheringSignal(const bool enabled);
  void changeTetheringPassword(const QString newPassword);

public slots:
  void toggleTethering(bool enable);
};

class Networking : public QWidget {
  Q_OBJECT
  QThread wifiThread;

public:
  explicit Networking(QWidget* parent = 0);
  ~Networking() {  // TODO
    wifiThread.quit();
    wifiThread.wait();
  }

private:
  WifiManager* wifiManager = nullptr;

  QStackedLayout* main_layout = nullptr; // nm_warning, wifiScreen, advanced
  QWidget* wifiScreen = nullptr;
  AdvancedNetworking* an = nullptr;
  bool ui_setup_complete = false;

  Network selectedNetwork;

  WifiUI* wifiWidget;
  void attemptInitialization();

signals:
  void connectToNetwork(const Network n, const QString pass);
  void refreshWifiManager();

public slots:
  void refresh(const QVector<Network> seen_networks, const QString ipv4_address);
  void wrongPassword(const Network n);

};
