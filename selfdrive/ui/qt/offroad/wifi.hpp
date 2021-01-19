#pragma once

#include <QWidget>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QTimer>

#include "wifiManager.hpp"
#include "widgets/input_field.hpp"


class WifiUI : public QWidget {
  Q_OBJECT

public:
  int page;
  explicit WifiUI(QWidget *parent = 0, int page_length = 5, WifiManager* wifi = 0);

private:
  WifiManager *wifi = nullptr;
  int networks_per_page;
  QVBoxLayout *vlayout;

  QButtonGroup *connectButtons;
  bool tetheringEnabled;

signals:
  void openKeyboard();
  void closeKeyboard();
  void connectToNetwork(Network n);

public slots:
  void handleButton(QAbstractButton* m_button);
  void refresh();

  void prevPage();
  void nextPage();
};

enum class NetworkingState {
  IDLE,
  CONNECTING_TO_WIFI_NETWORK,
};

class Networking : public QWidget {
  Q_OBJECT

public:
  explicit Networking(QWidget* parent = 0);

private:
  QStackedLayout* s;// keyboard, wifiScreen, advanced
  
  NetworkingState state;
  Network selectedNetwork;

  WifiUI* wifiWidget;
  WifiManager* wifi = nullptr;
  InputField* inputField;

signals:
  void openKeyboard();
  void closeKeyboard();

private slots:
  void connectToNetwork(Network n);
  void refresh();
  void receiveText(QString text);
  void abortTextInput();
  void wrongPassword(QString ssid);
  void successfulConnection(QString ssid);
};

class AdvancedNetworking : public QWidget {
  Q_OBJECT
public:
  explicit AdvancedNetworking(QWidget* parent = 0);

private:
  QStackedLayout* s;
  InputField* inputField;

signals:
  void openKeyboard();
  void closeKeyboard();
  void backPress();

private slots:
  void receiveText(QString text);
  void abortTextInput();
};
