#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QLineEdit>
#include <QRandomGenerator>

#include "wifi.hpp"
#include "widgets/toggle.hpp"

void clearLayout(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()) {
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      clearLayout(childLayout);
    }
    delete item;
  }
}

QWidget* layoutToWidget(QLayout* l, QWidget* parent){
  QWidget* q = new QWidget(parent);
  q->setLayout(l);
  return q;
}
//==================================================================================================================================================///

Networking::Networking(QWidget* parent){
  try {
    wifi = new WifiManager(this);
  } catch (std::exception &e) {
    QLabel* warning = new QLabel("Network manager is inactive!");
    warning->setStyleSheet(R"(font-size: 65px;)");

    QVBoxLayout* warning_layout = new QVBoxLayout;
    warning_layout->addWidget(warning, 0, Qt::AlignCenter);
    setLayout(warning_layout);
    return;
  }
  connect(wifi, SIGNAL(wrongPassword(QString)), this, SLOT(wrongPassword(QString)));
  connect(wifi, SIGNAL(successfulConnection(QString)), this, SLOT(successfulConnection(QString)));
  

  s = new QStackedLayout(this);

  inputField = new InputField();
  connect(inputField, SIGNAL(emitText(QString)), this, SLOT(receiveText(QString)));
  connect(inputField, SIGNAL(cancel()), this, SLOT(abortTextInput()));
  s->addWidget(inputField);

  QVBoxLayout* vlayout = new QVBoxLayout(this);
  QPushButton* advancedSettings = new QPushButton("Advanced");
  advancedSettings->setStyleSheet(R"(margin-right: 30px)");
  advancedSettings->setFixedSize(300, 100);
  connect(advancedSettings, &QPushButton::released, [=](){s->setCurrentIndex(2);});
  vlayout->addSpacing(10);
  vlayout->addWidget(advancedSettings, 0, Qt::AlignRight);
  vlayout->addSpacing(10);

  wifiWidget = new WifiUI(this, 5, wifi);
  connect(wifiWidget, SIGNAL(connectToNetwork(Network)), this, SLOT(connectToNetwork(Network)));
  vlayout->addWidget(wifiWidget, 1);

  s->addWidget(layoutToWidget(vlayout, this));

  AdvancedNetworking* an = new AdvancedNetworking(this, wifi);
  connect(an, &AdvancedNetworking::backPress, [=](){s->setCurrentIndex(1);});
  s->addWidget(an);

  s->setCurrentIndex(1);


  // Update network status
  QTimer* timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  timer->start(2000);
  state = NetworkingState::IDLE;
  
  setStyleSheet(R"(
    QPushButton {
      background-color: #393939;
    }
    QPushButton:disabled {
      background-color: #000000;
    }
  )");
}

void Networking::refresh(){
  if(s->currentIndex() == 0 && state == NetworkingState::IDLE){
    qDebug()<<"Running text input on idle state. That shouldn't be possible";
  }
  wifiWidget->refresh();
}

void Networking::connectToNetwork(Network n) {
  qDebug()<<"Connecting to network"<<n.ssid;
  if(state == NetworkingState::CONNECTING_TO_WIFI_NETWORK){
    qDebug()<<"Killing existing connection";
    wifi->disconnect();
  }
  if (n.security_type == SecurityType::OPEN) {
    wifi->connect(n);
  } else if (n.security_type == SecurityType::WPA) {
    inputField->setPromptText("Enter password for \"" + n.ssid + "\"");
    s->setCurrentIndex(0);
    state = NetworkingState::CONNECTING_TO_WIFI_NETWORK;
    selectedNetwork = n;
  }
}

void Networking::abortTextInput(){
  qDebug()<<"User stopped providing text, aborting connecting";
  state = NetworkingState::IDLE;
  s->setCurrentIndex(1);
}

void Networking::receiveText(QString text) {
  qDebug()<<"got text"<<text;
  if(state != NetworkingState::CONNECTING_TO_WIFI_NETWORK){
    qDebug()<<"Logic error. Recevied some text while not connecting:"<<text;
    return;
  }
  if(text.size()<8){
    qDebug()<<"Password was too short";
    state = NetworkingState::IDLE;
    s->setCurrentIndex(1);
    return;
  }
  wifi->connect(selectedNetwork, text);
  s->setCurrentIndex(1);
}

void Networking::wrongPassword(QString ssid) {
  qDebug()<<"Wrong password for"<<ssid;
  if(state != NetworkingState::CONNECTING_TO_WIFI_NETWORK){
    qDebug()<<"Logic error, got wrong password while not connecing. Wrong password for"<<ssid;
    return;
  }

  for (Network n : wifi->seen_networks) {
    if (n.ssid == ssid) {
      inputField->setPromptText("Wrong password for \"" + n.ssid +"\"");
      s->setCurrentIndex(0);
      return;
    }
  }
  qDebug()<<"Network we just provided the wrong password to doesn't seem to exist...";

}

void Networking::successfulConnection(QString ssid) {
  qDebug()<<"Success for"<<ssid;
  if(state != NetworkingState::CONNECTING_TO_WIFI_NETWORK){
    qDebug()<<"Logic error, got successfulConnection while not connecing. Success for"<<ssid;
    return;
  }

  for (Network n : wifi->seen_networks) {
    if (n.ssid == ssid) {
      state = NetworkingState::IDLE;
      return;
    }
  }
  qDebug()<<"Network we just connected to doesn't seem to exist...";
}

//=====================================================================================================================================================//

AdvancedNetworking::AdvancedNetworking(QWidget* parent, WifiManager* wifi): QWidget(parent), wifi(wifi){
  s = new QStackedLayout(this);// inputField and settings
  inputField = new InputField();
  connect(inputField, SIGNAL(emitText(QString)), this, SLOT(receiveText(QString)));
  connect(inputField, SIGNAL(cancel()), this, SLOT(abortTextInput()));
  s->addWidget(inputField);

  QVBoxLayout* vlayout = new QVBoxLayout(this);
  
  //Back button
  QPushButton* back = new QPushButton("BACK");
  connect(back, &QPushButton::released, [=](){emit backPress();});
  vlayout->addWidget(back, 0, Qt::AlignLeft);

  //Enable tethering layout
  QHBoxLayout* tetheringToggleLayout = new QHBoxLayout(this);
  tetheringToggleLayout->addWidget(new QLabel("Enable tethering", this));
  Toggle* toggle_switch = new Toggle(this);
  toggle_switch->setFixedSize(150, 100);
  tetheringToggleLayout->addWidget(toggle_switch);
  if (wifi->tetheringEnabled()) {
    toggle_switch->togglePosition();
  }
  QObject::connect(toggle_switch, SIGNAL(stateChanged(int)), this, SLOT(toggleTethering(int)));
  vlayout->addWidget(layoutToWidget(tetheringToggleLayout, this));

  //Change tethering password
  QHBoxLayout *tetheringPassword = new QHBoxLayout(this);
  tetheringPassword->addWidget(new QLabel("Edit tethering password", this));
  QPushButton* editPasswordButton = new QPushButton("EDIT", this);
  tetheringPassword->addWidget(editPasswordButton);
  vlayout->addWidget(layoutToWidget(tetheringPassword, this));

  //IP adress
  QHBoxLayout* IPlayout = new QHBoxLayout(this);
  IPlayout->addWidget(new QLabel("IP address: "));
  IPlayout->addWidget(new QLabel(wifi->ipv4_address));
  vlayout->addWidget(layoutToWidget(IPlayout, this));

  //Enable SSH
  QHBoxLayout* enableSSHLayout = new QHBoxLayout(this);
  enableSSHLayout->addWidget(new QLabel("Enable SSH", this));
  Toggle* toggle_switch_SSH = new Toggle(this);
  toggle_switch_SSH->setFixedSize(150, 100);
  enableSSHLayout->addWidget(toggle_switch_SSH);
  vlayout->addWidget(layoutToWidget(enableSSHLayout, this));

  //Authorized SSH keys
  QHBoxLayout* authSSHLayout = new QHBoxLayout(this);
  authSSHLayout->addWidget(new QLabel("Authorized SSH keys", this));
  QPushButton* editAuthSSHButton = new QPushButton("EDIT", this);
  authSSHLayout->addWidget(editAuthSSHButton);
  vlayout->addWidget(layoutToWidget(authSSHLayout, this));

  //Disconnect or delete connections
  QHBoxLayout* dangerZone = new QHBoxLayout(this);
  QPushButton* disconnect = new QPushButton("Disconnect from WiFi", this);
  dangerZone->addWidget(disconnect);
  QPushButton* deleteAll = new QPushButton("DELETE ALL NETWORKS", this);
  dangerZone->addWidget(deleteAll);
  vlayout->addWidget(layoutToWidget(dangerZone, this));

  //vlayout to widget and stylesheet
  QWidget* settingsWidget = layoutToWidget(vlayout, this);
  s->addWidget(settingsWidget);
  s->setCurrentIndex(1);
  setLayout(s);
}

void AdvancedNetworking::toggleTethering(int enable) {
  if (enable) {
    wifi->enableTethering();
  } else {
    wifi->disableTethering();
  }
}

void AdvancedNetworking::receiveText(QString text){
  qDebug()<<"Advanced text:"<<text;
}
void AdvancedNetworking::abortTextInput(){
  qDebug()<<"User aborted text input";
}

//=====================================================================================================================================================//

WifiUI::WifiUI(QWidget *parent, int page_length, WifiManager* wifi) : QWidget(parent), networks_per_page(page_length), wifi(wifi) {
  vlayout = new QVBoxLayout(this);
  setLayout(vlayout);

  // Scan on startup
  QLabel *scanning = new QLabel("Scanning for networks");
  scanning->setStyleSheet(R"(font-size: 65px;)");
  vlayout->addWidget(scanning, 0, Qt::AlignCenter);
  vlayout->setSpacing(25);

  page = 0;
  refresh();
}
// TODO check it looks nice with tethering
void WifiUI::refresh() {
  wifi->request_scan();
  wifi->refreshNetworks();
  clearLayout(vlayout);

  connectButtons = new QButtonGroup(this); // TODO check if this is a leak
  QObject::connect(connectButtons, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(handleButton(QAbstractButton*)));

  int i = 0;
  int countWidgets = 0;
  int button_height = static_cast<int>(this->height() / (networks_per_page + 1) * 0.6);
  for (Network &network : wifi->seen_networks) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    if (page * networks_per_page <= i && i < (page + 1) * networks_per_page) {
      // SSID
      hlayout->addSpacing(50);
      hlayout->addWidget(new QLabel(QString::fromUtf8(network.ssid)));

      // strength indicator
      unsigned int strength_scale = network.strength / 17;
      QPixmap pix("../assets/images/network_" + QString::number(strength_scale) + ".png");
      QLabel *icon = new QLabel();
      icon->setPixmap(pix.scaledToWidth(100, Qt::SmoothTransformation));
      icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
      hlayout->addWidget(icon);
      hlayout->addSpacing(20);

      // connect button
      QPushButton* btn = new QPushButton(network.connected == ConnectedType::CONNECTED ? "Connected" : (network.connected == ConnectedType::CONNECTING ? "Connecting" : "Connect"));
      btn->setFixedWidth(300);
      btn->setFixedHeight(button_height);
      btn->setDisabled(network.connected == ConnectedType::CONNECTED || network.connected == ConnectedType::CONNECTING || network.security_type == SecurityType::UNSUPPORTED);
      hlayout->addWidget(btn);
      hlayout->addSpacing(20);

      connectButtons->addButton(btn, i);

      QWidget * w = new QWidget;
      w->setLayout(hlayout);
      vlayout->addWidget(w);
      countWidgets++;
    }
    i++;
  }

  // Pad vlayout to prevert oversized network widgets in case of low visible network count
  for (int i = countWidgets; i < networks_per_page; i++) {
    QWidget *w = new QWidget;
    vlayout->addWidget(w);
  }

  QHBoxLayout *prev_next_buttons = new QHBoxLayout;
  QPushButton* prev = new QPushButton("Previous");
  prev->setEnabled(page);
  prev->setFixedHeight(button_height);
  
  QPushButton* next = new QPushButton("Next");
  next->setFixedHeight(button_height);

  // If there are more visible networks then we can show, enable going to next page
  next->setEnabled(wifi->seen_networks.size() > (page + 1) * networks_per_page);

  QObject::connect(prev, SIGNAL(released()), this, SLOT(prevPage()));
  QObject::connect(next, SIGNAL(released()), this, SLOT(nextPage()));
  prev_next_buttons->addWidget(prev);
  prev_next_buttons->addWidget(next);

  QWidget *w = new QWidget;
  w->setLayout(prev_next_buttons);
  vlayout->addWidget(w);
}

void WifiUI::handleButton(QAbstractButton* button) {
  QPushButton* btn = static_cast<QPushButton*>(button);
  Network n = wifi->seen_networks[connectButtons->id(btn)];
  emit connectToNetwork(n);
}

void WifiUI::prevPage() {
  page--;
  refresh();
}
void WifiUI::nextPage() {
  page++;
  refresh();
}
