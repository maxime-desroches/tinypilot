#include <cassert>
#include <iostream>

#include <QFile>
#include <QDebug>
#include <QVBoxLayout>
#include <QLabel>
#include <QStackedLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkRequest>

#include "drive_stats.hpp"
#include "common/params.h"
#include "common/utilpp.h"
#include "api.hpp"

constexpr double MILE_TO_KM = 1.60934;


#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif


QLayout *build_stat(QString name, int stat) {
  QVBoxLayout *layout = new QVBoxLayout;

  QLabel *metric = new QLabel(QString("%1").arg(stat));
  metric->setStyleSheet(R"(
    font-size: 72px;
    font-weight: 700;
  )");
  layout->addWidget(metric, 0, Qt::AlignLeft);
  
  QLabel *label = new QLabel(name);
  label->setStyleSheet(R"(
    font-size: 32px;
    font-weight: 600;
  )");
  layout->addWidget(label, 0, Qt::AlignLeft);
  
  return layout;
}

void DriveStats::replyFinished() {
  QString answer = reply->readAll();
  answer.chop(1);

  QJsonDocument doc = QJsonDocument::fromJson(answer.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting past drives statistics";
    reply->deleteLater();
    reply = NULL;
    return;
  }

  QString IsMetric = QString::fromStdString(Params().get("IsMetric"));
  bool metric = (IsMetric =="1");

  QJsonObject json = doc.object();
  auto all = json["all"].toObject();
  auto week = json["week"].toObject();

  QGridLayout *gl = new QGridLayout();

  int all_distance = all["distance"].toDouble() * (metric ? MILE_TO_KM : 1);
  gl->addWidget(new QLabel("ALL TIME"), 0, 0, 1, 3);
  gl->addLayout(build_stat("DRIVES", all["routes"].toDouble()), 1, 0, 3, 1);
  gl->addLayout(build_stat(metric ? "KM" : "MILES", all_distance), 1, 1, 3, 1);
  gl->addLayout(build_stat("HOURS", all["minutes"].toDouble() / 60), 1, 2, 3, 1);

  int week_distance = week["distance"].toDouble() * (metric ? MILE_TO_KM : 1);
  gl->addWidget(new QLabel("PAST WEEK"), 6, 0, 1, 3);
  gl->addLayout(build_stat("DRIVES", week["routes"].toDouble()), 7, 0, 3, 1);
  gl->addLayout(build_stat(metric ? "KM" : "MILES", week_distance), 7, 1, 3, 1);
  gl->addLayout(build_stat("HOURS", week["minutes"].toDouble() / 60), 7, 2, 3, 1);

  QWidget *q = new QWidget;
  q->setLayout(gl);

  slayout->addWidget(q);
  slayout->setCurrentIndex(1);

  reply->deleteLater();
  reply = NULL;
}

DriveStats::DriveStats(QWidget *parent) : QWidget(parent) {
  api = new CommaApi(this);

  QString dongle_id = QString::fromStdString(Params().get("DongleId"));

  QVector<QPair<QString, QJsonValue>> payloads;
  payloads.push_back(qMakePair(QString("identity"), dongle_id));
  QString token = api->create_jwt(payloads);

  QNetworkRequest request;
  request.setUrl(QUrl("https://api.commadotai.com/v1.1/devices/" + dongle_id + "/stats"));
  request.setRawHeader("Authorization", ("JWT "+token).toUtf8());
  if(reply == NULL){
    reply = api->get(request);
    connect(reply, &QNetworkReply::finished, this, &DriveStats::replyFinished);
  }else{
    qDebug()<<"Too many requests, previous request was not yet removed";
  }

  slayout = new QStackedLayout;

  slayout->addWidget(new QLabel("No network connection"));
  
  setLayout(slayout);
  setStyleSheet(R"(
    QLabel {
      font-size: 48px;
      font-weight: 600;
    }
  )");
}
