#pragma once

#include <QCryptographicHash>
#include <QJsonValue>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPair>
#include <QString>
#include <QVector>
#include <QWidget>

#include <atomic>
#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>

class CommaApi : public QObject {
  Q_OBJECT

public:
  static QByteArray rsa_sign(QByteArray data);
  static QString create_jwt(const std::map<QString, QJsonValue> *payloads = nullptr, int expiry=3600);

private:
  QNetworkAccessManager* networkAccessManager;
};

/**
 * Makes repeated requests to the request endpoint.
 */
class RequestRepeater : public QObject {
  Q_OBJECT

public:
  explicit RequestRepeater(QWidget* parent, QString requestURL, int period = 10, const std::map<QString, QJsonValue> *payloads = nullptr, bool disableWithScreen = true);
  bool active = true;

private:
  bool disableWithScreen;
  QNetworkReply* reply;
  QNetworkAccessManager* networkAccessManager;
  QTimer* networkTimer;
  std::atomic<bool> aborted = false; // Not 100% sure we need atomic
  void sendRequest(QString requestURL, const std::map<QString, QJsonValue>* payloads);

private slots:
  void requestTimeout();
  void requestFinished();

signals:
  void receivedResponse(QString response);
  void failedResponse(QString errorString);
};
