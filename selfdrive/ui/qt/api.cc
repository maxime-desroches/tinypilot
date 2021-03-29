#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QString>
#include <QWidget>
#include <QTimer>
#include <QRandomGenerator>
#include <QEventLoop>
#include "api.hpp"
#include "home.hpp"
#include "common/params.h"
#include "common/util.h"

#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif

QByteArray CommaApi::rsa_sign(QByteArray data) {
  auto file = QFile(private_key_path.c_str());
  if (!file.open(QIODevice::ReadOnly)) {
    qDebug() << "No RSA private key found, please run manager.py or registration.py";
    return QByteArray();
  }
  auto key = file.readAll();
  file.close();
  file.deleteLater();
  BIO* mem = BIO_new_mem_buf(key.data(), key.size());
  assert(mem);
  RSA* rsa_private = PEM_read_bio_RSAPrivateKey(mem, NULL, NULL, NULL);
  assert(rsa_private);
  auto sig = QByteArray();
  sig.resize(RSA_size(rsa_private));
  unsigned int sig_len;
  int ret = RSA_sign(NID_sha256, (unsigned char*)data.data(), data.size(), (unsigned char*)sig.data(), &sig_len, rsa_private);
  assert(ret == 1);
  assert(sig_len == sig.size());
  BIO_free(mem);
  RSA_free(rsa_private);
  return sig;
}

QString CommaApi::create_jwt(QVector<QPair<QString, QJsonValue>> payloads, int expiry) {
  QString dongle_id = QString::fromStdString(Params().get("DongleId"));

  QJsonObject header;
  header.insert("alg", "RS256");

  QJsonObject payload;
  payload.insert("identity", dongle_id);

  auto t = QDateTime::currentSecsSinceEpoch();
  payload.insert("nbf", t);
  payload.insert("iat", t);
  payload.insert("exp", t + expiry);
  for (auto load : payloads) {
    payload.insert(load.first, load.second);
  }

  auto b64_opts = QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals;
  QString jwt = QJsonDocument(header).toJson(QJsonDocument::Compact).toBase64(b64_opts) + '.' +
                QJsonDocument(payload).toJson(QJsonDocument::Compact).toBase64(b64_opts);

  auto hash = QCryptographicHash::hash(jwt.toUtf8(), QCryptographicHash::Sha256);
  auto sig = rsa_sign(hash);
  jwt += '.' + sig.toBase64(b64_opts);
  return jwt;
}

QString CommaApi::create_jwt() {
  return create_jwt(*(new QVector<QPair<QString, QJsonValue>>()));
}

std::pair<QNetworkReply::NetworkError, QString> httpGet(const QString& url, int timeout_ms, QMap<QString, QString>* headers) {
  QNetworkRequest request(QUrl{url});
  if (headers) {
    for (auto it = headers->constBegin(); it != headers->constEnd(); ++it) {
      request.setRawHeader(it.key().toUtf8(), it.value().toUtf8());
    }
  }
#ifdef QCOM
  QSslConfiguration ssl = QSslConfiguration::defaultConfiguration();
  ssl.setCaCertificates(QSslCertificate::fromPath("/usr/etc/tls/cert.pem", QSsl::Pem, QRegExp::Wildcard));
  request.setSslConfiguration(ssl);
#endif

  QEventLoop loop;

  QNetworkAccessManager nam;
  QNetworkReply* reply = nam.get(request);
  QObject::connect(reply, SIGNAL(finished()), &loop, SLOT(quit()), Qt::DirectConnection);

  QTimer timer;
  timer.setSingleShot(true);
  QObject::connect(&timer, SIGNAL(timeout()), &loop, SLOT(quit()));
  timer.start(timeout_ms);

  loop.exec();

  QNetworkReply::NetworkError err = timer.isActive() ? QNetworkReply::TimeoutError : reply->error();
  return std::make_pair(err, reply->readAll());
}

RequestRepeater::RequestRepeater(QWidget* parent, QString requestURL, int period_seconds, const QString& cache_key, QVector<QPair<QString, QJsonValue>> payloads, bool disableWithScreen)
    : disableWithScreen(disableWithScreen), cache_key(cache_key), sending(false), QObject(parent) {
  if (!cache_key.isEmpty()) {
    if (std::string cached_resp = Params().get(cache_key.toStdString()); !cached_resp.empty()) {
      QTimer::singleShot(0, [=]() { emit receivedResponse(QNetworkReply::NoError, QString::fromStdString(cached_resp)); });
    }
  }

  sendRequest(requestURL, payloads);

  QTimer *repeatTimer = new QTimer(this);
  QObject::connect(repeatTimer, &QTimer::timeout, [=]() { sendRequest(requestURL, payloads); });
  repeatTimer->start(period_seconds * 1000);
}

void RequestRepeater::sendRequest(const QString& requestURL, QVector<QPair<QString, QJsonValue>> payloads) {
  if (GLWindow::ui_state.scene.started || !active || sending ||
      (!GLWindow::ui_state.awake && disableWithScreen)) {
    return;
  }

  sending = true;
  QMap<QString, QString> headers{{"Authorization", "JWT " + CommaApi::create_jwt(payloads)}};
  auto [err, resp] = httpGet(requestURL, 20000, &headers);
  if (!cache_key.isEmpty()) {
    if (err == QNetworkReply::NoError) {
      Params().write_db_value(cache_key.toStdString(), resp.toStdString());
    } else if (err != QNetworkReply::TimeoutError) {
      Params().delete_db_value(cache_key.toStdString());
    }
  }
  emit receivedResponse(err, resp);
  sending = false;
}
