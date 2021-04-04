#include "updater.hpp"

#include <sys/stat.h>
#include <sys/statvfs.h>

#include <QApplication>
#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTimer>
#include <QVBoxLayout>
#include <fstream>
#include "common/util.h"
#include "qt_window.hpp"

#define USER_AGENT "NEOSUpdater-0.2"

#define MANIFEST_URL_NEOS_STAGING "https://github.com/commaai/eon-neos/raw/master/update.staging.json"
#define MANIFEST_URL_NEOS_LOCAL "http://192.168.5.1:8000/neosupdate/update.local.json"
#define MANIFEST_URL_NEOS "https://github.com/commaai/eon-neos/raw/master/update.json"
const char *manifest_url = MANIFEST_URL_NEOS;

#define RECOVERY_DEV "/dev/block/bootdevice/by-name/recovery"
#define RECOVERY_COMMAND "/cache/recovery/command"

// #define UPDATE_DIR "/data/neoupdate"
#define UPDATE_DIR "/home/deanlee/neoupdate"
#define MIN_BATTERY_CAP 35

QString sha256_file(const QString &fn, size_t limit = 0) {
  QCryptographicHash crypto(QCryptographicHash::Sha256);
  QFile file(fn);
  if (!file.open(QFile::ReadOnly)) return "";

  while (!file.atEnd()) {
    crypto.addData(file.read(8192));
  }
  return crypto.result().toHex();
}

size_t download_string_write(void *ptr, size_t size, size_t nmeb, void *up) {
  size_t sz = size * nmeb;
  ((QByteArray *)up)->append((const char *)ptr, sz);
  return sz;
}

QString download_string(CURL *curl, std::string url) {
  QByteArray os;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 0);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, USER_AGENT);
  curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1);
  curl_easy_setopt(curl, CURLOPT_RESUME_FROM, 0);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_string_write);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &os);
  CURLcode res = curl_easy_perform(curl);
  return res == CURLE_OK ? os : "";
}

size_t download_file_write(void *ptr, size_t size, size_t nmeb, void *up) {
  return fwrite(ptr, size, nmeb, (FILE *)up);
}

int battery_capacity() {
  std::string bat_cap_s = util::read_file("/sys/class/power_supply/battery/capacity");
  return atoi(bat_cap_s.c_str());
}

int battery_current() {
  std::string current_now_s = util::read_file("/sys/class/power_supply/battery/current_now");
  return atoi(current_now_s.c_str());
}

bool check_battery() {
  int bat_cap = battery_capacity();
  int current_now = battery_current();
  return bat_cap > 35 || (current_now < 0 && bat_cap > 10);
}

bool check_space() {
  return true;
  struct statvfs stat;
  if (statvfs("/data/", &stat) != 0) {
    return false;
  }
  size_t space = stat.f_bsize * stat.f_bavail;
  return space > 2000000000ULL;  // 2GB
}

static void start_settings_activity(const char *name) {
  char launch_cmd[1024];
  snprintf(launch_cmd, sizeof(launch_cmd),
           "am start -W --ez :settings:show_fragment_as_subsetting true -n 'com.android.settings/.%s'", name);
  system(launch_cmd);
}

// UpdaterThread

UpdaterThread::UpdaterThread(QObject *parent) : QThread(parent) {}

void UpdaterThread::checkBattery() {
  return;
  if (!check_battery()) {
    int battery_cap = 0;
    do {
      battery_cap = battery_capacity();
      emit lowBattery(battery_cap);
      util::sleep_for(1000);
    } while (battery_cap < MIN_BATTERY_CAP);
  }
}

void UpdaterThread::run() {
  curl = curl_easy_init();
  qInfo() << "run_stages start";

  checkBattery();

  // ** download update **
  if (!download_stage()) {
    return;
  }

  // ** install update **

  checkBattery();

  if (!recovery_fn.isEmpty()) {
    // flash recovery
    emit progressText("Flashing recovery...");

    std::ifstream src(recovery_fn.toStdString(), std::ios::binary);
    std::ofstream dest(RECOVERY_DEV, std::ios::binary);
    dest << src.rdbuf();
    if (!src || !dest) {
      emit error("failed to flash recovery: write failed");
      return;
    }

    emit progressText("Verifying flash...");
    QString new_recovery_hash = sha256_file(RECOVERY_DEV, recovery_len);
    qInfo() << "new recovery hash: " << new_recovery_hash;

    if (new_recovery_hash != recovery_hash) {
      emit error("recovery flash corrupted");
      return;
    }
  }

  // write arguments to recovery
  FILE *cmd_file = fopen(RECOVERY_COMMAND, "wb");
  if (!cmd_file) {
    emit error("failed to reboot into recovery");
    return;
  }
  fprintf(cmd_file, "--update_package=%s\n", ota_fn.toStdString().c_str());
  fclose(cmd_file);

  emit progressText("Rebooting");

  // remove the continue.sh so we come back into the setup.
  // maybe we should go directly into the installer, but what if we don't come back with internet? :/
  //unlink("/data/data/com.termux/files/continue.sh");

  // TODO: this should be generic between android versions
  // IPowerManager.reboot(confirm=false, reason="recovery", wait=true)
  system("service call power 16 i32 0 s16 recovery i32 1");
  while (1) pause();

  // execl("/system/bin/reboot", "recovery");
  // set_error("failed to reboot into recovery");
}

bool UpdaterThread::download_stage() {
  // ** quick checks before download **
  if (!check_space()) {
    emit error("2GB of free space required to update");
    return false;
  }

  mkdir(UPDATE_DIR, 0777);

  emit progressText("Finding latest version...");
  QString manifest_s = download_string(curl, manifest_url);
  qInfo() << "manifest: " << manifest_s;

  // parse maiifest
  QJsonDocument doc = QJsonDocument::fromJson(manifest_s.toUtf8());
  if (doc.isNull()) {
    emit error("failed to load update manifest");
    return false;
  }
  QJsonObject json = doc.object();
  QString ota_url = json["ota_url"].toString();
  QString ota_hash = json["ota_hash"].toString();
  QString recovery_url = json["recovery_url"].toString();
  recovery_hash = json["recovery_hash"].toString();
  recovery_len = json["recovery_len"].toInt();

  if (ota_url.isEmpty() || ota_hash.isEmpty()) {
    emit error("invalid update manifest");
    return false;
  }

  // ** handle recovery download **
  if (recovery_url.isEmpty() || recovery_hash.isEmpty() || recovery_len == 0) {
    emit progressText("Skipping recovery flash...");
  } else {
    // only download the recovery if it differs from what's flashed
    emit progressText("Checking recovery...");
    QString existing_recovery_hash = sha256_file(RECOVERY_DEV, recovery_len);
    qInfo() << "existing recovery hash: " << existing_recovery_hash;

    if (existing_recovery_hash != recovery_hash) {
      recovery_fn = download(recovery_url, recovery_hash, "recovery");
      if (recovery_fn.isEmpty()) {
        // error'd
        return false;
      }
    }
  }

  // ** handle ota download **
  ota_fn = download(ota_url, ota_hash, "update");
  return !ota_fn.isEmpty();
}

int UpdaterThread::download_file_xferinfo(curl_off_t dltotal, curl_off_t dlno, curl_off_t ultotal, curl_off_t ulnow) {
  static float prev_val = 0;
  if (dltotal != 0) {
    float progress_frac = ((float)dlno / dltotal) * 100;
    if ((progress_frac - prev_val) > 0.5) {
      emit progressPos(progress_frac);
    }
  }
  return 0;
};

bool UpdaterThread::download_file(const QString &url, const QString &out_fn) {
  FILE *of = fopen(out_fn.toStdString().c_str(), "ab");
  assert(of);
  fseek(of, 0, SEEK_END);

  int tries = 4;
  bool ret = false;
  long last_resume_from = 0;
  std::string url_string = url.toStdString();
  while (true) {
    long resume_from = ftell(of);

    curl_easy_setopt(curl, CURLOPT_URL, url_string.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 0);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, USER_AGENT);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1);
    curl_easy_setopt(curl, CURLOPT_RESUME_FROM, resume_from);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_file_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, of);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, this);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, &UpdaterThread::download_file_xferinfo);

    CURLcode res = curl_easy_perform(curl);

    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    // double content_length = 0.0;
    // curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &content_length);

    qInfo() << QString("download %1 res %2, code %3, resume from %4\n").arg(url).arg(res).arg(response_code).arg(resume_from);
    if (res == CURLE_OK) {
      ret = true;
      break;
    } else if (res == CURLE_HTTP_RETURNED_ERROR && response_code == 416) {
      // failed because the file is already complete?
      ret = true;
      break;
    } else if (resume_from == last_resume_from) {
      // failed and dind't make make forward progress. only retry a couple times
      tries--;
      if (tries <= 0) {
        break;
      }
    }
    last_resume_from = resume_from;
  }

  fclose(of);
  return ret;
}

QString UpdaterThread::download(const QString &url, const QString &hash, const QString &name) {
  QFileInfo fi(url);
  QString out_fn = UPDATE_DIR "/" + fi.fileName();

  // start or resume downloading if hash doesn't match
  QString fn_hash = sha256_file(out_fn);
  if (hash.compare(fn_hash) != 0) {
    emit progressText(QString("Downloading %1 ...").arg(name));
    bool r = download_file(url, out_fn);
    if (!r) {
      emit error(QString("failed to download %1").arg(name));
      unlink(out_fn.toStdString().c_str());
      return "";
    }
    fn_hash = sha256_file(out_fn);
  };
  emit progressText(QString("Verifying %1 ...").arg(name));
  qInfo() << QString("got %1 hash: %2").arg(name).arg(hash);
  if (fn_hash != hash) {
    emit error(QString("%1 was corrupt").arg(name));
    unlink(out_fn.toStdString().c_str());
    return "";
  }
  return out_fn;
}

// UpdateWindow

QLabel *textLabel(const QString &text, int font_size) {
  QLabel *label = new QLabel(text);
  label->setAlignment(Qt::AlignHCenter);
  label->setWordWrap(true);
  label->setStyleSheet(QString("font-size:%1px;").arg(font_size));
  return label;
}

UpdaterWidnow::UpdaterWidnow(QWidget *parent) : thread(this), QStackedWidget(parent) {
  setContentsMargins(200, 150, 200, 150);
  addWidget(confirmationPage());
  addWidget(progressPage());
  addWidget(batteryPage());
  addWidget(errPage());

  setStyleSheet(R"(
    * {
      font-family: Inter;
      color: white;
      background-color: black;
    }
    QPushButton {
      padding: 50px;
      padding-right: 100px;
      padding-left: 100px;
      border: 7px solid white;
      border-radius: 20px;
      font-size: 50px;
    }
    QProgressBar {
      background-color: #373737;
      width: 1000px;
      border solid white;
      border-radius: 10px;
    }
    QProgressBar::chunk {
      border-radius: 10px;
      background-color: white;
    }
  )");

  connect(&thread, &UpdaterThread::progressText, [=](const QString &text) {
    progressTitle->setText(text);
    setCurrentIndex(1);
  });
  connect(&thread, &UpdaterThread::progressPos, [=](int pos) {
    setCurrentIndex(1);
    progress_bar->setValue(pos);
  });
  connect(&thread, &UpdaterThread::error, [=](const QString &text) {
    setCurrentIndex(3);
    errLabel->setText(text);
  });
  connect(&thread, &UpdaterThread::lowBattery, [=](int battery_cap) {
    setCurrentIndex(2);
    batteryContext->setText(QString("Current battery charge: %1 %").arg(battery_cap));
  });
}

UpdaterWidnow::~UpdaterWidnow() {
  thread.exit();
}

QWidget *UpdaterWidnow::confirmationPage() {
  QWidget *w = new QWidget();
  QVBoxLayout *vl = new QVBoxLayout(w);
  vl->addWidget(textLabel("An update to NEOS is required.", 80));
  vl->addStretch();
  vl->addWidget(textLabel("Your device will now be reset and upgraded. You may want to connect to wifi as download is around 1 GB. Existing data on device should not be lost.", 50));
  vl->addStretch();

  // buttons
  QHBoxLayout *btnLayout = new QHBoxLayout();
  btnLayout->setSpacing(100);
  QPushButton *wifiBtn = new QPushButton("Connect to WiFi");
  QObject::connect(wifiBtn, &QPushButton::released, [=] {
    start_settings_activity("Settings$WifiSettingsActivity");
  });
  btnLayout->addWidget(wifiBtn);

  QPushButton *continueBtn = new QPushButton("Continue");
  QObject::connect(continueBtn, &QPushButton::released, [=] {
    setCurrentIndex(1);
    thread.start();
  });
  btnLayout->addWidget(continueBtn);

  vl->addLayout(btnLayout);
  return w;
}

QWidget *UpdaterWidnow::progressPage() {
  QWidget *w = new QWidget();
  QVBoxLayout *vl = new QVBoxLayout(w);
  progressTitle = textLabel("", 80);
  vl->addWidget(progressTitle);
  vl->addStretch();

  progress_bar = new QProgressBar();
  progress_bar->setRange(5, 100);
  progress_bar->setTextVisible(false);
  progress_bar->setFixedHeight(25);
  vl->addWidget(progress_bar);
  vl->addStretch();

  QHBoxLayout *btnLayout = new QHBoxLayout();
  btnLayout->setSpacing(100);
  QPushButton *wifiBtn = new QPushButton("Connect to WiFi");
  btnLayout->addWidget(wifiBtn);
  QLabel *desc = new QLabel("Ensure your device remains connected to a power source.");
  desc->setAlignment(Qt::AlignHCenter);
  desc->setWordWrap(true);
  desc->setStyleSheet("font-size:50px;");
  vl->addWidget(desc);
  vl->addStretch();
  return w;
}

QWidget *UpdaterWidnow::batteryPage() {
  QWidget *w = new QWidget();
  QVBoxLayout *vl = new QVBoxLayout(w);
  vl->addWidget(textLabel("Low Battery", 80));
  vl->addStretch();
  vl->addWidget(textLabel("Please connect EON to your charger. Update will continue once EON battery reaches 35%.", 50));
  vl->addStretch();
  batteryContext = textLabel("", 50);
  vl->addWidget(batteryContext);
  vl->addStretch();
  return w;
}

QWidget *UpdaterWidnow::errPage() {
  QWidget *w = new QWidget();
  QVBoxLayout *vl = new QVBoxLayout(w);
  vl->addWidget(textLabel("There was an error", 80));
  vl->addStretch();

  errLabel = textLabel("", 50);
  vl->addWidget(errLabel);
  vl->addStretch();

  QHBoxLayout *btnLayout = new QHBoxLayout();
  btnLayout->setAlignment(Qt::AlignRight);
  btnLayout->setSpacing(100);
  QPushButton *rebootBtn = new QPushButton("Reboot");
  QObject::connect(rebootBtn, &QPushButton::released, [=] {
    // reboot
    system("service call power 16 i32 0 i32 0 i32 1");
  });
  btnLayout->addWidget(rebootBtn);
  vl->addLayout(btnLayout);
  return w;
}

int main(int argc, char *argv[]) {
  bool background_cache = false;
  if (argc > 1) {
    if (strcmp(argv[1], "local") == 0) {
      manifest_url = MANIFEST_URL_NEOS_LOCAL;
    } else if (strcmp(argv[1], "staging") == 0) {
      manifest_url = MANIFEST_URL_NEOS_STAGING;
    } else if (strcmp(argv[1], "bgcache") == 0) {
      manifest_url = argv[2];
      background_cache = true;
    } else {
      manifest_url = argv[1];
    }
  }

  qInfo() << "updating from " << manifest_url;

  QApplication a(argc, argv);
  UpdaterWidnow updater;
  setMainWindow(&updater);
  return a.exec();
}
