#include <QHBoxLayout>
#include "widgets/input.h"
#include "widgets/ssh_keys.h"
#include "api.hpp"
#include "common/params.h"


SshControl::SshControl() : AbstractControl("SSH Keys", "Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username other than your own. A comma employee will NEVER ask you to add their GitHub username.", "") {

  // setup widget
  hlayout->addStretch(1);

  username_label.setAlignment(Qt::AlignVCenter);
  username_label.setStyleSheet("color: #aaaaaa");
  hlayout->addWidget(&username_label);

  btn.setStyleSheet(R"(
    padding: 0;
    border-radius: 50px;
    font-size: 35px;
    font-weight: 500;
    color: #E4E4E4;
    background-color: #393939;
  )");
  btn.setFixedSize(250, 100);
  hlayout->addWidget(&btn);

  QObject::connect(&btn, &QPushButton::released, [=]() {
    if (btn.text() == "ADD") {
      QString username = InputDialog::getText("Enter your GitHub username");
      if (username.length() > 0) {
        btn.setText("LOADING");
        btn.setEnabled(false);
        getUserKeys(username);
      }
    } else {
      params.remove("GithubUsername");
      params.remove("GithubSshKeys");
      refresh();
    }
  });

  refresh();
}

void SshControl::refresh() {
  QString param = QString::fromStdString(params.get("GithubSshKeys"));
  if (param.length()) {
    username_label.setText(QString::fromStdString(params.get("GithubUsername")));
    btn.setText("REMOVE");
  } else {
    username_label.setText("");
    btn.setText("ADD");
  }
  btn.setEnabled(true);
}

void SshControl::getUserKeys(const QString &username) {
  QEventLoop loop;
  HttpRequest request(this, "https://github.com/" + username + ".keys", "", false);

  QObject::connect(&request, &HttpRequest::receivedResponse, [&](const QString &resp) {
    loop.quit();
    if (!resp.isEmpty()) {
      Params params;
      params.put("GithubUsername", username.toStdString());
      params.put("GithubSshKeys", resp.toStdString());
    } else {
      ConfirmationDialog::alert("Username '" + username + "' has no keys on GitHub");
    }
  });
  QObject::connect(&request, &HttpRequest::failedResponse, [&] {
    loop.quit();
    ConfirmationDialog::alert("Username '" + username + "' doesn't exist on GitHub");
  });
  QObject::connect(&request, &HttpRequest::timeoutResponse, [&] {
    loop.quit();
    ConfirmationDialog::alert("Request timed out");
  });

  loop.exec();
  refresh();
}
