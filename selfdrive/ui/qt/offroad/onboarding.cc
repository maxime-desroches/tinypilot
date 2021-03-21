#include <QLabel>
#include <QString>
#include <QScroller>
#include <QScrollBar>
#include <QPushButton>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QDesktopWidget>
#include <QPainter>

#include "common/params.h"
#include "onboarding.hpp"
#include "home.hpp"
#include "util.h"


void TrainingGuide::mouseReleaseEvent(QMouseEvent *e) {
  const QRect restartRect{QPoint(1050, 773), QPoint(1500, 954)};
  const int leftOffset = (geometry().width()-1620)/2;

  QPoint pt(e->x() - leftOffset, e->y());

  int prevIndex = currentIndex;
  // Check for restart
  if (currentIndex == (std::size(boundingBox) - 1) && restartRect.contains(pt)) {
    currentIndex = 0;
  } else if (boundingBox[currentIndex].contains(pt)) {
    currentIndex += 1;
  }

  if (currentIndex >= std::size(boundingBox)) {
    emit completedTraining();
  } else if (prevIndex != curentIndex) {
    image.load("../assets/training/step" + QString::number(currentIndex) + ".jpg");
    update();
  }
}

TrainingGuide::TrainingGuide(QWidget* parent) {
  image.load("../assets/training/step0.jpg");
}

void TrainingGuide::paintEvent(QPaintEvent *event) {
  QPainter painter(this);

  QRect devRect(0, 0, painter.device()->width(), painter.device()->height());
  QBrush bgBrush("#072339");
  painter.fillRect(devRect, bgBrush);

  QRect rect(image.rect());
  rect.moveCenter(devRect.center());
  painter.drawImage(rect.topLeft(), image);
}


QWidget* OnboardingWindow::terms_screen() {
  QVBoxLayout *main_layout = new QVBoxLayout;
  main_layout->setContentsMargins(40, 20, 40, 20);

  QString terms_html = QString::fromStdString(util::read_file("../assets/offroad/tc.html"));
  terms_text = new QTextEdit();
  terms_text->setReadOnly(true);
  terms_text->setTextInteractionFlags(Qt::NoTextInteraction);
  terms_text->setHtml(terms_html);
  main_layout->addWidget(terms_text);

  // TODO: add decline page
  QHBoxLayout* buttons = new QHBoxLayout;
  main_layout->addLayout(buttons);

  buttons->addWidget(new QPushButton("Decline"));
  buttons->addSpacing(50);

  QPushButton *accept_btn = new QPushButton("Scroll to accept");
  accept_btn->setEnabled(false);
  buttons->addWidget(accept_btn);
  QObject::connect(accept_btn, &QPushButton::released, [=]() {
    Params().write_db_value("HasAcceptedTerms", current_terms_version);
    updateActiveScreen();
  });

  // TODO: tune the scrolling
  auto sb = terms_text->verticalScrollBar();
#ifdef QCOM2
  sb->setStyleSheet(R"(
    QScrollBar {
      width: 150px;
      background: grey;
    }
    QScrollBar::handle {
      background-color: white;
    }
    QScrollBar::add-line, QScrollBar::sub-line{
      width: 0;
      height: 0;
    }
  )");
#else
  QScroller::grabGesture(terms_text, QScroller::TouchGesture);
  terms_text->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
#endif

  QObject::connect(sb, &QScrollBar::valueChanged, [sb, accept_btn]() {
    if (sb->value() == sb->maximum()){
      accept_btn->setText("Accept");
      accept_btn->setEnabled(true);
    }
  });

  QWidget *widget = new QWidget;
  widget->setLayout(main_layout);
  widget->setStyleSheet(R"(
    * {
      font-size: 50px;
    }
    QPushButton {
      padding: 50px;
      border-radius: 10px;
      background-color: #292929;
    }
  )");

  return widget;
}

void OnboardingWindow::updateActiveScreen() {
  bool accepted_terms = params.get("HasAcceptedTerms", false).compare(current_terms_version) == 0;
  bool training_done = params.get("CompletedTrainingVersion", false).compare(current_training_version) == 0;

  if (!accepted_terms) {
    setCurrentIndex(0);
  } else if (!training_done) {
    setCurrentIndex(1);
  } else {
    emit onboardingDone();
  }
}

OnboardingWindow::OnboardingWindow(QWidget *parent) : QStackedWidget(parent) {
  params = Params();
  current_terms_version = params.get("TermsVersion", false);
  current_training_version = params.get("TrainingVersion", false);

  addWidget(terms_screen());

  TrainingGuide* tr = new TrainingGuide(this);
  connect(tr, &TrainingGuide::completedTraining, [=](){
    Params().write_db_value("CompletedTrainingVersion", current_training_version);
    updateActiveScreen();
  });
  addWidget(tr);

  setStyleSheet(R"(
    * {
      color: white;
      background-color: black;
    }
    QPushButton {
      padding: 50px;
      border-radius: 30px;
      background-color: #292929;
    }
    QPushButton:disabled {
      color: #777777;
      background-color: #222222;
    }
  )");

  updateActiveScreen();
}
