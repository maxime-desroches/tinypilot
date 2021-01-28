#include <QLabel>
#include <QString>
#include <QPushButton>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QDebug>

#include "onboarding.hpp"
#include "common/params.h"


QLabel * title_label(QString text) {
  QLabel *l = new QLabel(text);
  l->setStyleSheet(R"(font-size: 100px; font-weight: bold;)");
  return l;
}

QWidget* layout2Widget(QLayout* l){
  QWidget* q = new QWidget;
  q->setLayout(l);
  return q;
}




void TrainingGuide::mouseReleaseEvent(QMouseEvent *e) {
  // Uncomment to help with tuning the bounding boxes
  // qDebug()<<"Got event at index"<<currentIndex;
  // qDebug()<<"Got coords at "<<e->x()<<e->y();

  if(boundingBox[currentIndex][0] <= e->x() && e->x() <= boundingBox[currentIndex][1] && boundingBox[currentIndex][2] <= e->y() && e->y() <= boundingBox[currentIndex][3] ){
    slayout->setCurrentIndex(++currentIndex);
  }
  if (currentIndex >= numberOfFrames){
    emit completedTraining();
    return;
  }
}

TrainingGuide::TrainingGuide(QWidget* parent) {
  QHBoxLayout* hlayout = new QHBoxLayout;
  hlayout->addSpacing(270); // 2160*1080 -> 1620*1080
  slayout = new QStackedLayout(this);
  for (int i = 0; i <= 14; i++) {
    QWidget* w = new QWidget;
    w->setStyleSheet(".QWidget {background-image: url(../assets/training/step" + QString::number(i) + ".png); background-color: rgba(255, 255, 255, 10);}");
    slayout->addWidget(w);
  }

  hlayout->addWidget(layout2Widget(slayout));
  hlayout->addSpacing(270);
  setLayout(hlayout);
}


QWidget* OnboardingWindow::terms_screen() {

  QGridLayout *main_layout = new QGridLayout();
  main_layout->setMargin(100);
  main_layout->setSpacing(30);

  main_layout->addWidget(title_label("Review Terms"), 0, 0, 1, -1);

  QLabel *terms = new QLabel("See terms at https://my.comma.ai/terms");
  terms->setAlignment(Qt::AlignCenter);
  terms->setStyleSheet(R"(
    font-size: 75px;
    border-radius: 10px;
    background-color: #292929;
  )");
  main_layout->addWidget(terms, 1, 0, 1, -1);
  main_layout->setRowStretch(1, 1);

  QPushButton *accept_btn = new QPushButton("Accept");
  main_layout->addWidget(accept_btn, 2, 1);
  QObject::connect(accept_btn, &QPushButton::released, [=]() {
    Params().write_db_value("HasAcceptedTerms", LATEST_TERMS_VERSION);
    updateActiveScreen();
  });

  main_layout->addWidget(new QPushButton("Decline"), 2, 0);

  QWidget *widget = new QWidget;
  widget->setLayout(main_layout);
  widget->setStyleSheet(R"(
    QPushButton {
      font-size: 50px;
      padding: 50px;
      border-radius: 10px;
      background-color: #292929;
    }
  )");

  return widget;
}

void OnboardingWindow::updateActiveScreen() {

  Params params = Params();
  bool accepted_terms = params.get("HasAcceptedTerms", false).compare(LATEST_TERMS_VERSION) == 0;
  bool training_done = params.get("CompletedTrainingVersion", false).compare(LATEST_TRAINING_VERSION) == 0;

  if (!accepted_terms) {
    setCurrentIndex(0);
  } else if (!training_done) {
    setCurrentIndex(1);
  } else {
    emit onboardingDone();
  }
}

OnboardingWindow::OnboardingWindow(QWidget *parent) : QStackedWidget(parent) {
  addWidget(terms_screen());
  TrainingGuide* tr = new TrainingGuide(this);
  connect(tr, &TrainingGuide::completedTraining, [=](){Params().write_db_value("CompletedTrainingVersion", LATEST_TRAINING_VERSION); updateActiveScreen();});
  addWidget(tr);

  setStyleSheet(R"(
    * {
      color: white;
      background-color: black;
    }
    QPushButton {
      padding: 50px;
      border-radius: 10px;
      background-color: #292929;
    }
  )");

  // TODO: remove this after training guide is done
  // Params().write_db_value("CompletedTrainingVersion", LATEST_TRAINING_VERSION);

  updateActiveScreen();
}
