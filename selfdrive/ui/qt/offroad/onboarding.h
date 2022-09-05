#pragma once

#include <QElapsedTimer>
#include <QImage>
#include <QMouseEvent>
#include <QPushButton>
#include <QStackedWidget>
#include <QWidget>

#include "common/params.h"
#include "selfdrive/ui/qt/qt_window.h"

class TrainingGuide : public QFrame {
  Q_OBJECT

public:
  explicit TrainingGuide(QWidget *parent = 0);

private:
  static constexpr qreal TOP_MARGIN = 100;
  static constexpr qreal LEFT_MARGIN = 100;
  static constexpr qreal BOTTOM_MARGIN = 100;
  static constexpr qreal RIGHT_MARGIN = 800;
  typedef void (TrainingGuide::*paintFunction)(QPainter &p);

  void showEvent(QShowEvent *event) override;
  void drawButton(QPainter &p, const QRect &rect, const QString(&text), const QColor &bg, const QColor &f);
  void drawBody(QPainter &p, const QString &title, const QString &text, const QString &foot = {}, int right_margin = RIGHT_MARGIN, bool has_icon = true);
  void step0(QPainter &p);
  void step1(QPainter &p);
  void step2(QPainter &p);
  void step3(QPainter &p);
  void step4(QPainter &p);
  void step5(QPainter &p);
  void step6(QPainter &p);
  void step7(QPainter &p);
  void step8(QPainter &p);
  void step9(QPainter &p);
  void step10(QPainter &p);
  void step11(QPainter &p);
  void step12(QPainter &p);
  void step13(QPainter &p);
  void step14(QPainter &p);
  void step15(QPainter &p);
  void step16(QPainter &p);
  void step17(QPainter &p);
  void step18(QPainter &p);

  void paintEvent(QPaintEvent *event) override;
  void mouseReleaseEvent(QMouseEvent* e) override;

  QImage image;
  int currentIndex = 0;

  const QRect dm_yes = QRect(650, 780, 460, 150);
  const QRect dm_no = QRect(LEFT_MARGIN, 780, 460, 150);
  const QRect restart_training = QRect(108, 804, 426, 164);
  const QRect finish_training = QRect(630, 804, 626, 164);

  // auto d = TrainingGuide::step0;
  // Bounding boxes for each training guide step
  const QRect continueBtnStandard = {1620, 0, 300, 1080};
  QVector<QPair<paintFunction, QRect>> standardPages {
    {&TrainingGuide::step0, QRect(112, 804, 619, 150)},
    {&TrainingGuide::step1, continueBtnStandard},
    {&TrainingGuide::step2, continueBtnStandard},
    {&TrainingGuide::step3, QRect(1476, 565, 253, 308)},
    {&TrainingGuide::step4, QRect(1501, 529, 184, 108)},
    {&TrainingGuide::step5, continueBtnStandard},
    {&TrainingGuide::step6, QRect(1613, 665, 178, 153)},
    {&TrainingGuide::step7, QRect(1220, 0, 420, 730)},
    {&TrainingGuide::step8, QRect(1335, 499, 440, 147)},
    {&TrainingGuide::step9, dm_no.united(dm_yes)},
    {&TrainingGuide::step10, QRect(1412, 199, 316, 333)},
    {&TrainingGuide::step11, continueBtnStandard},
    {&TrainingGuide::step12, QRect(1237, 63, 683, 1017)},
    {&TrainingGuide::step13, continueBtnStandard},
    {&TrainingGuide::step14, QRect(1455, 110, 313, 860)},
    {&TrainingGuide::step15, QRect(1253, 519, 383, 228)},
    {&TrainingGuide::step16, continueBtnStandard},
    {&TrainingGuide::step17, continueBtnStandard},
    {&TrainingGuide::step18, finish_training},
  };

  const QRect continueBtnWide = {1840, 0, 320, 1080};
  const QVector<QPair<paintFunction, QRect>> widePages {
    {&TrainingGuide::step0, QRect(112, 804, 618, 164)},
    {&TrainingGuide::step1, continueBtnWide},
    {&TrainingGuide::step2, continueBtnWide},
    {&TrainingGuide::step3, QRect(1641, 558, 210, 313)},
    {&TrainingGuide::step4, QRect(1662, 528, 184, 108)},
    {&TrainingGuide::step5, continueBtnWide},
    {&TrainingGuide::step6, QRect(1814, 621, 211, 170)},
    {&TrainingGuide::step7, QRect(1350, 0, 497, 755)},
    {&TrainingGuide::step8, QRect(1553, 516, 406, 112)},
    {&TrainingGuide::step9, dm_no.united(dm_yes)},
    {&TrainingGuide::step10, QRect(1598, 199, 316, 333)},
    {&TrainingGuide::step11, continueBtnWide},
    {&TrainingGuide::step12, QRect(1364, 90, 796, 990)},
    {&TrainingGuide::step13, continueBtnWide},
    {&TrainingGuide::step14, QRect(1593, 114, 318, 853)},
    {&TrainingGuide::step15, QRect(1379, 511, 391, 243)},
    {&TrainingGuide::step16, continueBtnWide},
    {&TrainingGuide::step17, continueBtnWide},
    {&TrainingGuide::step18, finish_training},
  };

  QString img_path;
  QVector<QPair<paintFunction, QRect>> pages;
  QElapsedTimer click_timer;

signals:
  void completedTraining();
};


class TermsPage : public QFrame {
  Q_OBJECT

public:
  explicit TermsPage(QWidget *parent = 0) : QFrame(parent) {};

public slots:
  void enableAccept();

private:
  void showEvent(QShowEvent *event) override;

  QPushButton *accept_btn;

signals:
  void acceptedTerms();
  void declinedTerms();
};

class DeclinePage : public QFrame {
  Q_OBJECT

public:
  explicit DeclinePage(QWidget *parent = 0) : QFrame(parent) {};

private:
  void showEvent(QShowEvent *event) override;

signals:
  void getBack();
};

class OnboardingWindow : public QStackedWidget {
  Q_OBJECT

public:
  explicit OnboardingWindow(QWidget *parent = 0);
  inline void showTrainingGuide() { setCurrentIndex(1); }
  inline bool completed() const { return accepted_terms && training_done; }

private:
  void updateActiveScreen();

  Params params;
  bool accepted_terms = false, training_done = false;

signals:
  void onboardingDone();
};
