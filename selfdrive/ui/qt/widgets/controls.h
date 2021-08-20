#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QPushButton>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/widgets/toggle.h"

QFrame *horizontal_line(QWidget *parent = nullptr);

class ElidedLabel : public QLabel {
  Q_OBJECT

 public:
  explicit ElidedLabel(QWidget *parent = 0);
  explicit ElidedLabel(const QString &text, QWidget *parent = 0);

 protected:
  void paintEvent(QPaintEvent *event) override;
  void resizeEvent(QResizeEvent* event) override;
  QString lastText_, elidedText_;
};

class AbstractControl : public QFrame {
  Q_OBJECT

public:
  void setDescription(const QString &desc) {
    if(description) description->setText(desc);
  }

signals:
  void showDescription();

protected:
  AbstractControl(const QString &title, const QString &desc = "", const QString &icon = "", QWidget *parent = nullptr);
  void hideEvent(QHideEvent *e) override;

  QHBoxLayout *hlayout;
  QPushButton *title_label;
  QLabel *description = nullptr;
};

// widget to display a value
class LabelControl : public AbstractControl {
  Q_OBJECT

public:
  LabelControl(const QString &title, const QString &text = "", const QString &desc = "", QWidget *parent = nullptr) : AbstractControl(title, desc, "", parent) {
    label.setText(text);
    label.setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    hlayout->addWidget(&label);
  }
  void setText(const QString &text) { label.setText(text); }

private:
  ElidedLabel label;
};

// widget for a button with a label
class ButtonControl : public AbstractControl {
  Q_OBJECT

public:
  ButtonControl(const QString &title, const QString &text, const QString &desc = "", QWidget *parent = nullptr);
  inline void setText(const QString &text) { btn.setText(text); }
  inline QString text() const { return btn.text(); }

signals:
  void clicked();

public slots:
  void setEnabled(bool enabled) { btn.setEnabled(enabled); };

private:
  QPushButton btn;
};

class ToggleControl : public AbstractControl {
  Q_OBJECT

public:
  ToggleControl(const QString &title, const QString &desc = "", const QString &icon = "", const bool state = false, QWidget *parent = nullptr) : AbstractControl(title, desc, icon, parent) {
    toggle.setFixedSize(150, 100);
    if (state) {
      toggle.togglePosition();
    }
    hlayout->addWidget(&toggle);
    QObject::connect(&toggle, &Toggle::stateChanged, this, &ToggleControl::toggleFlipped);
  }

  void setEnabled(bool enabled) { toggle.setEnabled(enabled); }

signals:
  void toggleFlipped(bool state);

protected:
  Toggle toggle;
};

// widget to toggle params
class ParamControl : public ToggleControl {
  Q_OBJECT

public:
  ParamControl(const QString &param, const QString &title, const QString &desc, const QString &icon, QWidget *parent = nullptr) : ToggleControl(title, desc, icon, false, parent) {
    key = param.toStdString();
    QObject::connect(this, &ToggleControl::toggleFlipped, [=](bool state) {
      params.putBool(key, state);
    });
  }

  void showEvent(QShowEvent *event) override {
    if (params.getBool(key) != toggle.on) {
      toggle.togglePosition();
    }
  };

private:
  std::string key;
  Params params;
};

class ListWidget : public QWidget {
  Q_OBJECT
 public:
  explicit ListWidget(QWidget *parent = 0) : QWidget(parent), mainLayout_(this) {
    mainLayout_.setMargin(0);
    mainLayout_.setSpacing(0);
    mainLayout_.addLayout(&layout_);
    layout_.setMargin(0);
    layout_.setSpacing(25); // default spacing is 25
    mainLayout_.addStretch();
  }
  inline void addItem(QWidget *w) { layout_.addWidget(w); }
  inline void addItem(QLayout *layout) { layout_.addLayout(layout); }
  inline void setSpacing(int spacing) { layout_.setSpacing(spacing); }

 private:
  void paintEvent(QPaintEvent *) override {
    QPainter p(this);
    p.setPen(Qt::gray);
    for (int i = 0; i < layout_.count() - 1; ++i) {
      QRect r = layout_.itemAt(i)->geometry();
      int bottom = r.bottom() + layout_.spacing() / 2;
      p.drawLine(r.left() + 40, bottom, r.right() - 40, bottom);
    }
  }
  QVBoxLayout mainLayout_; 
  QVBoxLayout layout_;
};
