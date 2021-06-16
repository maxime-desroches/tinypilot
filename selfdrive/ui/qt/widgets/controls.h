#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/widgets/toggle.h"

QFrame *horizontal_line(QWidget *parent = nullptr);
#include <QDebug>
class ElidedLabel : public QFrame {
  Q_OBJECT

 public:
  ElidedLabel(QWidget *parent = 0) : ElidedLabel("", parent) {}
  explicit ElidedLabel(const QString &text, QWidget *parent = 0) : content_(text.trimmed()), QFrame(parent) {
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  }
  void setAlignment(size_t alignment) { alignment_ = alignment; }
  const QString &text() const { return content_; }
  QSize 	minimumSizeHint() const override {
    return QSize(100, 80);
  }
  void setText(const QString &text) {
    content_ = text.trimmed();
    elidedText_ = "";
    update();
  }

 protected:
  void paintEvent(QPaintEvent *event) override;
  size_t alignment_ = Qt::AlignLeft | Qt::AlignVCenter;
  QString content_;
  QString elidedText_;
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

  QSize minimumSizeHint() const override {
    QSize size = QFrame::minimumSizeHint();
    size.setHeight(120);
    return size;
  };

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
  void released();

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
    if (params.getBool(param.toStdString().c_str())) {
      toggle.togglePosition();
    }
    QObject::connect(this, &ToggleControl::toggleFlipped, [=](bool state) {
      params.putBool(param.toStdString().c_str(), state);
    });
  }

private:
  Params params;
};
