#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QPushButton>

#include "common/params.h"
#include "selfdrive/ui/qt/widgets/input.h"
#include "selfdrive/ui/qt/widgets/toggle.h"

QFrame *horizontal_line(QWidget *parent = nullptr);

class ElidedLabel : public QLabel {
  Q_OBJECT

public:
  explicit ElidedLabel(QWidget *parent = 0);
  explicit ElidedLabel(const QString &text, QWidget *parent = 0);

signals:
  void clicked();

protected:
  void paintEvent(QPaintEvent *event) override;
  void resizeEvent(QResizeEvent* event) override;
  void mouseReleaseEvent(QMouseEvent *event) override {
    if (rect().contains(event->pos())) {
      emit clicked();
    }
  }
  QString lastText_, elidedText_;
};


class AbstractControl : public QFrame {
  Q_OBJECT

public:
  void setDescription(const QString &desc) {
    if (description) description->setText(desc);
  }

  void setTitle(const QString &title) {
    title_label->setText(title);
  }

  void setValue(const QString &val) {
    value->setText(val);
  }

  const QString getDescription() {
    return description->text();
  }

  QLabel *icon_label;
  QPixmap icon_pixmap;

public slots:
  void showDescription() {
    description->setVisible(true);
  };

signals:
  void showDescriptionEvent();

protected:
  AbstractControl(const QString &title, const QString &desc = "", const QString &icon = "", QWidget *parent = nullptr);
  void hideEvent(QHideEvent *e) override;

  QHBoxLayout *hlayout;
  QPushButton *title_label;

private:
  ElidedLabel *value;
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

  void setEnabled(bool enabled) {
    toggle.setEnabled(enabled);
    toggle.update();
  }

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
    QObject::connect(this, &ParamControl::toggleFlipped, [=](bool state) {
      QString content("<body><h2 style=\"text-align: center;\">" + title + "</h2><br>"
                      "<p style=\"text-align: center; margin: 0 128px; font-size: 50px;\">" + getDescription() + "</p></body>");
      ConfirmationDialog dialog(content, tr("Enable"), tr("Cancel"), true, this);

      bool confirmed = store_confirm && params.getBool(key + "Confirmed");
      if (!confirm || confirmed || !state || dialog.exec()) {
        if (store_confirm && state) params.putBool(key + "Confirmed", true);
        params.putBool(key, state);
        setIcon(state);
      } else {
        toggle.togglePosition();
      }
    });
  }

  void setConfirmation(bool _confirm, bool _store_confirm) {
    confirm = _confirm;
    store_confirm = _store_confirm;
  };

  void setActiveIcon(const QString &icon) {
    active_icon_pixmap = QPixmap(icon).scaledToWidth(80, Qt::SmoothTransformation);
  }

  void refresh() {
    bool state = params.getBool(key);
    if (state != toggle.on) {
      toggle.togglePosition();
      setIcon(state);
    }
  };

  void showEvent(QShowEvent *event) override {
    refresh();
  };

private:
  void setIcon(bool state) {
    if (state && !active_icon_pixmap.isNull()) {
      icon_label->setPixmap(active_icon_pixmap);
    } else if (!icon_pixmap.isNull()) {
      icon_label->setPixmap(icon_pixmap);
    }
  };

  std::string key;
  Params params;
  QPixmap active_icon_pixmap;
  bool confirm = false;
  bool store_confirm = false;
};
 
class ButtonParamControl : public AbstractControl {
  Q_OBJECT
public:
  ButtonParamControl(const QString &param, const QString &title, const QString &desc, const QString &icon,
                                                  std::vector<QString> button_texts, std::vector<int> button_widths) : AbstractControl(title, desc, icon) {

    select_style = (R"(
        padding: 0;
        border-radius: 50px;
        font-size: 45px;
        font-weight: 500;
        color: #E4E4E4;
        background-color: #33Ab4C;
      )");
    unselect_style = (R"(
        padding: 0;
        border-radius: 50px;
        font-size: 45px;
        font-weight: 350;
        color: #E4E4E4;
        background-color: #393939;
      )");
    key = param.toStdString();
    for (int i = 0; i < button_texts.size(); i++) {
      QPushButton *button = new QPushButton();
      button->setText(button_texts[i]);
      hlayout->addWidget(button);
      button->setFixedSize(button_widths[i], 100);
      button->setStyleSheet(unselect_style);
      buttons.push_back(button);
      QObject::connect(button, &QPushButton::clicked, [=]() {
        params.put(key, (QString::number(i)).toStdString());
      refresh();
      });
    }
  }

  int get_param() {
    auto value = params.get(key);
    if (!value.empty()) {
      return std::stoi(value);
    } else {
      return 0;
    };
  };
  void set_param(int new_value) {
      QString values = QString::number(new_value);
      params.put(key, values.toStdString());
      refresh();
  };

  void refresh() {
    int value = get_param();
    for (int i = 0; i < buttons.size(); i++) {
      buttons[i]->setStyleSheet(unselect_style);
      if (value == i) {
        buttons[i]->setStyleSheet(select_style);
      }
    }
  }

private:
  std::string key;
  std::vector<QPushButton*> buttons;
  QString unselect_style;
  QString select_style;
  Params params;
};




class ListWidget : public QWidget {
  Q_OBJECT
 public:
  explicit ListWidget(QWidget *parent = 0) : QWidget(parent), outer_layout(this) {
    outer_layout.setMargin(0);
    outer_layout.setSpacing(0);
    outer_layout.addLayout(&inner_layout);
    inner_layout.setMargin(0);
    inner_layout.setSpacing(25); // default spacing is 25
    outer_layout.addStretch();
  }
  inline void addItem(QWidget *w) { inner_layout.addWidget(w); }
  inline void addItem(QLayout *layout) { inner_layout.addLayout(layout); }
  inline void setSpacing(int spacing) { inner_layout.setSpacing(spacing); }

private:
  void paintEvent(QPaintEvent *) override {
    QPainter p(this);
    p.setPen(Qt::gray);
    for (int i = 0; i < inner_layout.count() - 1; ++i) {
      QWidget *widget = inner_layout.itemAt(i)->widget();
      if (widget == nullptr || widget->isVisible()) {
        QRect r = inner_layout.itemAt(i)->geometry();
        int bottom = r.bottom() + inner_layout.spacing() / 2;
        p.drawLine(r.left() + 40, bottom, r.right() - 40, bottom);
      }
    }
  }
  QVBoxLayout outer_layout;
  QVBoxLayout inner_layout;
};

// convenience class for wrapping layouts
class LayoutWidget : public QWidget {
  Q_OBJECT

public:
  LayoutWidget(QLayout *l, QWidget *parent = nullptr) : QWidget(parent) {
    setLayout(l);
  };
};

class ClickableWidget : public QWidget {
  Q_OBJECT

public:
  ClickableWidget(QWidget *parent = nullptr);

protected:
  void mouseReleaseEvent(QMouseEvent *event) override;
  void paintEvent(QPaintEvent *) override;

signals:
  void clicked();
};
