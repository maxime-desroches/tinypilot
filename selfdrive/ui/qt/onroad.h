#pragma once

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsWidget>
#include <QHBoxLayout>
#include <QWidget>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "selfdrive/ui/ui.h"

// ***** onroad widgets *****

class MaxSpeedItem : public QGraphicsItem {
public:
  MaxSpeedItem(QGraphicsItem *parent = nullptr) : QGraphicsItem(parent) {}
  void update(bool cruise_set, const QString &speed);
  QRectF boundingRect() const override { return {0, 0, 184, 202}; }

protected:
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) override;

  QString maxSpeed;
  bool is_cruise_set = false;
};

class CurrentSpeedItem : public QGraphicsItem {
public:
  CurrentSpeedItem(QGraphicsItem *parent = nullptr) : QGraphicsItem(parent) {}
  void update(const QString &speed, const QString &unit);
  QRectF boundingRect() const override { return {0, 0, 300, 300}; }
 
protected:
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) override;

  QString speed;
  QString speedUnit;
};

class OnroadAlerts : public QGraphicsRectItem {
public:
  OnroadAlerts(QGraphicsItem *parent = 0) : QGraphicsRectItem(parent) {}
  void update(const Alert &a, const QColor &color);
  QRectF boundingRect() const override;

protected:
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
  QColor bg;
  Alert alert = {};
};

class IconItem : public QGraphicsItem {
public:
  IconItem(const QString &fn, QGraphicsItem *parent = 0) : QGraphicsItem(parent) {
    pixmap = QPixmap(fn).scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    setVisible(false);
  }
  QRectF boundingRect() const override { return {0, 0, (qreal)radius, (qreal)radius}; }
  void update(const QColor color, float opacity);

protected:
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

  QPixmap pixmap;
  QColor bg;
  bool opacity = 1.0;
  const int radius = 192;
  const int img_size = (radius / 2) * 1.5;
};

class OnroadHud : public QGraphicsWidget {
public:
  explicit OnroadHud(QGraphicsItem *parent = nullptr);
  void updateState(const UIState &s);
  void 	setGeometry(const QRectF &rect) override;
  inline void updateAlert(const Alert &alert, const QColor &color) {
    alerts->update(alert, color);
  }

private:
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
  void drawLaneLines(QPainter &painter, const UIScene &scene);
  void drawLead(QPainter &painter, const cereal::ModelDataV2::LeadDataV3::Reader &lead_data, const QPointF &vd);
  inline QColor redColor(int alpha = 255) { return QColor(201, 34, 49, alpha); }

  OnroadAlerts *alerts;
  QGraphicsRectItem *header;
  MaxSpeedItem *max_speed;
  CurrentSpeedItem *current_speed;
  IconItem *dm, *wheel;
};

class OnroadGraphicsView : public QGraphicsView {
public:
  OnroadGraphicsView(QWidget *parent = nullptr);
  inline void updateAlert(const Alert &alert, const QColor &color) {
    hud->updateAlert(alert, color);
  }

protected:
  void offroadTransition(bool offroad);
  void resizeEvent(QResizeEvent *event) override;
  void drawBackground(QPainter *painter, const QRectF &rect) override {
    camera_view->doPaint();
  }

  OnroadHud *hud;
  CameraViewWidget *camera_view;
};

// container for all onroad widgets
class OnroadWindow : public QWidget {
  Q_OBJECT

public:
  OnroadWindow(QWidget* parent = 0);
  bool isMapVisible() const { return map && map->isVisible(); }

private:
  void mousePressEvent(QMouseEvent* e) override;

  QColor bg = bg_colors[STATUS_DISENGAGED];
  OnroadGraphicsView *onroad_view;
  QWidget *map = nullptr;
  QHBoxLayout* split;

private slots:
  void offroadTransition(bool offroad);
  void updateState(const UIState &s);
};
