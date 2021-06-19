#pragma once

#include <QScrollArea>

class ScrollView : public QScrollArea {
  Q_OBJECT

public:
  explicit ScrollView(QWidget *w = nullptr, QWidget *parent = nullptr);
protected:
  void paintEvent(QPaintEvent *event) override;
  void hideEvent(QHideEvent *e) override;
  void 	scrollContentsBy(int dx, int dy) override;
  bool event(QEvent *e) override;
};
