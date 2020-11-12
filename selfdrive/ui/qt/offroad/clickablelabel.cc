#include "clickablelabel.hpp"

#include <QDebug>
#include <QEvent>

ClickableLabel::ClickableLabel(QWidget *parent, int index)
  : QLabel(parent),
    index(index) {
  setStyleSheet(R"(color: #8a8a8a;)");
}

void ClickableLabel::enterEvent(QEvent *e) {
  setCursor(Qt::PointingHandCursor);
}

void ClickableLabel::leaveEvent(QEvent *e) {
  setCursor(Qt::ArrowCursor);
}

void ClickableLabel::mousePressEvent(QMouseEvent *e) {
  if (index != -1) emit selected(index);
}

void ClickableLabel::emphasize() {
  setStyleSheet(R"(color: #ffffff;)");
}

void ClickableLabel::deemphasize() {
  setStyleSheet(R"(color: #8a8a8a;)");
}
