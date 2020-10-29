#pragma once
#include <QWidget>
#include <QPushButton>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedLayout>

class KeyboardLayout : public QWidget{
  Q_OBJECT
  private:

  public:
    explicit KeyboardLayout(QWidget *parent, QVector<QString> p[4]);
};

class Keyboard : public QWidget {
  Q_OBJECT

  public:
    explicit Keyboard(QWidget *parent = 0);
  private:
    QStackedLayout* main_layout;
  private slots:
    void handleButton(QAbstractButton* m_button);
  signals:
    void emitButton(QString s);

};
