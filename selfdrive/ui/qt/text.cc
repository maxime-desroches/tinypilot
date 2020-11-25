#include <cstdlib>

#include <QString>
#include <QLabel>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>

#include "qt_window.hpp"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QWidget *window = new QWidget();
  setMainWindow(window);

  QVBoxLayout *main_layout = new QVBoxLayout();

  QString text = "";
  for (int i = 1; i < argc; i++) {
    if (i > 1) {
      text.append(" ");
    }
    text.append(argv[i]);
  }

  QLabel *label = new QLabel(text);
  label->setAlignment(Qt::AlignTop);
  main_layout->addWidget(label);

  QPushButton *btn = new QPushButton();
#ifdef __aarch64__
  btn->setText("Reboot");
  QObject::connect(btn, &QPushButton::released, [=]() {
    std::system("sudo reboot");
  });
#else
  btn->setText("Exit");
  QObject::connect(btn, SIGNAL(released()), &a, SLOT(quit()));
#endif
  main_layout->addWidget(btn);

  window->setLayout(main_layout);
  window->setStyleSheet(R"(
    QWidget {
      margin: 60px;
      background-color: black;
    }
    QLabel {
      color: white;
      font-size: 60px;
    }
    QPushButton {
      color: white;
      font-size: 50px;
      padding: 60px;
      margin-left: 1500px;
      border-color: white;
      border-width: 2px;
      border-style: solid;
      border-radius: 20px;
    }
  )");

  return a.exec();
}
