#include "selfdrive/ui/replay/replay.h"

#include <termios.h>

#include <QApplication>
#include <QCommandLineParser>
#include <QThread>

int getch() {
  int ch;
  struct termios oldt;
  struct termios newt;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);

  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  return ch;
}

void keyboardThread(Replay *replay) {
  char c;
  while (true) {
    c = getch();
    if (c == '\n') {
      printf("Enter seek request: ");
      std::string r;
      std::cin >> r;

      try {
        if (r[0] == '#') {
          r.erase(0, 1);
          replay->seekTo(std::stoi(r) * 60);
        } else {
          replay->seekTo(std::stoi(r));
        }
      } catch (std::invalid_argument) {
        qDebug() << "invalid argument";
      }
      getch();  // remove \n from entering seek
    } else if (c == 'm') {
      replay->relativeSeek(+60);
    } else if (c == 'M') {
      replay->relativeSeek(-60);
    } else if (c == 's') {
      replay->relativeSeek(+10);
    } else if (c == 'S') {
      replay->relativeSeek(-10);
    } else if (c == 'G') {
      replay->relativeSeek(0);
    }
  }
}

int main(int argc, char *argv[]){
  QApplication a(argc, argv);

  QCommandLineParser parser;
  parser.setApplicationDescription("Mock openpilot components by publishing logged messages.");
  parser.addHelpOption();
  parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai\n"
                                        "here's a public demo route, use '3533c53bb29502d1|2019-12-10--01-13-27'");

  parser.process(a);
  if (parser.positionalArguments().empty()) {
    parser.showHelp();
  }

  QString route = parser.positionalArguments()[0];
  Replay *replay = new Replay(route);
  replay->start();

  // start keyboard control thread
  QThread *t = QThread::create(keyboardThread, replay);
  QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
  t->start();

  return a.exec();
}
