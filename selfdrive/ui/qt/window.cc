#include <cassert>
#include <iostream>
#include <cmath>
#include <iostream>
#include <signal.h>

#include <QVBoxLayout>
#include <QMouseEvent>
#include <QPushButton>
#include <QGridLayout>

#include "window.hpp"
#include "settings.hpp"

#include "paint.hpp"

volatile sig_atomic_t do_exit = 0;

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  main_layout = new QStackedLayout;

  GLWindow * glWindow = new GLWindow(this);
  main_layout->addWidget(glWindow);

  SettingsWindow * settingsWindow = new SettingsWindow(this);
  main_layout->addWidget(settingsWindow);


  main_layout->setMargin(0);
  setLayout(main_layout);
  QObject::connect(glWindow, SIGNAL(openSettings()), this, SLOT(openSettings()));
  QObject::connect(settingsWindow, SIGNAL(closeSettings()), this, SLOT(closeSettings()));

  setStyleSheet(R"(
    * {
      color: white;
      background-color: #072339;
    }
  )");
}

void MainWindow::openSettings() {
  main_layout->setCurrentIndex(1);
}

void MainWindow::closeSettings() {
  main_layout->setCurrentIndex(0);
}


GLWindow::GLWindow(QWidget *parent) : QOpenGLWidget(parent) {
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));

}

GLWindow::~GLWindow() {
  makeCurrent();
  doneCurrent();
}

void GLWindow::initializeGL() {
  initializeOpenGLFunctions();
  std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
  std::cout << "OpenGL vendor: " << glGetString(GL_VENDOR) << std::endl;
  std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;
  std::cout << "OpenGL language version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

  ui_state = new UIState();
  ui_init(ui_state);
  ui_state->sound = &sound;
  ui_state->fb_w = vwp_w;
  ui_state->fb_h = vwp_h;

  timer->start(50);
}

void GLWindow::timerUpdate(){
  ui_update(ui_state);
  update();
}

void GLWindow::resizeGL(int w, int h) {
  std::cout << "resize " << w << "x" << h << std::endl;
}

void GLWindow::paintGL() {
  ui_draw(ui_state);
}

void GLWindow::mousePressEvent(QMouseEvent *e) {
  // Settings button click
  if (!ui_state->scene.uilayout_sidebarcollapsed && e->x() <= sbr_w) {
    if (e->x() >= settings_btn_x && e->x() < (settings_btn_x + settings_btn_w)
        && e->y() >= settings_btn_y && e->y() < (settings_btn_y + settings_btn_h)) {
      emit openSettings();
    }
  }

  // Vision click
  if (ui_state->started && (e->x() >= ui_state->scene.ui_viz_rx - bdr_s)){
    ui_state->scene.uilayout_sidebarcollapsed = !ui_state->scene.uilayout_sidebarcollapsed;
  }
}


/* HACKS */
bool Sound::init(int volume) { return true; }
bool Sound::play(AudibleAlert alert) { printf("play sound: %d\n", (int)alert); return true; }
void Sound::stop() {}
void Sound::setVolume(int volume) {}
Sound::~Sound() {}

EGLImageTexture::EGLImageTexture(int width, int height, int stride, int size, int bpp,int fd, void *addr) {
  assert((size % stride) == 0);
  assert((stride % bpp) == 0);
  glGenTextures(1, &frame_tex);
  glBindTexture(GL_TEXTURE_2D, frame_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, addr);
  glGenerateMipmap(GL_TEXTURE_2D);
}

EGLImageTexture::~EGLImageTexture() {
  glDeleteTextures(1, &frame_tex);
}

FramebufferState* framebuffer_init(const char* name, int32_t layer, int alpha,
                                   int *out_w, int *out_h) {
  return (FramebufferState*)1; // not null
}
