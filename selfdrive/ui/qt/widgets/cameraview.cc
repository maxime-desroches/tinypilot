#include "selfdrive/ui/qt/widgets/cameraview.h"

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GLES3/gl3.h>
#endif

#include <QOpenGLBuffer>
#include <QOffscreenSurface>

namespace {

const char frame_vertex_shader[] =
#ifdef __APPLE__
  "#version 330 core\n"
#else
  "#version 300 es\n"
#endif
  "layout(location = 0) in vec4 aPosition;\n"
  "layout(location = 1) in vec2 aTexCoord;\n"
  "uniform mat4 uTransform;\n"
  "out vec2 vTexCoord;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vTexCoord = aTexCoord;\n"
  "}\n";

const char frame_fragment_shader[] =
#ifdef __APPLE__
  "#version 330 core\n"
#else
  "#version 300 es\n"
  "precision mediump float;\n"
#endif
  "uniform sampler2D uTextureY;\n"
  "uniform sampler2D uTextureU;\n"
  "uniform sampler2D uTextureV;\n"
  "in vec2 vTexCoord;\n"
  "out vec4 colorOut;\n"
  "void main() {\n"
  "  float y = texture(uTextureY, vTexCoord).r;\n"
  "  float u = texture(uTextureU, vTexCoord).r - 0.5;\n"
  "  float v = texture(uTextureV, vTexCoord).r - 0.5;\n"
  "  float r = y + 1.402 * v;\n"
  "  float g = y - 0.344 * u - 0.714 * v;\n"
  "  float b = y + 1.772 * u;\n"
  "  colorOut = vec4(r, g, b, 1.0);\n"
  "}\n";

const mat4 device_transform = {{
  1.0,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

mat4 get_driver_view_transform(int screen_width, int screen_height, int stream_width, int stream_height) {
  const float driver_view_ratio = 1.333;
  mat4 transform;
  if (stream_width == TICI_CAM_WIDTH) {
    const float yscale = stream_height * driver_view_ratio / tici_dm_crop::width;
    const float xscale = yscale*screen_height/screen_width*stream_width/stream_height;
    transform = (mat4){{
      xscale,  0.0, 0.0, xscale*tici_dm_crop::x_offset/stream_width*2,
      0.0,  yscale, 0.0, yscale*tici_dm_crop::y_offset/stream_height*2,
      0.0,  0.0, 1.0, 0.0,
      0.0,  0.0, 0.0, 1.0,
    }};
  } else {
    // frame from 4/3 to 16/9 display
    transform = (mat4){{
      driver_view_ratio * screen_height / screen_width,  0.0, 0.0, 0.0,
      0.0,  1.0, 0.0, 0.0,
      0.0,  0.0, 1.0, 0.0,
      0.0,  0.0, 0.0, 1.0,
    }};
  }
  return transform;
}

mat4 get_fit_view_transform(float widget_aspect_ratio, float frame_aspect_ratio) {
  float zx = 1, zy = 1;
  if (frame_aspect_ratio > widget_aspect_ratio) {
    zy = widget_aspect_ratio / frame_aspect_ratio;
  } else {
    zx = frame_aspect_ratio / widget_aspect_ratio;
  }

  const mat4 frame_transform = {{
    zx, 0.0, 0.0, 0.0,
    0.0, zy, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
  }};
  return frame_transform;
}

} // namespace

CameraViewWidget::CameraViewWidget(std::string stream_name, VisionStreamType type, bool zoom, bool manual_update, QWidget* parent) :
                                   stream_name(stream_name), stream_type(type), zoomed_view(zoom), manual_update(manual_update), QOpenGLWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  connect(this, &CameraViewWidget::vipcThreadConnected, this, &CameraViewWidget::vipcConnected, Qt::BlockingQueuedConnection);
  connect(this, &CameraViewWidget::vipcThreadFrameReceived, this, &CameraViewWidget::vipcFrameReceived);
}

CameraViewWidget::~CameraViewWidget() {
  makeCurrent();
  if (isValid()) {
    glDeleteVertexArrays(1, &frame_vao);
    glDeleteBuffers(1, &frame_vbo);
    glDeleteBuffers(1, &frame_ibo);
    glDeleteBuffers(3, textures);
  }
  doneCurrent();
}

void CameraViewWidget::initializeGL() {
  initializeOpenGLFunctions();

  program = std::make_unique<QOpenGLShaderProgram>(context());
  bool ret = program->addShaderFromSourceCode(QOpenGLShader::Vertex, frame_vertex_shader);
  assert(ret);
  ret = program->addShaderFromSourceCode(QOpenGLShader::Fragment, frame_fragment_shader);
  assert(ret);

  program->link();
  GLint frame_pos_loc = program->attributeLocation("aPosition");
  GLint frame_texcoord_loc = program->attributeLocation("aTexCoord");

  auto [x1, x2, y1, y2] = stream_type == VISION_STREAM_DRIVER ? std::tuple(0.f, 1.f, 1.f, 0.f) : std::tuple(1.f, 0.f, 1.f, 0.f);
  const uint8_t frame_indicies[] = {0, 1, 2, 0, 2, 3};
  const float frame_coords[4][4] = {
    {-1.0, -1.0, x2, y1}, // bl
    {-1.0,  1.0, x2, y2}, // tl
    { 1.0,  1.0, x1, y2}, // tr
    { 1.0, -1.0, x1, y1}, // br
  };

  glGenVertexArrays(1, &frame_vao);
  glBindVertexArray(frame_vao);
  glGenBuffers(1, &frame_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, frame_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(frame_coords), frame_coords, GL_STATIC_DRAW);
  glEnableVertexAttribArray(frame_pos_loc);
  glVertexAttribPointer(frame_pos_loc, 2, GL_FLOAT, GL_FALSE,
                        sizeof(frame_coords[0]), (const void *)0);
  glEnableVertexAttribArray(frame_texcoord_loc);
  glVertexAttribPointer(frame_texcoord_loc, 2, GL_FLOAT, GL_FALSE,
                        sizeof(frame_coords[0]), (const void *)(sizeof(float) * 2));
  glGenBuffers(1, &frame_ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, frame_ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(frame_indicies), frame_indicies, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glGenTextures(3, textures);
  glUseProgram(program->programId());
  glUniform1i(program->uniformLocation("uTextureY"), 0);
  glUniform1i(program->uniformLocation("uTextureU"), 1);
  glUniform1i(program->uniformLocation("uTextureV"), 2);
}

void CameraViewWidget::showEvent(QShowEvent *event) {
  latest_frame = nullptr;
  if (!vipc_thread) {
    vipc_thread = new QThread();
    connect(vipc_thread, &QThread::started, [=]() { vipcThread(); });
    connect(vipc_thread, &QThread::finished, vipc_thread, &QObject::deleteLater);
    vipc_thread->start();
  }
}

void CameraViewWidget::hideEvent(QHideEvent *event) {
  if (vipc_thread) {
    vipc_thread->requestInterruption();
    vipc_thread->quit();
    vipc_thread->wait();
    vipc_thread = nullptr;
  }
}

void CameraViewWidget::updateFrameMat(int w, int h) {
  if (zoomed_view) {
    if (stream_type == VISION_STREAM_DRIVER) {
      frame_mat = matmul(device_transform, get_driver_view_transform(w, h, stream_width, stream_height));
    } else {
      auto intrinsic_matrix = stream_type == VISION_STREAM_WIDE_ROAD ? ecam_intrinsic_matrix : fcam_intrinsic_matrix;
      float zoom = ZOOM / intrinsic_matrix.v[0];
      if (stream_type == VISION_STREAM_WIDE_ROAD) {
        zoom *= 0.5;
      }
      float zx = zoom * 2 * intrinsic_matrix.v[2] / width();
      float zy = zoom * 2 * intrinsic_matrix.v[5] / height();

      const mat4 frame_transform = {{
        zx, 0.0, 0.0, 0.0,
        0.0, zy, 0.0, -y_offset / height() * 2,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
      }};
      frame_mat = matmul(device_transform, frame_transform);
    }
  } else if (stream_width > 0 && stream_height > 0) {
    // fit frame to widget size
    float widget_aspect_ratio = (float)width() / height();
    float frame_aspect_ratio = (float)stream_width  / stream_height;
    frame_mat = matmul(device_transform, get_fit_view_transform(widget_aspect_ratio, frame_aspect_ratio));
  }
}

void CameraViewWidget::paintGL() {
  glClearColor(bg.redF(), bg.greenF(), bg.blueF(), bg.alphaF());
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  if (latest_frame == nullptr) return;

//  qDebug() << "Requested frame id:" << draw_frame_id;
//  qDebug() << "Drawing frame id:  " << cam_frame_id;
//  qDebug() << "Correct:           " << (draw_frame_id == cam_frame_id);
//  qDebug() << "Model frame id new:" << (prev_model_frame_id < draw_frame_id);
//  qDebug() << "Cam frame id new:  " << (prev_cam_frame_id < cam_frame_id);

  if (draw_frame_id != cam_frame_id) {
    qDebug() << "WRROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG FRAME!";
    wrong_frame += 1;
  } else {
    if (wrong_frame == 1 && )
    wrong_frame = std::max(wrong_frame - 1, 0);
  }

  assert((draw_frame_id > prev_model_frame_id) || (draw_frame_id == 0));  // never should draw duplicate model frames
  assert((cam_frame_id >= prev_cam_frame_id) || (cam_frame_id == 0));  // can draw same camera frames sometimes
  // ok if drawing on same frame or: (if during startup, or we never draw wrong frame greater than 4 times and it's not continually drawing wrong frames (sometimes when touching screen))
  assert((draw_frame_id == cam_frame_id) || (draw_frame_id < 5));
  // this allows touches, but you don't know if user touched, or we're just randomly drawing wrong frames
  // assert((draw_frame_id == cam_frame_id) || (draw_frame_id < 5) || ((wrong_frame <= 3)));

  prev_model_frame_id = draw_frame_id;
  prev_cam_frame_id = cam_frame_id;

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glViewport(0, 0, width(), height());
  glBindVertexArray(frame_vao);

  glUseProgram(program->programId());
  uint8_t *address[3] = {latest_frame->y, latest_frame->u, latest_frame->v};
  for (int i = 0; i < 3; ++i) {
    glActiveTexture(GL_TEXTURE0 + i);
    glBindTexture(GL_TEXTURE_2D, textures[i]);
    int width = i == 0 ? stream_width : stream_width / 2;
    int height = i == 0 ? stream_height : stream_height / 2;
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, address[i]);
    assert(glGetError() == GL_NO_ERROR);
  }

  glUniformMatrix4fv(program->uniformLocation("uTransform"), 1, GL_TRUE, frame_mat.v);
  assert(glGetError() == GL_NO_ERROR);
  glEnableVertexAttribArray(0);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (const void *)0);
  glDisableVertexAttribArray(0);
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
}

void CameraViewWidget::vipcConnected(VisionIpcClient *vipc_client) {
  makeCurrent();
  latest_frame = nullptr;
  stream_width = vipc_client->buffers[0].width;
  stream_height = vipc_client->buffers[0].height;

  for (int i = 0; i < 3; ++i) {
    glBindTexture(GL_TEXTURE_2D, textures[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    int width = i == 0 ? stream_width : stream_width / 2;
    int height = i == 0 ? stream_height : stream_height / 2;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    assert(glGetError() == GL_NO_ERROR);
  }

  updateFrameMat(width(), height());
}

void CameraViewWidget::vipcFrameReceived(VisionBuf *buf, int frame_id) {
  latest_frame = buf;
  cam_frame_id = frame_id;
  if (!manual_update) update();
}

void CameraViewWidget::vipcThread() {
  VisionStreamType cur_stream_type = stream_type;
  std::unique_ptr<VisionIpcClient> vipc_client;
  VisionIpcBufExtra meta_main = {0};

  while (!QThread::currentThread()->isInterruptionRequested()) {
    if (!vipc_client || cur_stream_type != stream_type) {
      cur_stream_type = stream_type;
      vipc_client.reset(new VisionIpcClient(stream_name, cur_stream_type, true));
    }

    if (!vipc_client->connected) {
      if (!vipc_client->connect(false)) {
        QThread::msleep(100);
        continue;
      }
      emit vipcThreadConnected(vipc_client.get());
    }

    if (VisionBuf *buf = vipc_client->recv(&meta_main, 1000)) {
      emit vipcThreadFrameReceived(buf, meta_main.frame_id);
    }
  }
}
