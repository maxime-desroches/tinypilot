#include "selfdrive/ui/qt/widgets/cameraview.h"

#include "selfdrive/ui/qt/qt_window.h"

namespace {
const char frame_vertex_shader[] =
#ifdef NANOVG_GL3_IMPLEMENTATION
  "#version 150 core\n"
#else
  "#version 300 es\n"
#endif
  "in vec4 aPosition;\n"
  "in vec4 aTexCoord;\n"
  "uniform mat4 uTransform;\n"
  "out vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vTexCoord = aTexCoord;\n"
  "}\n";

const char frame_fragment_shader[] =
#ifdef NANOVG_GL3_IMPLEMENTATION
  "#version 150 core\n"
#else
  "#version 300 es\n"
#endif
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "in vec4 vTexCoord;\n"
  "out vec4 colorOut;\n"
  "void main() {\n"
  "  colorOut = texture(uTexture, vTexCoord.xy);\n"
#ifdef QCOM
  "  vec3 dz = vec3(0.0627f, 0.0627f, 0.0627f);\n"
  "  colorOut.rgb = ((vec3(1.0f, 1.0f, 1.0f) - dz) * colorOut.rgb / vec3(1.0f, 1.0f, 1.0f)) + dz;\n"
#endif
  "}\n";

const mat4 device_transform = {{
  1.0,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

mat4 get_driver_view_transform() {
  const float driver_view_ratio = 1.333;
  mat4 transform;
  if (Hardware::TICI()) {
    // from dmonitoring.cc
    const int full_width_tici = 1928;
    const int full_height_tici = 1208;
    const int adapt_width_tici = 668;
    const int crop_x_offset = 32;
    const int crop_y_offset = -196;
    const float yscale = full_height_tici * driver_view_ratio / adapt_width_tici;
    const float xscale = yscale*(1080-2*bdr_s)/(2160-2*bdr_s)*full_width_tici/full_height_tici;
    transform = (mat4){{
      xscale,  0.0, 0.0, xscale*crop_x_offset/full_width_tici*2,
      0.0,  yscale, 0.0, yscale*crop_y_offset/full_height_tici*2,
      0.0,  0.0, 1.0, 0.0,
      0.0,  0.0, 0.0, 1.0,
    }};
  
  } else {
     // frame from 4/3 to 16/9 display
    transform = (mat4){{
      driver_view_ratio*(1080-2*bdr_s)/(1920-2*bdr_s),  0.0, 0.0, 0.0,
      0.0,  1.0, 0.0, 0.0,
      0.0,  0.0, 1.0, 0.0,
      0.0,  0.0, 0.0, 1.0,
    }};
  }
  return transform;
}

} // namespace

CameraViewWidget::CameraViewWidget(VisionStreamType stream_type, QWidget* parent) : stream_type(stream_type), QOpenGLWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  
  timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, &CameraViewWidget::updateFrame);
}

CameraViewWidget::~CameraViewWidget() {
  makeCurrent();
  doneCurrent();
  glDeleteVertexArrays(1, &frame_vao);
  glDeleteBuffers(1, &frame_vbo);
  glDeleteBuffers(1, &frame_ibo);
}

void CameraViewWidget::initializeGL() {
  initializeOpenGLFunctions();

  video_rect = {bdr_s, bdr_s, vwp_w - 2 * bdr_s, vwp_h - 2 * bdr_s};
  gl_shader = std::make_unique<GLShader>(frame_vertex_shader, frame_fragment_shader);
  GLint frame_pos_loc = glGetAttribLocation(gl_shader->prog, "aPosition");
  GLint frame_texcoord_loc = glGetAttribLocation(gl_shader->prog, "aTexCoord");

  auto [x1, x2, y1, y2] = stream_type == VISION_STREAM_RGB_FRONT ? std::tuple(0.f, 1.f, 1.f, 0.f) : std::tuple(1.f, 0.f, 1.f, 0.f);
  const uint8_t frame_indicies[] = {0, 1, 2, 0, 2, 3};
  const float frame_coords[4][4] = {
    {-1.0, -1.0, x2, y1}, //bl
    {-1.0,  1.0, x2, y2}, //tl
    { 1.0,  1.0, x1, y2}, //tr
    { 1.0, -1.0, x1, y1}, //br
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

  if (stream_type == VISION_STREAM_RGB_FRONT) {
    frame_mat = matmul(device_transform, get_driver_view_transform());
  } else {
    auto intrinsic_matrix = stream_type == VISION_STREAM_RGB_WIDE ? ecam_intrinsic_matrix : fcam_intrinsic_matrix;
    float zoom_ = zoom / intrinsic_matrix.v[0];
    if (stream_type == VISION_STREAM_RGB_WIDE) {
      zoom_ *= 0.5;
    }
    float zx = zoom_ * 2 * intrinsic_matrix.v[2] / video_rect.width();
    float zy = zoom_ * 2 * intrinsic_matrix.v[5] / video_rect.height();

    const mat4 frame_transform = {{
      zx, 0.0, 0.0, 0.0,
      0.0, zy, 0.0, -y_offset / video_rect.height() * 2,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
    }};
    frame_mat = matmul(device_transform, frame_transform);
  }
  vipc_client = std::make_unique<VisionIpcClient>("camerad", stream_type, true);
  timer->start(0);
}

void CameraViewWidget::resizeGL(int w, int h) {
  viz_rect = {bdr_s, bdr_s, w - 2 * bdr_s, h - 2 * bdr_s};
}

void CameraViewWidget::paintGL() {
  // draw background

  const QColor &color = bg_colors[ui_status];
  glClearColor(color.redF(), color.greenF(), color.blueF(), 1.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  // draw frame

  glEnable(GL_SCISSOR_TEST);
  glViewport(video_rect.left(), video_rect.top(), video_rect.width(), video_rect.height());
  glScissor(viz_rect.left(), viz_rect.top(), viz_rect.width(), viz_rect.height());
  
  if (last_frame) {
    glBindVertexArray(frame_vao);
    glActiveTexture(GL_TEXTURE0);

    glBindTexture(GL_TEXTURE_2D, texture[last_frame->idx]->frame_tex);
    if (!Hardware::EON()) {
      // this is handled in ion on QCOM
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, last_frame->width, last_frame->height,
                   0, GL_RGB, GL_UNSIGNED_BYTE, last_frame->addr);
    }
  
    glUseProgram(gl_shader->prog);
    glUniform1i(gl_shader->getUniformLocation("uTexture"), 0);
    glUniformMatrix4fv(gl_shader->getUniformLocation("uTransform"), 1, GL_TRUE, frame_mat.v);

    assert(glGetError() == GL_NO_ERROR);
    glEnableVertexAttribArray(0);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (const void *)0);
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
  }

  // draw others

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glViewport(0, 0, vwp_w, vwp_h);
  draw();
  glDisable(GL_BLEND);
  glDisable(GL_SCISSOR_TEST);
}


void CameraViewWidget::updateFrame() {
   if (!vipc_client->connected && vipc_client->connect(false)) {
    // init vision
    for (int i = 0; i < vipc_client->num_buffers; i++) {
      texture[i].reset(new EGLImageTexture(&vipc_client->buffers[i]));

      glBindTexture(GL_TEXTURE_2D, texture[i]->frame_tex);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

      // BGR
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
      assert(glGetError() == GL_NO_ERROR);
    }
    last_frame = nullptr;
  }

  if (vipc_client->connected) {
    if (VisionBuf *buf = vipc_client->recv(); buf != nullptr) {
      last_frame = buf;
    } else if (!Hardware::PC()) {
      LOGE("visionIPC receive timeout");
    }
  }

  // repaint
  QOpenGLWidget::update();
}
