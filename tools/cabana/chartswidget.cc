#include "tools/cabana/chartswidget.h"

#include <QGraphicsLayout>
#include <QLabel>
#include <QPushButton>
#include <QStackedLayout>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

int64_t get_raw_value(uint8_t *data, size_t data_size, const Signal &sig) {
  int64_t ret = 0;

  int i = sig.msb / 8;
  int bits = sig.size;
  while (i >= 0 && i < data_size && bits > 0) {
    int lsb = (int)(sig.lsb / 8) == i ? sig.lsb : i * 8;
    int msb = (int)(sig.msb / 8) == i ? sig.msb : (i + 1) * 8 - 1;
    int size = msb - lsb + 1;

    uint64_t d = (data[i] >> (lsb - (i * 8))) & ((1ULL << size) - 1);
    ret |= d << (bits - size);

    bits -= size;
    i = sig.is_little_endian ? i - 1 : i + 1;
  }
  return ret;
}

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  connect(parser, &Parser::showPlot, this, &ChartsWidget::addChart);
  connect(parser, &Parser::hidePlot, this, &ChartsWidget::removeChart);
  connect(parser, &Parser::signalRemoved, this, &ChartsWidget::removeChart);
}

void ChartsWidget::addChart(const QString &id, const QString &sig_name) {
  const QString char_name = id + sig_name;
  if (charts.find(char_name) == charts.end()) {
    auto chart = new ChartWidget(id, sig_name, this);
    QObject::connect(chart, &ChartWidget::remove, this, &ChartsWidget::removeChart);
    main_layout->addWidget(chart);
    charts[char_name] = chart;
  }
}

void ChartsWidget::removeChart(const QString &id, const QString &sig_name) {
  if (auto it = charts.find(id + sig_name); it != charts.end()) {
    delete it->second;
    charts.erase(it);
  }
}

ChartWidget::ChartWidget(const QString &id, const QString &sig_name, QWidget *parent) : id(id), sig_name(sig_name), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QStackedLayout *stacked = new QStackedLayout();
  stacked->setStackingMode(QStackedLayout::StackAll);
  main_layout->addLayout(stacked);

  QWidget *chart_widget = new QWidget(this);
  QVBoxLayout *chart_layout = new QVBoxLayout(chart_widget);
  chart_layout->setSpacing(0);
  chart_layout->setContentsMargins(0, 0, 0, 0);

  QWidget *header = new QWidget(this);
  header->setStyleSheet("background-color:white");
  QHBoxLayout *header_layout = new QHBoxLayout(header);
  header_layout->setContentsMargins(11, 11, 11, 0);
  auto title = new QLabel(tr("%1 %2").arg(parser->getMsg(id)->name.c_str()).arg(id));
  header_layout->addWidget(title);
  header_layout->addStretch();
  zoom_label = new QLabel("", this);
  header_layout->addWidget(zoom_label);
  QPushButton *zoom_in = new QPushButton("↺", this);
  zoom_in->setToolTip(tr("reset zoom"));
  QObject::connect(zoom_in, &QPushButton::clicked, [this]() {
    chart_view->chart()->zoomReset();
  });
  header_layout->addWidget(zoom_in);

  QPushButton *remove_btn = new QPushButton("✖", this);
  QObject::connect(remove_btn, &QPushButton::clicked, [=]() {
    emit remove(id, sig_name);
  });
  header_layout->addWidget(remove_btn);
  chart_layout->addWidget(header);

  QLineSeries *series = new QLineSeries();
  series->setUseOpenGL(true);
  auto chart = new QChart();
  chart->setTitle(sig_name);
  chart->addSeries(series);
  chart->createDefaultAxes();
  chart->legend()->hide();
  QFont font;
  font.setBold(true);
  chart->setTitleFont(font);
  chart->setMargins({0, 0, 0, 0});
  chart->layout()->setContentsMargins(0, 0, 0, 0);

  chart_view = new QChartView(chart);
  chart_view->setFixedHeight(300);
  chart_view->setRenderHint(QPainter::Antialiasing);
  chart_view->setRubberBand(QChartView::HorizontalRubberBand);
  chart_layout->addWidget(chart_view);
  chart_layout->addStretch();

  stacked->addWidget(chart_widget);
  line_marker = new LineMarker(chart, this);
  stacked->addWidget(line_marker);
  line_marker->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  line_marker->raise();

  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  QObject::connect(parser, &Parser::updated, this, &ChartWidget::updateState);
  QObject::connect(parser->replay, &Replay::segmentsMerged, this, &ChartWidget::updateSeries);

  updateSeries();
}

void ChartWidget::updateState() {
  auto axis_x = dynamic_cast<QValueAxis *>(chart_view->chart()->axisX());
  double sec = parser->replay->currentSeconds();
  if (sec >= (int)axis_x->max() || sec < (int)axis_x->min()) {
    // loop replay in selected range.
    parser->replay->seekTo(axis_x->min(), false);
  }

  line_marker->update();
}

void ChartWidget::updateSeries() {
  auto events = parser->replay->events();
  if (!events) return;

  auto getSig = [=]() -> const Signal * {
    for (auto &sig : parser->getMsg(id)->sigs) {
      if (sig_name == sig.name.c_str()) return &sig;
    }
    return nullptr;
  };

  const Signal *sig = getSig();
  if (!sig) return;

  auto l = id.split(':');
  int bus = l[0].toInt();
  uint32_t address = l[1].toUInt(nullptr, 16);

  vals.clear();
  vals.reserve(3 * 60 * 100);
  double min_y = 0, max_y = 0;
  uint64_t route_start_time = parser->replay->routeStartTime();
  for (auto &evt : *events) {
    if (evt->which == cereal::Event::Which::CAN) {
      for (auto c : evt->event.getCan()) {
        if (bus == c.getSrc() && address == c.getAddress()) {
          auto dat = c.getDat();
          int64_t val = get_raw_value((uint8_t *)dat.begin(), dat.size(), *sig);
          if (sig->is_signed) {
            val -= ((val >> (sig->size - 1)) & 0x1) ? (1ULL << sig->size) : 0;
          }
          double value = val * sig->factor + sig->offset;
          if (value > max_y) max_y = value;
          if (value < min_y) min_y = value;
          double ts = (evt->event.getLogMonoTime() - route_start_time) / (double)1e9;
          vals.push_back({ts, value});
        }
      }
    }
  }
  assert(vals.size() != 0);
  QLineSeries *series = (QLineSeries *)chart_view->chart()->series()[0];
  series->replace(vals);
  if (!chart_view->chart()->isZoomed()) {
    chart_view->chart()->axisX()->setRange(vals.front().x(), vals.back().x());
  }
}

LineMarker::LineMarker(QChart *chart, QWidget *parent) : chart(chart), QWidget(parent) {}

void LineMarker::paintEvent(QPaintEvent *event) {
  auto axis_x = dynamic_cast<QValueAxis *>(chart->axisX());
  if (axis_x->max() <= axis_x->min()) return;

  double x = chart->plotArea().left() + chart->plotArea().width() * (parser->replay->currentSeconds() - axis_x->min()) / (axis_x->max() - axis_x->min());
  QPainter p(this);
  QPen pen = QPen(Qt::black);
  pen.setWidth(2);
  p.setPen(pen);
  p.drawLine(QPointF{x, 50.}, QPointF{x, (qreal)height() - 11});
}
