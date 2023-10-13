#include "tools/cabana/messageswidget.h"

#include <algorithm>
#include <limits>

#include <QCheckBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>

#include "tools/cabana/commands.h"

MessagesWidget::MessagesWidget(QWidget *parent) : menu(new QMenu(this)), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);
  // toolbar
  main_layout->addWidget(createToolBar());
  // message table
  main_layout->addWidget(view = new MessageView(this));
  view->setItemDelegate(delegate = new MessageBytesDelegate(view, settings.multiple_lines_bytes));
  view->setHeader(header = new MessageViewHeader(this));
  view->setModel(model = new MessageListModel(this));
  view->setSortingEnabled(true);
  view->sortByColumn(MessageListModel::Column::NAME, Qt::AscendingOrder);
  view->setAllColumnsShowFocus(true);
  view->setEditTriggers(QAbstractItemView::NoEditTriggers);
  view->setItemsExpandable(false);
  view->setIndentation(0);
  view->setRootIsDecorated(false);

  // Must be called before setting any header parameters to avoid overriding
  restoreHeaderState(settings.message_header_state);
  header->setSectionsMovable(true);
  header->setSectionResizeMode(MessageListModel::Column::DATA, QHeaderView::Fixed);
  header->setStretchLastSection(true);
  // Header context menu
  header->setContextMenuPolicy(Qt::CustomContextMenu);

  // signals/slots
  QObject::connect(menu, &QMenu::aboutToShow, this, &MessagesWidget::menuAboutToShow);
  QObject::connect(header, &MessageViewHeader::filtersUpdated, model, &MessageListModel::setFilterStrings);
  QObject::connect(header, &MessageViewHeader::customContextMenuRequested, this, &MessagesWidget::headerContextMenuEvent);
  QObject::connect(view->horizontalScrollBar(), &QScrollBar::valueChanged, header, &MessageViewHeader::updateHeaderPositions);
  QObject::connect(can, &AbstractStream::msgsReceived, model, &MessageListModel::msgsReceived);
  QObject::connect(dbc(), &DBCManager::changed, model, &MessageListModel::dbcModified);
  QObject::connect(model, &MessageListModel::modelReset, [this]() {
    if (current_msg_id) {
      selectMessage(*current_msg_id);
    }
    view->updateBytesSectionSize();
    updateTitle();
  });
  QObject::connect(view->selectionModel(), &QItemSelectionModel::currentChanged, [=](const QModelIndex &current, const QModelIndex &previous) {
    if (current.isValid() && current.row() < model->msgs.size()) {
      auto &id = model->msgs[current.row()];
      if (!current_msg_id || id != *current_msg_id) {
        current_msg_id = id;
        emit msgSelectionChanged(*current_msg_id);
      }
    }
  });

  updateSuppressedButtons(0);

  setWhatsThis(tr(R"(
    <b>Message View</b><br/>
    <!-- TODO: add descprition here -->
    <span style="color:gray">Byte color</span><br />
    <span style="color:gray;">■ </span> constant changing<br />
    <span style="color:blue;">■ </span> increasing<br />
    <span style="color:red;">■ </span> decreasing
  )"));
}

QToolBar *MessagesWidget::createToolBar() {
  QToolBar *toolbar = new QToolBar(this);
  int icon_size = style()->pixelMetric(QStyle::PM_SmallIconSize);
  toolbar->setIconSize({icon_size, icon_size});

  toolbar->addAction(tr("&Suppress Highlighted"), [this]() {
    size_t cnt = can->suppressHighlighted();
    updateSuppressedButtons(cnt);
  });
  suppress_clear = toolbar->addAction(tr("&Clear"), [this]() {
    can->clearSuppressed();
    updateSuppressedButtons(0);
  });

  suppress_clear->setToolTip(tr("Clear suppressed ighlighted"));
  auto suppress_signals = toolbar->addAction(tr("Suppress Signals"),
                                             can, &AbstractStream::suppressDefinedSignals);
  suppress_signals->setCheckable(true);
  suppress_signals->setChecked(settings.suppress_defined_signals);

  QLabel *stretch_label = new QLabel(this);
  stretch_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  toolbar->addWidget(stretch_label);

  auto view_menu = toolbar->addAction(utils::icon("three-dots"), tr("View..."));
  view_menu->setMenu(menu);
  qobject_cast<QToolButton *>(toolbar->widgetForAction(view_menu))->setPopupMode(QToolButton::InstantPopup);
  return toolbar;
}

void MessagesWidget::updateTitle() {
  auto count = std::accumulate(
      model->msgs.begin(), model->msgs.end(), std::pair<size_t, size_t>(),
      [](const auto &pair, auto id) {
        auto m = dbc()->msg(id);
        return m ? std::make_pair(pair.first + 1, pair.second + m->sigs.size()) : pair;
      });
  emit titleChanged(tr("%1 Messages (%2 DBC Messages, %3 Signals)")
                      .arg(model->msgs.size()).arg(count.first).arg(count.second));
}

void MessagesWidget::selectMessage(const MessageId &msg_id) {
  auto it = std::find(model->msgs.cbegin(), model->msgs.cend(), msg_id);
  if (it != model->msgs.cend()) {
    view->setCurrentIndex(model->index(std::distance(model->msgs.cbegin(), it), 0));
  }
}

void MessagesWidget::updateSuppressedButtons(size_t n) {
  suppress_clear->setEnabled(n > 0);
  suppress_clear->setText(n > 0 ? tr("&Clear (%1)").arg(n) : tr("&Clear"));
}

void MessagesWidget::headerContextMenuEvent(const QPoint &pos) {
  menu->exec(header->mapToGlobal(pos));
}

void MessagesWidget::menuAboutToShow() {
  menu->clear();
  for (int i = 0; i < header->count(); ++i) {
    int logical_index = header->logicalIndex(i);
    auto action = menu->addAction(model->headerData(logical_index, Qt::Horizontal).toString(),
                                  [=](bool checked) { header->setSectionHidden(logical_index, !checked); });
    action->setCheckable(true);
    action->setChecked(!header->isSectionHidden(logical_index));
    // Can't hide the name column
    action->setEnabled(logical_index > 0);
  }
  menu->addSeparator();
  auto action = menu->addAction(tr("Mutlti-Line bytes"), this, &MessagesWidget::setMultiLineBytes);
  action->setCheckable(true);
  action->setChecked(settings.multiple_lines_bytes);
}

void MessagesWidget::setMultiLineBytes(bool multi) {
  settings.multiple_lines_bytes = multi;
  delegate->setMultipleLines(multi);
  view->updateBytesSectionSize();
  view->doItemsLayout();
}

// MessageListModel
MessageListModel::MessageListModel(QObject *parent) : QAbstractTableModel(parent) {
  sort_timer.setSingleShot(true);
  sort_timer.callOnTimeout(this, &MessageListModel::filterAndSort);
}

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  static const QVariant titles[] = {tr("Name"), tr("Bus"), tr("Address"), tr("Freq"), tr("Count"), tr("Bytes")};
  return orientation == Qt::Horizontal && role == Qt::DisplayRole ? titles[section] : QVariant();
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid() || index.row() >= msgs.size()) return {};

  auto getFreq = [](const CanData &d) {
    if (d.freq > 0 && ((can->currentSec() - can->toSeconds(d.mono_time) - 1.0 / settings.fps) < (5.0 / d.freq))) {
      return d.freq >= 0.95 ? QString::number(std::nearbyint(d.freq)) : QString::number(d.freq, 'f', 2);
    } else {
      return QString("--");
    }
  };

  const auto &id = msgs[index.row()];
  auto &can_data = can->lastMessage(id);
  if (role == Qt::DisplayRole) {
    switch (index.column()) {
      case Column::NAME: return msgName(id);
      case Column::SOURCE: return id.source != INVALID_SOURCE ? QString::number(id.source) : "N/A";
      case Column::ADDRESS: return QString::number(id.address, 16);
      case Column::FREQ: return id.source != INVALID_SOURCE ? getFreq(can_data) : "N/A";
      case Column::COUNT: return id.source != INVALID_SOURCE ? QString::number(can_data.count) : "N/A";
      case Column::DATA: return id.source != INVALID_SOURCE ? "" : "N/A";
    }
  } else if (role == ColorsRole) {
    return QVariant::fromValue((void*)(&can_data.colors));
  } else if (role == BytesRole && index.column() == Column::DATA && id.source != INVALID_SOURCE) {
    return QVariant::fromValue((void*)(&can_data.dat));
  } else if (role == Qt::ToolTipRole && index.column() == Column::NAME) {
    auto msg = dbc()->msg(id);
    auto tooltip = msg ? msg->name : UNTITLED;
    if (msg && !msg->comment.isEmpty()) tooltip += "<br /><span style=\"color:gray;\">" + msg->comment + "</span>";
    return tooltip;
  }
  return {};
}

void MessageListModel::setFilterStrings(const QMap<int, QString> &filters) {
  filter_str = filters;
  filterAndSort();
}

void MessageListModel::dbcModified() {
  dbc_address.clear();
  for (const auto &[_, m] : dbc()->getMessages(-1)) {
    dbc_address.insert(m.address);
  }
  filterAndSort();
}

void MessageListModel::sortMessages(std::vector<MessageId> &new_msgs) {
  auto do_sort = [order = sort_order](std::vector<MessageId> &new_msgs, auto proj) {
    std::sort(new_msgs.begin(), new_msgs.end(), [order, proj = std::move(proj)](auto &l, auto &r) {
      return order == Qt::AscendingOrder ? proj(l) < proj(r) : proj(l) > proj(r);
    });
  };
  switch (sort_column) {
    case Column::NAME: do_sort(new_msgs, [](auto &id) { return std::make_pair(msgName(id), id); }); break;
    case Column::SOURCE: do_sort(new_msgs, [](auto &id) { return std::tie(id.source, id); }); break;
    case Column::ADDRESS: do_sort(new_msgs, [](auto &id) { return std::tie(id.address, id);}); break;
    case Column::FREQ: do_sort(new_msgs, [](auto &id) { return std::tie(can->lastMessage(id).freq, id); }); break;
    case Column::COUNT: do_sort(new_msgs, [](auto &id) { return std::tie(can->lastMessage(id).count, id); }); break;
  }
  last_sort_ts = millis_since_boot();
}

static bool parseRange(const QString &filter, uint32_t value, int base = 10) {
  // Parse out filter string into a range (e.g. "1" -> {1, 1}, "1-3" -> {1, 3}, "1-" -> {1, inf})
  unsigned int min = std::numeric_limits<unsigned int>::min();
  unsigned int max = std::numeric_limits<unsigned int>::max();
  auto s = filter.split('-');
  bool ok = s.size() >= 1 && s.size() <= 2;
  if (ok && !s[0].isEmpty()) min = s[0].toUInt(&ok, base);
  if (ok && s.size() == 1) {
    max = min;
  } else if (ok && s.size() == 2 && !s[1].isEmpty()) {
    max = s[1].toUInt(&ok, base);
  }
  return ok && value >= min && value <= max;
}

bool MessageListModel::matchMessage(const MessageId &id, const CanData &data, const QMap<int, QString> &filters) {
  bool match = true;
  for (auto it = filters.cbegin(); it != filters.cend() && match; ++it) {
    const QString &txt = it.value();
    QRegularExpression re(txt, QRegularExpression::CaseInsensitiveOption | QRegularExpression::DotMatchesEverythingOption);
    switch (it.key()) {
      case Column::NAME: {
        const auto msg = dbc()->msg(id);
        match = re.match(msg ? msg->name : UNTITLED).hasMatch();
        match = match || (msg && std::any_of(msg->sigs.cbegin(), msg->sigs.cend(),
                                             [&re](const auto &s) { return re.match(s->name).hasMatch(); }));
        break;
      }
      case Column::SOURCE:
        match = parseRange(txt, id.source);
        break;
      case Column::ADDRESS: {
        match = re.match(QString::number(id.address, 16)).hasMatch();
        match = match || parseRange(txt, id.address, 16);
        break;
      }
      case Column::FREQ:
        // TODO: Hide stale messages?
        match = parseRange(txt, data.freq);
        break;
      case Column::COUNT:
        match = parseRange(txt, data.count);
        break;
      case Column::DATA: {
        QString hex = utils::toHex(data.dat);
        match = hex.contains(txt, Qt::CaseInsensitive);
        match = match || re.match(hex).hasMatch();
        match = match || re.match(utils::toHex(data.dat, ' ')).hasMatch();
        break;
      }
    }
  }
  return match;
}

void MessageListModel::filterAndSort() {
  std::vector<MessageId> new_msgs;
  new_msgs.reserve(can->lastMessages().size() + dbc_address.size());

  auto address = dbc_address;
  for (const auto &[id, m] : can->lastMessages()) {
    if (filter_str.isEmpty() || matchMessage(id, m, filter_str)) {
      new_msgs.push_back(id);
    }
    address.erase(id.address);
  }

  // merge all DBC messages
  for (const auto &addr : address) {
    MessageId id{.source = INVALID_SOURCE, .address = addr};
    if (filter_str.isEmpty() || matchMessage(id, {}, filter_str)) {
      new_msgs.push_back(id);
    }
  }

  sortMessages(new_msgs);

  if (msgs != new_msgs) {
    beginResetModel();
    msgs = std::move(new_msgs);
    endResetModel();
  }
}

void MessageListModel::msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids) {
  if (has_new_ids) {
    sort_timer.start(110);
  } else if (!filter_str.empty()) {
    bool resort = (filter_str.contains(Column::FREQ) || filter_str.contains(Column::COUNT) ||
                   filter_str.contains(Column::DATA));
    if (resort && ((millis_since_boot() - last_sort_ts) >= 1000)) {
      filterAndSort();
      return;
    }
  }

  for (int i = 0; i < msgs.size(); ++i) {
    if (!new_msgs || new_msgs->count(msgs[i])) {
      for (int col = Column::FREQ; col < columnCount(); ++col)
        emit dataChanged(index(i, col), index(i, col), {Qt::DisplayRole});
    }
  }
}

void MessageListModel::sort(int column, Qt::SortOrder order) {
  if (column != columnCount() - 1) {
    sort_column = column;
    sort_order = order;
    filterAndSort();
  }
}

// MessageView

void MessageView::drawRow(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  QTreeView::drawRow(painter, option, index);
  const int gridHint = style()->styleHint(QStyle::SH_Table_GridLineColor, &option, this);
  const QColor gridColor = QColor::fromRgba(static_cast<QRgb>(gridHint));
  QPen old_pen = painter->pen();
  painter->setPen(gridColor);
  painter->drawLine(option.rect.left(), option.rect.bottom(), option.rect.right(), option.rect.bottom());

  auto y = option.rect.y();
  painter->translate(visualRect(model()->index(0, 0)).x() - indentation() - .5, -.5);
  for (int i = 0; i < header()->count(); ++i) {
    painter->translate(header()->sectionSize(header()->logicalIndex(i)), 0);
    painter->drawLine(0, y, 0, y + option.rect.height());
  }
  painter->setPen(old_pen);
  painter->resetTransform();
}

void MessageView::dataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles) {
  // Bypass the slow call to QTreeView::dataChanged.
  // QTreeView::dataChanged will invalidate the height cache and that's what we don't need in MessageView.
  QTreeView::dataChanged(topLeft, bottomRight, roles);
}

void MessageView::updateBytesSectionSize() {
  int max_bytes = 8;
  for (const auto &[_, m] : can->lastMessages()) {
    max_bytes = std::max<int>(max_bytes, m.dat.size());
  }
  int width = ((MessageBytesDelegate *)itemDelegate())->sizeForBytes(max_bytes).width();
  header()->resizeSection(MessageListModel::Column::DATA, width);
}

// MessageViewHeader

MessageViewHeader::MessageViewHeader(QWidget *parent) : QHeaderView(Qt::Horizontal, parent) {
  QObject::connect(this, &QHeaderView::sectionResized, this, &MessageViewHeader::updateHeaderPositions);
  QObject::connect(this, &QHeaderView::sectionMoved, this, &MessageViewHeader::updateHeaderPositions);
}

void MessageViewHeader::updateFilters() {
  QMap<int, QString> filters;
  for (int i = 0; i < count(); i++) {
    if (editors[i] && !editors[i]->text().isEmpty()) {
      filters[i] = editors[i]->text();
    }
  }
  emit filtersUpdated(filters);
}

void MessageViewHeader::updateHeaderPositions() {
  QSize sz = QHeaderView::sizeHint();
  for (int i = 0; i < count(); i++) {
    if (editors[i]) {
      int h = editors[i]->sizeHint().height();
      editors[i]->move(sectionViewportPosition(i), sz.height());
      editors[i]->resize(sectionSize(i), h);
      editors[i]->setHidden(isSectionHidden(i));
    }
  }
}

void MessageViewHeader::updateGeometries() {
  for (int i = 0; i < count(); i++) {
    if (!editors[i]) {
      QString column_name = model()->headerData(i, Qt::Horizontal, Qt::DisplayRole).toString();
      editors[i] = new QLineEdit(this);
      editors[i]->setClearButtonEnabled(true);
      editors[i]->setPlaceholderText(tr("Filter %1").arg(column_name));

      QObject::connect(editors[i], &QLineEdit::textChanged, this, &MessageViewHeader::updateFilters);
    }
  }
  setViewportMargins(0, 0, 0, editors[0] ? editors[0]->sizeHint().height() : 0);

  QHeaderView::updateGeometries();
  updateHeaderPositions();
}

QSize MessageViewHeader::sizeHint() const {
  QSize sz = QHeaderView::sizeHint();
  if (editors[0])
    sz.setHeight(sz.height() + editors[0]->minimumSizeHint().height() + 1);
  return sz;
}
