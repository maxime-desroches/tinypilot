#include "tools/cabana/messageswidget.h"

#include <QComboBox>
#include <QCompleter>
#include <QHeaderView>
#include <QPushButton>
#include <QVBoxLayout>

#include "tools/cabana/dbcmanager.h"

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *dbc_file_layout = new QHBoxLayout();
  QComboBox *combo = new QComboBox(this);
  auto dbc_names = dbc()->allDBCNames();
  for (const auto &name : dbc_names) {
    combo->addItem(QString::fromStdString(name));
  }
  combo->setEditable(true);
  combo->setCurrentText(QString());
  combo->setInsertPolicy(QComboBox::NoInsert);
  combo->completer()->setCompletionMode(QCompleter::PopupCompletion);
  QFont font;
  font.setBold(true);
  combo->lineEdit()->setFont(font);
  dbc_file_layout->addWidget(combo);

  dbc_file_layout->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save DBC"), this);
  dbc_file_layout->addWidget(save_btn);
  main_layout->addLayout(dbc_file_layout);

  filter = new QLineEdit(this);
  filter->setPlaceholderText(tr("filter messages"));
  main_layout->addWidget(filter);

  table_widget = new QTableWidget(this);
  table_widget->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  table_widget->setColumnCount(4);
  table_widget->setColumnWidth(0, 250);
  table_widget->setColumnWidth(1, 80);
  table_widget->setColumnWidth(2, 80);
  table_widget->setHorizontalHeaderLabels({tr("Name"), tr("ID"), tr("Count"), tr("Bytes")});
  table_widget->horizontalHeader()->setStretchLastSection(true);
  main_layout->addWidget(table_widget);

  QObject::connect(can, &CANMessages::updated, this, &MessagesWidget::updateState);
  QObject::connect(combo, SIGNAL(activated(const QString &)), SLOT(dbcSelectionChanged(const QString &)));
  QObject::connect(save_btn, &QPushButton::clicked, [=]() {
    // TODO: save DBC to file
  });
  QObject::connect(table_widget, &QTableWidget::itemSelectionChanged, [=]() {
    emit msgSelectionChanged(table_widget->selectedItems()[1]->text());
  });

  // For test purpose
  combo->setCurrentText("toyota_nodsu_pt_generated");
}

void MessagesWidget::dbcSelectionChanged(const QString &dbc_file) {
  dbc()->open(dbc_file);
  // update detailwidget
  auto selected = table_widget->selectedItems();
  if (!selected.isEmpty())
    emit msgSelectionChanged(selected[1]->text());
}

void MessagesWidget::updateState() {
  auto getTableItem = [=](int row, int col) -> QTableWidgetItem * {
    auto item = table_widget->item(row, col);
    if (!item) {
      item = new QTableWidgetItem();
      item->setFlags(item->flags() ^ Qt::ItemIsEditable);
      table_widget->setItem(row, col, item);
    }
    return item;
  };

  table_widget->setRowCount(can->can_msgs.size());
  int i = 0;
  QString name, untitled = tr("untitled");
  const QString filter_str = filter->text();
  for (const auto &[_, msgs] : can->can_msgs) {
    const auto &c = msgs.back();

    if (auto msg = dbc()->msg(c.address)) {
      name = msg->name.c_str();
    } else {
      name = untitled;
    }
    if (!filter_str.isEmpty() && !name.contains(filter_str, Qt::CaseInsensitive)) {
      table_widget->hideRow(i++);
      continue;
    }

    getTableItem(i, 0)->setText(name);
    getTableItem(i, 1)->setText(c.id);
    getTableItem(i, 2)->setText(QString::number(c.count));
    getTableItem(i, 3)->setText(toHex(c.dat));
    table_widget->showRow(i);
    i++;
  }
  if (table_widget->currentRow() == -1) {
    table_widget->selectRow(0);
  }
}
