#include "tools/cabana/dbc/dbcfile.h"

#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>
#include <numeric>
#include <sstream>

DBCFile::DBCFile(const QString &dbc_file_name, QObject *parent) : QObject(parent) {
  QFile file(dbc_file_name);
  if (file.open(QIODevice::ReadOnly)) {
    name_ = QFileInfo(dbc_file_name).baseName();
    filename = dbc_file_name;
    // Remove auto save file extension
    if (dbc_file_name.endsWith(AUTO_SAVE_EXTENSION)) {
      filename.chop(AUTO_SAVE_EXTENSION.length());
    }
    parse(file.readAll());
  } else {
    throw std::runtime_error("Failed to open file.");
  }
}

DBCFile::DBCFile(const QString &name, const QString &content, QObject *parent) : QObject(parent), name_(name), filename("") {
  // Open from clipboard
  parse(content);
}

bool DBCFile::save() {
  assert (!filename.isEmpty());
  if (writeContents(filename)) {
    cleanupAutoSaveFile();
    return true;
  }
  return false;
}

bool DBCFile::saveAs(const QString &new_filename) {
  filename = new_filename;
  return save();
}

bool DBCFile::autoSave() {
  return !filename.isEmpty() && writeContents(filename + AUTO_SAVE_EXTENSION);
}

void DBCFile::cleanupAutoSaveFile() {
  if (!filename.isEmpty()) {
    QFile::remove(filename + AUTO_SAVE_EXTENSION);
  }
}

bool DBCFile::writeContents(const QString &fn) {
  QFile file(fn);
  if (file.open(QIODevice::WriteOnly)) {
    file.write(generateDBC().toUtf8());
    return true;
  }
  return false;
}

cabana::Signal *DBCFile::addSignal(const MessageId &id, const cabana::Signal &sig) {
  if (auto m = const_cast<cabana::Msg *>(msg(id.address))) {
    m->sigs.push_back(sig);
    m->update();
    return &m->sigs.last();
  }
  return nullptr;
}

 cabana::Signal *DBCFile::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  if (auto m = const_cast<cabana::Msg *>(msg(id))) {
    if (auto s = (cabana::Signal *)m->sig(sig_name)) {
      *s = sig;
      m->update();
      return s;
    }
  }
  return nullptr;
}

cabana::Signal *DBCFile::getSignal(const MessageId &id, const QString &sig_name) {
  auto m = msg(id);
  return m ? (cabana::Signal *)m->sig(sig_name) : nullptr;
}

void DBCFile::removeSignal(const MessageId &id, const QString &sig_name) {
  if (auto m = const_cast<cabana::Msg *>(msg(id))) {
    auto it = std::find_if(m->sigs.begin(), m->sigs.end(), [&](auto &s) { return s.name == sig_name; });
    if (it != m->sigs.end()) {
      m->sigs.erase(it);
      m->update();
    }
  }
}

void DBCFile::updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &comment) {
  auto &m = msgs[id.address];
  m.address = id.address;
  m.name = name;
  m.size = size;
  m.comment = comment;
}

void DBCFile::removeMsg(const MessageId &id) {
  msgs.erase(id.address);
}

QString DBCFile::newMsgName(const MessageId &id) {
  return QString("NEW_MSG_") + QString::number(id.address, 16).toUpper();
}

QString DBCFile::newSignalName(const MessageId &id) {
  auto m = msg(id);
  assert(m != nullptr);

  QString name;
  for (int i = 1; /**/; ++i) {
    name = QString("NEW_SIGNAL_%1").arg(i);
    if (m->sig(name) == nullptr) break;
  }
  return name;
}

const QList<uint8_t>& DBCFile::mask(const MessageId &id) const {
  auto m = msg(id);
  return m ? m->mask : empty_mask;
}

const cabana::Msg *DBCFile::msg(uint32_t address) const {
  auto it = msgs.find(address);
  return it != msgs.end() ? &it->second : nullptr;
}

const cabana::Msg* DBCFile::msg(const QString &name) {
  auto it = std::find_if(msgs.cbegin(), msgs.cend(), [&name](auto &m) { return m.second.name == name; });
  return it != msgs.cend() ? &(it->second) : nullptr;
}

QStringList DBCFile::signalNames() const {
  // Used for autocompletion
  QStringList ret;
  for (auto const& [_, msg] : msgs) {
    for (auto sig: msg.getSignals()) {
      ret << sig->name;
    }
  }
  ret.sort();
  ret.removeDuplicates();
  return ret;
}

int DBCFile::signalCount(const MessageId &id) const {
  if (msgs.count(id.address) == 0) return 0;
  return msgs.at(id.address).sigs.size();
}

int DBCFile::signalCount() const {
  return std::accumulate(msgs.cbegin(), msgs.cend(), 0, [](int &n, const auto &m) { return n + m.second.sigs.size(); });
}

void DBCFile::parse(const QString &content) {
  static QRegularExpression bo_regexp(R"(^BO_ (\w+) (\w+) *: (\w+) (\w+))");
  static QRegularExpression sg_regexp(R"(^SG_ (\w+) : (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static QRegularExpression sgm_regexp(R"(^SG_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static QRegularExpression msg_comment_regexp(R"(^CM_ BO_ *(\w+) *\"(.*)\";)");
  static QRegularExpression sg_comment_regexp(R"(^CM_ SG_ *(\w+) *(\w+) *\"(.*)\";)");
  static QRegularExpression val_regexp(R"(VAL_ (\w+) (\w+) (\s*[-+]?[0-9]+\s+\".+?\"[^;]*))");

  int line_num = 0;
  QString line;
  auto dbc_assert = [&line_num, &line, this](bool condition, const QString &msg = "") {
    if (!condition) throw std::runtime_error(QString("[%1:%2]%3: %4").arg(filename).arg(line_num).arg(msg).arg(line).toStdString());
  };
  auto get_sig = [this](uint32_t address, const QString &name) -> cabana::Signal * {
    auto m = (cabana::Msg *)msg(address);
    return m ? (cabana::Signal *)m->sig(name) : nullptr;
  };

  msgs.clear();
  QTextStream stream((QString *)&content);
  cabana::Msg *current_msg = nullptr;
  int multiplexor_cnt = 0;
  while (!stream.atEnd()) {
    ++line_num;
    line = stream.readLine().trimmed();
    if (line.startsWith("BO_ ")) {
      auto match = bo_regexp.match(line);
      dbc_assert(match.hasMatch());
      auto address = match.captured(1).toUInt();
      dbc_assert(msgs.count(address) == 0, QString("Duplicate message address: %1").arg(address));
      current_msg = &msgs[address];
      current_msg->address = address;
      current_msg->name = match.captured(2);
      current_msg->size = match.captured(3).toULong();
    } else if (line.startsWith("SG_ ")) {
      int offset = 0;
      auto match = sg_regexp.match(line);
      if (!match.hasMatch()) {
        match = sgm_regexp.match(line);
        offset = 1;
      }
      dbc_assert(match.hasMatch());
      dbc_assert(current_msg, "No Message");
      auto name = match.captured(1);
      dbc_assert(current_msg->sig(name) == nullptr, "Duplicate signal name");
      cabana::Signal s{};
      if (offset == 1) {
        auto indicator = match.captured(2);
        if (indicator == "M") {
          // Only one signal within a single message can be the multiplexer switch.
          dbc_assert(++multiplexor_cnt < 2, "Multiple multiplexor");
          s.type = cabana::Signal::Type::Multiplexor;
        } else {
          dbc_assert(multiplexor_cnt == 1, "No multiplexor");
          s.type = cabana::Signal::Type::Multiplexed;
          s.multiplex_value = indicator.mid(1).toInt();
        }
      }
      s.name = name;
      s.start_bit = match.captured(offset + 2).toInt();
      s.size = match.captured(offset + 3).toInt();
      s.is_little_endian = match.captured(offset + 4).toInt() == 1;
      s.is_signed = match.captured(offset + 5) == "-";
      s.factor = match.captured(offset + 6).toDouble();
      s.offset = match.captured(offset + 7).toDouble();
      if (s.is_little_endian) {
        s.lsb = s.start_bit;
        s.msb = s.start_bit + s.size - 1;
      } else {
        s.lsb = bigEndianStartBitsIndex(bigEndianBitIndex(s.start_bit) + s.size - 1);
        s.msb = s.start_bit;
      }
      s.min = match.captured(8 + offset).toDouble();
      s.max = match.captured(9 + offset).toDouble();
      s.unit = match.captured(10 + offset);
      current_msg->sigs.push_back(s);
    } else if (line.startsWith("VAL_ ")) {
      auto match = val_regexp.match(line);
      dbc_assert(match.hasMatch());
      if (auto s = get_sig(match.captured(1).toUInt(), match.captured(2))) {
        QStringList desc_list = match.captured(3).trimmed().split('"');
        for (int i = 0; i < desc_list.size(); i += 2) {
          auto val = desc_list[i].trimmed();
          if (!val.isEmpty() && (i + 1) < desc_list.size()) {
            auto desc = desc_list[i + 1].trimmed();
            s->val_desc.push_back({val.toDouble(), desc});
          }
        }
      }
    } else if (line.startsWith("CM_ BO_")) {
      auto match = msg_comment_regexp.match(line);
      dbc_assert(match.hasMatch());
      if (auto m = (cabana::Msg *)msg(match.captured(1).toUInt())) {
        m->comment = match.captured(2).trimmed();
      }
    } else if (line.startsWith("CM_ SG_ ")) {
      auto match = sg_comment_regexp.match(line);
      dbc_assert(match.hasMatch());
      if (auto s = get_sig(match.captured(1).toUInt(), match.captured(2))) {
        s->comment = match.captured(3).trimmed();
      }
    }
  }

  for (auto &[_, m] : msgs) {
    m.update();
  }
}

QString DBCFile::generateDBC() {
  QString dbc_string, signal_comment, message_comment, val_desc;
  for (const auto &[address, m] : msgs) {
    dbc_string += QString("BO_ %1 %2: %3 XXX\n").arg(address).arg(m.name).arg(m.size);
    if (!m.comment.isEmpty()) {
      message_comment += QString("CM_ BO_ %1 \"%2\";\n").arg(address).arg(m.comment);
    }
    for (auto sig : m.getSignals()) {
      QString multiplexer_indicator;
      if (sig->type == cabana::Signal::Type::Multiplexor) {
        multiplexer_indicator = "M ";
      } else if (sig->type == cabana::Signal::Type::Multiplexed) {
        multiplexer_indicator = QString("m%1 ").arg(sig->multiplex_value);
      }
      dbc_string += QString(" SG_ %1 %2: %3|%4@%5%6 (%7,%8) [%9|%10] \"%11\" XXX\n")
                        .arg(sig->name)
                        .arg(multiplexer_indicator)
                        .arg(sig->start_bit)
                        .arg(sig->size)
                        .arg(sig->is_little_endian ? '1' : '0')
                        .arg(sig->is_signed ? '-' : '+')
                        .arg(doubleToString(sig->factor))
                        .arg(doubleToString(sig->offset))
                        .arg(doubleToString(sig->min))
                        .arg(doubleToString(sig->max))
                        .arg(sig->unit);
      if (!sig->comment.isEmpty()) {
        signal_comment += QString("CM_ SG_ %1 %2 \"%3\";\n").arg(address).arg(sig->name).arg(sig->comment);
      }
      if (!sig->val_desc.isEmpty()) {
        QStringList text;
        for (auto &[val, desc] : sig->val_desc) {
          text << QString("%1 \"%2\"").arg(val).arg(desc);
        }
        val_desc += QString("VAL_ %1 %2 %3;\n").arg(address).arg(sig->name).arg(text.join(" "));
      }
    }
    dbc_string += "\n";
  }
  return dbc_string + message_comment + signal_comment + val_desc;
}
