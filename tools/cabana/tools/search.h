#pragma once

#include <QDialog>
#include <QPushButton>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QDoubleValidator>
#include <QComboBox>
#include <QSpinBox>
#include <QLabel>
#include <QTableWidget>
#include <QTableWidgetItem>

#include <iostream>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

uint64_t getBitValue(uint64_t val, int offset, int size);

enum ScanType {
  ExactValue,
  BiggerThan,
  SmallerThan,
  ValueBetween,
  IncreasedValue,
  IncreasedValueBy,
  DecreasedValue,
  DecreasedValueBy,
  ChangedValue,
  UnchangedValue,
  UnknownInitialValue
};

class SearchSignal {
    public:
        SearchSignal(MessageId _messageID, size_t _offset, size_t _size) : messageID(_messageID), offset(_offset), size(_size) {}

        MessageId messageID;
        size_t offset;
        size_t size;

        uint64_t previousValue;
    
        uint64_t getValue(){
            auto msg = can->last_msgs[messageID];
            uint64_t* data = (uint64_t*)(msg.dat.data());
            return getBitValue(*data, offset, size);
        }
};

class SignalFilterer {
  public:
    virtual bool signalMatches(SearchSignal sig) = 0;

    virtual ~SignalFilterer() = default;

    std::vector<SearchSignal> filter(std::vector<SearchSignal> in) {
      std::vector<SearchSignal> ret;

      std::copy_if(in.begin(), in.end(), std::back_inserter(ret), [=] (SearchSignal sig) { return signalMatches(sig); });

      return ret;
    }
};

class SearchDlg : public QDialog {
  Q_OBJECT

public:
  SearchDlg(QWidget *parent);

private:
  void firstScan();
  void nextScan();
  void undoScan();

  void update();
  void updateRowData();
  void setRowData(int row, QString msgID, QString bitRange, QString currentValue, QString previousValue);

  std::vector<ScanType> enabledScanTypes();

  SignalFilterer* getCurrentFilterer();

  uint32_t scan_bits_range_min = 1;
  uint32_t scan_bits_range_max = 32;

  uint64_t scan_value1 = 0;
  uint64_t scan_value2 = 0;

  std::vector<SearchSignal> filteredSignals;

  ScanType selectedScanType;

  QLabel* numberOfSigsLabel;
  QComboBox *scan_type;
  QPushButton *first_scan_button;
  QPushButton *next_scan_button;
  QPushButton *undo_scan_button;

  QTableWidget *data_table;

  // Search history, at a specific time
  std::vector<std::tuple<SignalFilterer*, double>> searchHistory;

  bool scanningStarted();
};

class ZeroInputSignalFilterer : public SignalFilterer {
  public:
    ZeroInputSignalFilterer(){

    }
};

class SingleInputSignalFilterer : public SignalFilterer {
  public:
    SingleInputSignalFilterer(uint64_t _value) : value(_value){

    }
  protected:
    uint64_t value;
};

class DoubleInputSignalFilterer : public SignalFilterer {
  public:
    DoubleInputSignalFilterer(uint64_t _value1, uint64_t _value2) : value1(_value1), value2(_value2){

    }
  protected:
    uint64_t value1;
    uint64_t value2;
};

class ExactValueSignalFilterer : public SingleInputSignalFilterer {
  using SingleInputSignalFilterer::SingleInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue() == value;
  }
};

class BiggerThanSignalFilterer : public SingleInputSignalFilterer {
  using SingleInputSignalFilterer::SingleInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue() > value;
  }
};

class SmallerThanSignalFilterer : public SingleInputSignalFilterer {
  using SingleInputSignalFilterer::SingleInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue() < value;
  }
};

class UnknownInitialValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return true;
  }
};

class IncreasedValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue() > sig.previousValue;
  }
};

class DecreasedValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue() < sig.previousValue;
  }
};

class ChangedValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue() != sig.previousValue;
  }
};

class UnchangedValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue() == sig.previousValue;
  }
};