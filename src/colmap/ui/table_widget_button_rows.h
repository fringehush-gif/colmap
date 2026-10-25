// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <QtWidgets>
#include <algorithm>
#include <functional>

namespace colmap {

// Keep real, keyboard-accessible cell widgets only for visible rows. The table
// retains its ordinary items and selection model; the factory recreates each
// row's controls from current item data when it enters the viewport.
class TableWidgetButtonRows : public QObject {
 public:
  using Factory = std::function<QWidget*(int row, int column)>;

  TableWidgetButtonRows(QTableWidget* table,
                        int first_column,
                        int end_column,
                        Factory factory)
      : QObject(table),
        table_(table),
        first_column_(first_column),
        end_column_(end_column),
        factory_(std::move(factory)) {
    table_->viewport()->installEventFilter(this);
    connect(
        table_->verticalScrollBar(), &QScrollBar::valueChanged, this, [this]() {
          Update();
        });
    connect(
        table_->verticalHeader(), &QHeaderView::sectionResized, this, [this]() {
          ScheduleUpdate();
        });
  }

  // Call after replacing item data, including replacement at the same size.
  void Refresh() {
    for (int row = begin_; row < end_ && row < table_->rowCount(); ++row) {
      for (int col = first_column_; col < end_column_; ++col) {
        table_->removeCellWidget(row, col);
      }
    }
    begin_ = end_ = 0;
    Update();
  }

 private:
  bool eventFilter(QObject* object, QEvent* event) override {
    if (object == table_->viewport() &&
        (event->type() == QEvent::Resize || event->type() == QEvent::Show)) {
      ScheduleUpdate();
    }
    return QObject::eventFilter(object, event);
  }

  void ScheduleUpdate() {
    if (update_pending_) return;
    update_pending_ = true;
    QTimer::singleShot(0, this, [this]() {
      update_pending_ = false;
      Update();
    });
  }

  void Update() {
    if (updating_) return;
    QScopedValueRollback<bool> updating(updating_, true);
    int first = 0;
    int last = 0;
    if (table_->rowCount() > 0) {
      const int top = table_->rowAt(0);
      first = std::max(0, top - 1);
      // Before the first layout, create one row rather than treating an
      // unknown viewport as permission to instantiate the entire table.
      if (top < 0) {
        last = 1;
      } else {
        const int bottom =
            table_->rowAt(std::max(0, table_->viewport()->height() - 1));
        last = bottom < 0 ? table_->rowCount()
                          : std::min(table_->rowCount(), bottom + 2);
      }
    }
    for (int row = begin_; row < end_ && row < table_->rowCount(); ++row) {
      if (row >= first && row < last) continue;
      for (int col = first_column_; col < end_column_; ++col) {
        table_->removeCellWidget(row, col);
      }
    }
    for (int row = first; row < last; ++row) {
      if (table_->isRowHidden(row)) continue;
      for (int col = first_column_; col < end_column_; ++col) {
        if (table_->cellWidget(row, col) == nullptr) {
          if (QWidget* widget = factory_(row, col)) {
            table_->setCellWidget(row, col, widget);
          }
        }
      }
    }
    begin_ = first;
    end_ = last;
  }

  QTableWidget* table_;
  int first_column_;
  int end_column_;
  Factory factory_;
  int begin_ = 0;
  int end_ = 0;
  bool update_pending_ = false;
  bool updating_ = false;
};

}  // namespace colmap
