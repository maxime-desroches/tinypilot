/****************************************************************************
** Meta object code from reading C++ file 'api.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "api.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'api.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_CommaApi_t {
    QByteArrayData data[1];
    char stringdata0[9];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CommaApi_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CommaApi_t qt_meta_stringdata_CommaApi = {
    {
QT_MOC_LITERAL(0, 0, 8) // "CommaApi"

    },
    "CommaApi"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CommaApi[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void CommaApi::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject CommaApi::staticMetaObject = { {
    &QObject::staticMetaObject,
    qt_meta_stringdata_CommaApi.data,
    qt_meta_data_CommaApi,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *CommaApi::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CommaApi::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CommaApi.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int CommaApi::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_RequestRepeater_t {
    QByteArrayData data[9];
    char stringdata0[117];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RequestRepeater_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RequestRepeater_t qt_meta_stringdata_RequestRepeater = {
    {
QT_MOC_LITERAL(0, 0, 15), // "RequestRepeater"
QT_MOC_LITERAL(1, 16, 16), // "receivedResponse"
QT_MOC_LITERAL(2, 33, 0), // ""
QT_MOC_LITERAL(3, 34, 8), // "response"
QT_MOC_LITERAL(4, 43, 14), // "failedResponse"
QT_MOC_LITERAL(5, 58, 11), // "errorString"
QT_MOC_LITERAL(6, 70, 15), // "timeoutResponse"
QT_MOC_LITERAL(7, 86, 14), // "requestTimeout"
QT_MOC_LITERAL(8, 101, 15) // "requestFinished"

    },
    "RequestRepeater\0receivedResponse\0\0"
    "response\0failedResponse\0errorString\0"
    "timeoutResponse\0requestTimeout\0"
    "requestFinished"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RequestRepeater[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x06 /* Public */,
       4,    1,   42,    2, 0x06 /* Public */,
       6,    1,   45,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       7,    0,   48,    2, 0x08 /* Private */,
       8,    0,   49,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void, QMetaType::QString,    5,
    QMetaType::Void, QMetaType::QString,    5,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void RequestRepeater::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<RequestRepeater *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->receivedResponse((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->failedResponse((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 2: _t->timeoutResponse((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 3: _t->requestTimeout(); break;
        case 4: _t->requestFinished(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (RequestRepeater::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RequestRepeater::receivedResponse)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (RequestRepeater::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RequestRepeater::failedResponse)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (RequestRepeater::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RequestRepeater::timeoutResponse)) {
                *result = 2;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject RequestRepeater::staticMetaObject = { {
    &QObject::staticMetaObject,
    qt_meta_stringdata_RequestRepeater.data,
    qt_meta_data_RequestRepeater,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *RequestRepeater::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RequestRepeater::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_RequestRepeater.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int RequestRepeater::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void RequestRepeater::receivedResponse(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void RequestRepeater::failedResponse(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void RequestRepeater::timeoutResponse(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
