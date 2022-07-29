#pragma once
#include "common.h"

template<typename T>
class Identifiable {
public:
    size_t id() const { return id_value; }

    bool operator<(const T &other) const { return id_value < other.id_value; }

protected:
    Identifiable() : Identifiable(generate_id()) {}

    Identifiable(size_t id_value) : id_value(id_value) {}

private:
    static size_t generate_id() {
        static size_t s_id = 0;
        ++s_id;
        if (s_id == nil()) {
            s_id = 0;
        }
        return s_id;
    }

    const size_t id_value;
};