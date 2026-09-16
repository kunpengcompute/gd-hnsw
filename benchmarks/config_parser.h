/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

namespace gd_hnsw {

class ConfigParser {
public:
    explicit ConfigParser(const std::string &path)
    {
        std::ifstream f(path);
        if (!f.is_open())
            return;

        std::string line;
        while (std::getline(f, line)) {
            // Trim leading whitespace
            size_t start = line.find_first_not_of(" \t\r");
            if (start == std::string::npos)
                continue; // blank line
            if (line[start] == '#')
                continue; // comment

            // Trim trailing whitespace
            size_t end = line.find_last_not_of(" \t\r");
            std::string trimmed = line.substr(start, end - start + 1);

            // Split on first '='
            size_t eq = trimmed.find('=');
            if (eq == std::string::npos)
                continue;

            std::string key = trimmed.substr(0, eq);
            std::string val = trimmed.substr(eq + 1);

            // Trim key
            size_t k_end = key.find_last_not_of(" \t");
            if (k_end == std::string::npos)
                continue;
            key = key.substr(0, k_end + 1);

            // Trim value
            size_t v_start = val.find_first_not_of(" \t");
            if (v_start == std::string::npos) {
                entries_[key] = "";
            } else {
                val = val.substr(v_start);
                entries_[key] = val;
            }
        }
        ok_ = !entries_.empty();
    }

    bool ok() const { return ok_; }

    int get_int(const std::string &key, int def = 0) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return def;
        return std::stoi(it->second);
    }

    uint32_t get_uint(const std::string &key, uint32_t def = 0) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return def;
        return static_cast<uint32_t>(std::stoul(it->second));
    }

    double get_double(const std::string &key, double def = 0.0) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return def;
        return std::stod(it->second);
    }

    bool get_bool(const std::string &key, bool def = false) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return def;
        const std::string &v = it->second;
        return v == "true" || v == "1" || v == "yes";
    }

    std::string get_str(const std::string &key, const std::string &def = "") const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return def;
        return it->second;
    }

    bool has(const std::string &key) const { return entries_.find(key) != entries_.end(); }

private:
    std::map<std::string, std::string> entries_;
    bool ok_ = false;
};

} // namespace gd_hnsw
