/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
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

    // Numeric getters throw std::runtime_error (naming key and value) on
    // garbage input; callers are expected to catch and fail fast.
    int get_int(const std::string &key, int def = 0) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return def;
        const int64_t v = parse_i64(key, it->second);
        if (v < INT32_MIN || v > INT32_MAX)
            fail(key, it->second, "must be an integer in [-2147483648, 2147483647]");
        return static_cast<int>(v);
    }

    uint32_t get_uint(const std::string &key, uint32_t def = 0) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return def;
        const int64_t v = parse_i64(key, it->second);
        // stoul would accept "-1" as ULONG_MAX, silently bypassing >= 1
        // range checks at call sites — reject negatives explicitly here.
        if (v < 0 || v > UINT32_MAX)
            fail(key, it->second, "must be an integer in [0, 4294967295]");
        return static_cast<uint32_t>(v);
    }

    double get_double(const std::string &key, double def = 0.0) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return def;
        const std::string &s = it->second;
        size_t pos = 0;
        double v = 0.0;
        try {
            v = std::stod(s, &pos);
        } catch (const std::exception &) {
            fail(key, s, "must be a number");
        }
        check_trailing(key, s, pos);
        return v;
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
    static int64_t parse_i64(const std::string &key, const std::string &s)
    {
        size_t pos = 0;
        long long v = 0;
        try {
            v = std::stoll(s, &pos);
        } catch (const std::invalid_argument &) {
            fail(key, s, "must be an integer");
        } catch (const std::out_of_range &) {
            fail(key, s, "exceeds int64 range");
        }
        check_trailing(key, s, pos);
        return v;
    }

    // A number may be followed by whitespace and a '#' comment ("8 # note")
    // but not by other tokens ("8.5", "8x") — a silent prefix match would
    // hide config typos.
    static void check_trailing(const std::string &key, const std::string &s, size_t pos)
    {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t'))
            pos++;
        if (pos < s.size() && s[pos] != '#')
            fail(key, s, "has trailing characters after the number");
    }

    [[noreturn]] static void fail(const std::string &key, const std::string &s, const char *why)
    {
        throw std::runtime_error("config '" + key + "': '" + s + "' " + why);
    }

    std::map<std::string, std::string> entries_;
    bool ok_ = false;
};

} // namespace gd_hnsw
