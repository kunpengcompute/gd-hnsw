/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "config_parser.h"

using namespace gd_hnsw;

static std::string write_temp(const std::string &content)
{
    std::string path = "test_cfg_" + std::to_string(std::rand()) + ".tmp";
    std::ofstream f(path);
    f << content;
    f.close();
    return path;
}

TEST(ConfigParser, SingleKeyValue)
{
    auto p = write_temp("M = 16\n");
    ConfigParser cfg(p);
    EXPECT_TRUE(cfg.ok());
    EXPECT_EQ(cfg.get_uint("M"), 16u);
    std::remove(p.c_str());
}

TEST(ConfigParser, MultipleKeyValues)
{
    auto p = write_temp("M = 16\ntopk = 10\nef_search = 64\n");
    ConfigParser cfg(p);
    EXPECT_TRUE(cfg.ok());
    EXPECT_EQ(cfg.get_uint("M"), 16u);
    EXPECT_EQ(cfg.get_int("topk"), 10);
    EXPECT_EQ(cfg.get_uint("ef_search"), 64u);
    std::remove(p.c_str());
}

TEST(ConfigParser, SkipCommentLines)
{
    auto p = write_temp("# comment\nM = 16\n# another\ntopk = 10\n");
    ConfigParser cfg(p);
    EXPECT_TRUE(cfg.ok());
    EXPECT_EQ(cfg.get_uint("M"), 16u);
    EXPECT_EQ(cfg.get_int("topk"), 10);
    std::remove(p.c_str());
}

TEST(ConfigParser, SkipBlankLines)
{
    auto p = write_temp("\nM = 16\n\n\n\ntopk = 10\n\n");
    ConfigParser cfg(p);
    EXPECT_TRUE(cfg.ok());
    EXPECT_EQ(cfg.get_uint("M"), 16u);
    std::remove(p.c_str());
}

TEST(ConfigParser, TrimWhitespace)
{
    auto p = write_temp("  M   =   32  \n");
    ConfigParser cfg(p);
    EXPECT_TRUE(cfg.has("M"));
    EXPECT_EQ(cfg.get_uint("M"), 32u);
    std::remove(p.c_str());
}

TEST(ConfigParser, ValueWithSpaces)
{
    auto p = write_temp("desc = hello world\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_str("desc"), "hello world");
    std::remove(p.c_str());
}

TEST(ConfigParser, GetIntNegative)
{
    auto p = write_temp("k = -5\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_int("k"), -5);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetIntMissingDefault)
{
    auto p = write_temp("M = 16\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_int("nonexistent", 42), 42);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetUint)
{
    auto p = write_temp("M = 32\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_uint("M"), 32u);
    EXPECT_EQ(cfg.get_uint("N", 99), 99u);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetDouble)
{
    auto p = write_temp("rate = 3.14\n");
    ConfigParser cfg(p);
    EXPECT_NEAR(cfg.get_double("rate"), 3.14, 1e-9);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetBoolTrue)
{
    {
        auto p = write_temp("a = true\n");
        ConfigParser cfg(p);
        EXPECT_TRUE(cfg.get_bool("a"));
        std::remove(p.c_str());
    }
    {
        auto p = write_temp("a = 1\n");
        ConfigParser cfg(p);
        EXPECT_TRUE(cfg.get_bool("a"));
        std::remove(p.c_str());
    }
    {
        auto p = write_temp("a = yes\n");
        ConfigParser cfg(p);
        EXPECT_TRUE(cfg.get_bool("a"));
        std::remove(p.c_str());
    }
}

TEST(ConfigParser, GetBoolFalse)
{
    {
        auto p = write_temp("a = false\n");
        ConfigParser cfg(p);
        EXPECT_FALSE(cfg.get_bool("a"));
        std::remove(p.c_str());
    }
    {
        auto p = write_temp("a = 0\n");
        ConfigParser cfg(p);
        EXPECT_FALSE(cfg.get_bool("a"));
        std::remove(p.c_str());
    }
    {
        auto p = write_temp("a = no\n");
        ConfigParser cfg(p);
        EXPECT_FALSE(cfg.get_bool("a"));
        std::remove(p.c_str());
    }
}

TEST(ConfigParser, GetBoolDefault)
{
    auto p = write_temp("M = 16\n");
    ConfigParser cfg(p);
    EXPECT_TRUE(cfg.get_bool("profile", true));
    EXPECT_FALSE(cfg.get_bool("profile", false));
    std::remove(p.c_str());
}

TEST(ConfigParser, GetStr)
{
    auto p = write_temp("path = /data/sift1m.h5\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_str("path"), "/data/sift1m.h5");
    EXPECT_EQ(cfg.get_str("missing", "/dflt"), "/dflt");
    std::remove(p.c_str());
}

TEST(ConfigParser, HasKey)
{
    auto p = write_temp("M = 16\n");
    ConfigParser cfg(p);
    EXPECT_TRUE(cfg.has("M"));
    EXPECT_FALSE(cfg.has("N"));
    std::remove(p.c_str());
}

TEST(ConfigParser, OkFileNotFound)
{
    ConfigParser cfg("/nonexistent/path/config.file");
    EXPECT_FALSE(cfg.ok());
}

TEST(ConfigParser, OkEmptyFile)
{
    auto p = write_temp("");
    ConfigParser cfg(p);
    EXPECT_FALSE(cfg.ok());
    std::remove(p.c_str());
}

TEST(ConfigParser, OkValidFile)
{
    auto p = write_temp("M = 16\n");
    ConfigParser cfg(p);
    EXPECT_TRUE(cfg.ok());
    std::remove(p.c_str());
}

TEST(ConfigParser, SplitOnFirstEquals)
{
    auto p = write_temp("key = val=ue\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_str("key"), "val=ue");
    std::remove(p.c_str());
}

TEST(ConfigParser, EmptyValue)
{
    auto p = write_temp("key =\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_str("key"), "");
    std::remove(p.c_str());
}

TEST(ConfigParser, LineWithoutEquals)
{
    auto p = write_temp("no_equals\nM = 16\n");
    ConfigParser cfg(p);
    EXPECT_TRUE(cfg.has("M"));
    EXPECT_FALSE(cfg.has("no_equals"));
    std::remove(p.c_str());
}

TEST(ConfigParser, HashInValue)
{
    auto p = write_temp("tag = foo#bar\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_str("tag"), "foo#bar");
    std::remove(p.c_str());
}

TEST(ConfigParser, CRLF)
{
    auto p = write_temp("M = 16\r\ntopk = 10\r\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_uint("M"), 16u);
    EXPECT_EQ(cfg.get_int("topk"), 10);
    std::remove(p.c_str());
}

// ---- hardened numeric getters (throw std::runtime_error on bad input) ----

TEST(ConfigParser, GetUintNegativeThrows)
{
    // stoul("-1") == ULONG_MAX used to bypass >= 1 range checks downstream.
    auto p = write_temp("ef-search = -1\n");
    ConfigParser cfg(p);
    EXPECT_THROW(cfg.get_uint("ef-search"), std::runtime_error);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetUintOutOfRangeThrows)
{
    auto p = write_temp("ef-search = 4294967296\n");
    ConfigParser cfg(p);
    EXPECT_THROW(cfg.get_uint("ef-search"), std::runtime_error);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetUintGarbageThrows)
{
    auto p = write_temp("ef-search = abc\n");
    ConfigParser cfg(p);
    EXPECT_THROW(cfg.get_uint("ef-search"), std::runtime_error);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetUintTrailingGarbageThrows)
{
    // "64.5" must not silently truncate to 64.
    auto p = write_temp("ef-search = 64.5\n");
    ConfigParser cfg(p);
    EXPECT_THROW(cfg.get_uint("ef-search"), std::runtime_error);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetIntGarbageThrows)
{
    auto p = write_temp("k = x10\n");
    ConfigParser cfg(p);
    EXPECT_THROW(cfg.get_int("k"), std::runtime_error);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetDoubleGarbageThrows)
{
    auto p = write_temp("rate = fast\n");
    ConfigParser cfg(p);
    EXPECT_THROW(cfg.get_double("rate"), std::runtime_error);
    std::remove(p.c_str());
}

TEST(ConfigParser, NumericValueWithInlineComment)
{
    // Real config files use "key = 8 # note" (see configs/*.config).
    auto p = write_temp("build.shards = 8            # must equal mpirun -np\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_int("build.shards"), 8);
    EXPECT_EQ(cfg.get_uint("build.shards"), 8u);
    std::remove(p.c_str());
}

TEST(ConfigParser, NumericValueTrailingWhitespace)
{
    auto p = write_temp("M = 16   \n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_uint("M"), 16u);
    std::remove(p.c_str());
}

TEST(ConfigParser, GetUintAcceptsUint32Max)
{
    auto p = write_temp("M = 4294967295\n");
    ConfigParser cfg(p);
    EXPECT_EQ(cfg.get_uint("M"), 4294967295u);
    std::remove(p.c_str());
}
