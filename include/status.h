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

// status.h — public Status codes shared by the build layer and the deploy layer.
//
// Extracted from gd_hnsw_api.h so that build.h can reference Status without
// pulling in the full Context declaration (which would create a circular
// include: build.h -> gd_hnsw_api.h -> build.h). Both build.h and
// gd_hnsw_api.h include this header.

#pragma once

namespace gd_hnsw {

// ============================================================
// Status codes (returned by Context methods; also in *Result structs)
// ============================================================

enum class Status : int {
    Ok = 0,
    InvalidArg = -1,
    MpiError = -2,
    UbsemError = -3,
    IoError = -4,
    ValidationErr = -5,
    NotInitialized = -6,
    AlreadyInit = -7,
};

} // namespace gd_hnsw
