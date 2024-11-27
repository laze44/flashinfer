cmake_minimum_required(VERSION 3.23.1)
project(flashinfer CUDA CXX)

include(cmake/utils/Utils.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
  message(FATAL_ERROR "Python3 not found.")
endif()

# NOTE: do not modify this file to change option values. You can create a
# config.cmake at build folder and add set(OPTION VALUE) to override these build
# options. Alernatively, use cmake -DOPTION=VALUE through command-line.
flashinfer_option(FLASHINFER_ENABLE_FP8
                  "Whether to compile fp8 kernels or not." ON)
flashinfer_option(FLASHINFER_ENABLE_BF16
                  "Whether to compile bf16 kernels or not." ON)
flashinfer_option(
  FLASHINFER_PREFILL
  "Whether to compile prefill kernel tests/benchmarks or not." OFF)
flashinfer_option(
  FLASHINFER_DECODE "Whether to compile decode kernel tests/benchmarks or not."
  OFF)
flashinfer_option(FLASHINFER_PAGE
                  "Whether to compile page kernel tests/benchmarks or not." OFF)
flashinfer_option(
  FLASHINFER_CASCADE
  "Whether to compile cascade kernel tests/benchmarks or not." OFF)
flashinfer_option(
  FLASHINFER_SAMPLING
  "Whether to compile sampling kernel tests/benchmarks or not." OFF)
flashinfer_option(
  FLASHINFER_NORM
  "Whether to compile normalization kernel tests/benchmarks or not." OFF)
flashinfer_option(
  FLASHINFER_DISTRIBUTED
  "Whether to compile distributed kernel tests/benchmarks or not." OFF)
flashinfer_option(FLASHINFER_FASTDIV_TEST
                  "Whether to compile fastdiv kernel tests or not." OFF)
flashinfer_option(FLASHINFER_FASTDEQAUNT_TEST
                  "Whether to compile fast dequant kernel tests or not." OFF)
flashinfer_option(FLASHINFER_TVM_BINDING
                  "Whether to compile tvm binding or not." OFF)
flashinfer_option(FLASHINFER_TVM_SOURCE_DIR
                  "The path to tvm for building tvm binding." "")

# The following configurations can impact the binary size of the generated
# library
flashinfer_option(FLASHINFER_GEN_HEAD_DIMS "Head dims to enable" 64 128 256)
flashinfer_option(FLASHINFER_GEN_POS_ENCODING_MODES "Pos encodings to enable" 0
                  1 2)
flashinfer_option(FLASHINFER_GEN_ALLOW_FP16_QK_REDUCTIONS
                  "QK reductions to enable" "false" "true")
flashinfer_option(FLASHINFER_GEN_MASK_MODES "Mask modes to enable" 0 1 2)

if(DEFINED FLASHINFER_CUDA_ARCHITECTURES)
  message(
    STATUS "CMAKE_CUDA_ARCHITECTURES set to ${FLASHINFER_CUDA_ARCHITECTURES}.")
  set(CMAKE_CUDA_ARCHITECTURES ${FLASHINFER_CUDA_ARCHITECTURES})
else(DEFINED FLASHINFER_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES is ${CMAKE_CUDA_ARCHITECTURES}")
endif(DEFINED FLASHINFER_CUDA_ARCHITECTURES)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
if(FLASHINFER_PREFILL
   OR FLASHINFER_DECODE
   OR FLASHINFER_PAGE
   OR FLASHINFER_CASCADE
   OR FLASHINFER_SAMPLING
   OR FLASHINFER_NORM)
  message(STATUS "NVBench and GoogleTest enabled")
  if(FLASHINFER_DISTRIBUTED)
    add_subdirectory(3rdparty/mscclpp)
  endif(FLASHINFER_DISTRIBUTED)
endif(
  FLASHINFER_PREFILL
  OR FLASHINFER_DECODE
  OR FLASHINFER_PAGE
  OR FLASHINFER_CASCADE
  OR FLASHINFER_SAMPLING
  OR FLASHINFER_NORM)
find_package(Thrust REQUIRED)

set(FLASHINFER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(FLASHINFER_ENABLE_FP8)
  message(STATUS "Compile fp8 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_FP8)
endif(FLASHINFER_ENABLE_FP8)

if(FLASHINFER_ENABLE_BF16)
  message(STATUS "Compile bf16 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_BF16)
endif(FLASHINFER_ENABLE_BF16)

# generate kernel inst
set(HEAD_DIMS ${FLASHINFER_GEN_HEAD_DIMS})
set(POS_ENCODING_MODES ${FLASHINFER_GEN_POS_ENCODING_MODES})
set(ALLOW_FP16_QK_REDUCTIONS ${FLASHINFER_GEN_ALLOW_FP16_QK_REDUCTIONS})
set(MASK_MODES ${FLASHINFER_GEN_MASK_MODES})
set(DECODE_DTYPES "f16")
set(PREFILL_DTYPES "f16")
set(DECODE_FP8_DTYPES)
set(IDTYPES "i32")

if(FLASHINFER_ENABLE_FP8)
  list(APPEND DECODE_DTYPES "e4m3" "e5m2")
  list(APPEND DECODE_FP8_DTYPES "e4m3" "e5m2")
  list(APPEND PREFILL_FP8_DTYPES "e4m3" "e5m2")
endif(FLASHINFER_ENABLE_FP8)

if(FLASHINFER_ENABLE_BF16)
  list(APPEND DECODE_DTYPES "bf16")
  list(APPEND PREFILL_DTYPES "bf16")
endif(FLASHINFER_ENABLE_BF16)

# log options
message(STATUS "FLASHINFER_HEAD_DIMS=${HEAD_DIMS}")
message(STATUS "FLASHINFER_POS_ENCODING_MODES=${POS_ENCODING_MODES}")
message(
  STATUS "FLASHINFER_ALLOW_FP16_QK_REDUCTIONS=${ALLOW_FP16_QK_REDUCTIONS}")
message(STATUS "FLASHINFER_MASK_MODES=${MASK_MODES}")

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src/generated)

# single decode kernel inst generation
foreach(head_dim IN LISTS HEAD_DIMS)
  foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
    foreach(dtype IN LISTS DECODE_DTYPES)
      set(generated_kernel_src
          ${PROJECT_SOURCE_DIR}/src/generated/single_decode_head_${head_dim}_posenc_${pos_encoding_mode}_dtypeq_${dtype}_dtypekv_${dtype}_dtypeout_${dtype}.cu
      )
      if(NOT EXISTS ${generated_kernel_src})
        add_custom_command(
          OUTPUT ${generated_kernel_src}
          COMMAND
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_single_decode_inst.py
            ${generated_kernel_src}
          DEPENDS
            ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_single_decode_inst.py
          COMMENT "Generating additional source file ${generated_kernel_src}"
          VERBATIM)
      endif()
      list(APPEND single_decode_kernels_src ${generated_kernel_src})
    endforeach(dtype)

    # fp8 kv-cache
    foreach(dtype_kv IN LISTS DECODE_FP8_DTYPES)
      set(generated_kernel_src
          ${PROJECT_SOURCE_DIR}/src/generated/single_decode_head_${head_dim}_posenc_${pos_encoding_mode}_dtypeq_f16_dtypekv_${dtype_kv}_dtypeout_f16.cu
      )
      if(NOT EXISTS ${generated_kernel_src})
        add_custom_command(
          OUTPUT ${generated_kernel_src}
          COMMAND
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_single_decode_inst.py
            ${generated_kernel_src}
          DEPENDS
            ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_single_decode_inst.py
          COMMENT "Generating additional source file ${generated_kernel_src}"
          VERBATIM)
      endif()
      list(APPEND single_decode_kernels_src ${generated_kernel_src})
    endforeach(dtype_kv)
  endforeach(pos_encoding_mode)
endforeach(head_dim)

# batch decode kernel inst generation
foreach(head_dim IN LISTS HEAD_DIMS)
  foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
    # paged kv-cache
    foreach(idtype IN LISTS IDTYPES)
      foreach(dtype IN LISTS DECODE_DTYPES)
        set(generated_kernel_src
            ${PROJECT_SOURCE_DIR}/src/generated/batch_paged_decode_head_${head_dim}_posenc_${pos_encoding_mode}_dtypeq_${dtype}_dtypekv_${dtype}_dtypeout_${dtype}_idtype_${idtype}.cu
        )
        if(NOT EXISTS ${generated_kernel_src})
          add_custom_command(
            OUTPUT ${generated_kernel_src}
            COMMAND
              ${Python3_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_paged_decode_inst.py
              ${generated_kernel_src}
            DEPENDS
              ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_paged_decode_inst.py
            COMMENT "Generating additional source file ${generated_kernel_src}"
            VERBATIM)
        endif()
        list(APPEND batch_decode_kernels_src ${generated_kernel_src})
      endforeach(dtype)

      # fp8 kv-cache
      foreach(dtype_kv IN LISTS DECODE_FP8_DTYPES)
        set(generated_kernel_src
            ${PROJECT_SOURCE_DIR}/src/generated/batch_paged_decode_head_${head_dim}_posenc_${pos_encoding_mode}_dtypeq_f16_dtypekv_${dtype_kv}_dtypeout_f16_idtype_${idtype}.cu
        )
        if(NOT EXISTS ${generated_kernel_src})
          add_custom_command(
            OUTPUT ${generated_kernel_src}
            COMMAND
              ${Python3_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_paged_decode_inst.py
              ${generated_kernel_src}
            DEPENDS
              ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_paged_decode_inst.py
            COMMENT "Generating additional source file ${generated_kernel_src}"
            VERBATIM)
        endif()
        list(APPEND batch_decode_kernels_src ${generated_kernel_src})
      endforeach(dtype_kv)
    endforeach(idtype)
  endforeach(pos_encoding_mode)
endforeach(head_dim)

add_library(decode_kernels STATIC ${single_decode_kernels_src}
                                  ${batch_decode_kernels_src})
target_include_directories(decode_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_compile_options(decode_kernels PRIVATE -Xcompiler=-fPIC --fatbin-options
                                              -compress-all)
# single prefill kernel inst generation
foreach(head_dim IN LISTS HEAD_DIMS)
  foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
    foreach(allow_fp16_qk_reduction IN LISTS ALLOW_FP16_QK_REDUCTIONS)
      foreach(mask_mode IN LISTS MASK_MODES)
        foreach(dtype IN LISTS PREFILL_DTYPES)
          set(generated_kernel_src
              ${PROJECT_SOURCE_DIR}/src/generated/single_prefill_head_${head_dim}_posenc_${pos_encoding_mode}_fp16qkred_${allow_fp16_qk_reduction}_mask_${mask_mode}_dtypeq_${dtype}_dtypekv_${dtype}_dtypeout_${dtype}.cu
          )
          if(NOT EXISTS ${generated_kernel_src})
            add_custom_command(
              OUTPUT ${generated_kernel_src}
              COMMAND
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_single_prefill_inst.py
                ${generated_kernel_src}
              DEPENDS
                ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_single_prefill_inst.py
              COMMENT "Generating additional source file ${generated_kernel_src}"
              VERBATIM)
          endif()
          list(APPEND single_prefill_kernels_src ${generated_kernel_src})
        endforeach(dtype)

        foreach(dtype_kv IN LISTS PREFILL_FP8_DTYPES)
          set(generated_kernel_src
              ${PROJECT_SOURCE_DIR}/src/generated/single_prefill_head_${head_dim}_posenc_${pos_encoding_mode}_fp16qkred_${allow_fp16_qk_reduction}_mask_${mask_mode}_dtypeq_f16_dtypekv_${dtype_kv}_dtypeout_f16.cu
          )
          if(NOT EXISTS ${generated_kernel_src})
            add_custom_command(
              OUTPUT ${generated_kernel_src}
              COMMAND
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_single_prefill_inst.py
                ${generated_kernel_src}
              DEPENDS
                ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_single_prefill_inst.py
              COMMENT "Generating additional source file ${generated_kernel_src}"
              VERBATIM)
          endif()
          list(APPEND single_prefill_kernels_src ${generated_kernel_src})
        endforeach(dtype_kv)
      endforeach(mask_mode)
    endforeach(allow_fp16_qk_reduction)
  endforeach(pos_encoding_mode)
endforeach(head_dim)

# batch paged prefill kernel inst generation
foreach(head_dim IN LISTS HEAD_DIMS)
  foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
    foreach(allow_fp16_qk_reduction IN LISTS ALLOW_FP16_QK_REDUCTIONS)
      foreach(mask_mode IN LISTS MASK_MODES)
        foreach(idtype IN LISTS IDTYPES)
          foreach(dtype IN LISTS PREFILL_DTYPES)
            set(generated_kernel_src
                ${PROJECT_SOURCE_DIR}/src/generated/batch_paged_prefill_head_${head_dim}_posenc_${pos_encoding_mode}_fp16qkred_${allow_fp16_qk_reduction}_mask_${mask_mode}_dtypeq_${dtype}_dtypekv_${dtype}_dtypeout_${dtype}_idtype_${idtype}.cu
            )
            if(NOT EXISTS ${generated_kernel_src})
              add_custom_command(
                OUTPUT ${generated_kernel_src}
                COMMAND
                  ${Python3_EXECUTABLE}
                  ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_paged_prefill_inst.py
                  ${generated_kernel_src}
                DEPENDS
                  ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_paged_prefill_inst.py
                COMMENT
                  "Generating additional source file ${generated_kernel_src}"
                VERBATIM)
            endif()
            list(APPEND batch_paged_prefill_kernels_src ${generated_kernel_src})
          endforeach(dtype)

          foreach(dtype_kv IN LISTS PREFILL_FP8_DTYPES)
            set(generated_kernel_src
                ${PROJECT_SOURCE_DIR}/src/generated/batch_paged_prefill_head_${head_dim}_posenc_${pos_encoding_mode}_fp16qkred_${allow_fp16_qk_reduction}_mask_${mask_mode}_dtypeq_f16_dtypekv_${dtype_kv}_dtypeout_f16_idtype_${idtype}.cu
            )
            if(NOT EXISTS ${generated_kernel_src})
              add_custom_command(
                OUTPUT ${generated_kernel_src}
                COMMAND
                  ${Python3_EXECUTABLE}
                  ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_paged_prefill_inst.py
                  ${generated_kernel_src}
                DEPENDS
                  ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_paged_prefill_inst.py
                COMMENT
                  "Generating additional source file ${generated_kernel_src}"
                VERBATIM)
            endif()
            list(APPEND batch_paged_prefill_kernels_src ${generated_kernel_src})
          endforeach(dtype_kv)
        endforeach(idtype)
      endforeach(mask_mode)
    endforeach(allow_fp16_qk_reduction)
  endforeach(pos_encoding_mode)
endforeach(head_dim)

# batch ragged prefill kernel inst generation
foreach(head_dim IN LISTS HEAD_DIMS)
  foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
    foreach(allow_fp16_qk_reduction IN LISTS ALLOW_FP16_QK_REDUCTIONS)
      foreach(mask_mode IN LISTS MASK_MODES)
        foreach(idtype IN LISTS IDTYPES)
          foreach(dtype IN LISTS PREFILL_DTYPES)
            set(generated_kernel_src
                ${PROJECT_SOURCE_DIR}/src/generated/batch_ragged_prefill_head_${head_dim}_posenc_${pos_encoding_mode}_fp16qkred_${allow_fp16_qk_reduction}_mask_${mask_mode}_dtypeq_${dtype}_dtypekv_${dtype}_dtypeout_${dtype}_idtype_${idtype}.cu
            )
            if(NOT EXISTS ${generated_kernel_src})
              add_custom_command(
                OUTPUT ${generated_kernel_src}
                COMMAND
                  ${Python3_EXECUTABLE}
                  ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_ragged_prefill_inst.py
                  ${generated_kernel_src}
                DEPENDS
                  ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_ragged_prefill_inst.py
                COMMENT
                  "Generating additional source file ${generated_kernel_src}"
                VERBATIM)
            endif()
            list(APPEND batch_ragged_prefill_kernels_src
                 ${generated_kernel_src})
          endforeach(dtype)

          foreach(dtype_kv IN LISTS PREFILL_FP8_DTYPES)
            set(generated_kernel_src
                ${PROJECT_SOURCE_DIR}/src/generated/batch_ragged_prefill_head_${head_dim}_posenc_${pos_encoding_mode}_fp16qkred_${allow_fp16_qk_reduction}_mask_${mask_mode}_dtypeq_f16_dtypekv_${dtype_kv}_dtypeout_f16_idtype_${idtype}.cu
            )
            if(NOT EXISTS ${generated_kernel_src})
              add_custom_command(
                OUTPUT ${generated_kernel_src}
                COMMAND
                  ${Python3_EXECUTABLE}
                  ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_ragged_prefill_inst.py
                  ${generated_kernel_src}
                DEPENDS
                  ${PROJECT_SOURCE_DIR}/python/_aot_build_utils/generate_batch_ragged_prefill_inst.py
                COMMENT
                  "Generating additional source file ${generated_kernel_src}"
                VERBATIM)
            endif()
            list(APPEND batch_ragged_prefill_kernels_src
                 ${generated_kernel_src})
          endforeach(dtype_kv)
        endforeach(idtype)
      endforeach(mask_mode)
    endforeach(allow_fp16_qk_reduction)
  endforeach(pos_encoding_mode)
endforeach(head_dim)

message(STATUS "FINISH GENERATE FILE")
add_library(
  prefill_kernels STATIC
  ${single_prefill_kernels_src} ${batch_paged_prefill_kernels_src}
  ${batch_ragged_prefill_kernels_src})
target_include_directories(prefill_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_compile_options(prefill_kernels PRIVATE -Xcompiler=-fPIC
                                               --fatbin-options -compress-all)