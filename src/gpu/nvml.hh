// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_NVML_H
#define FML_GPU_NVML_H
#pragma once


#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

// https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html#nvml-api-reference
#include <nvml.h>

#define CHECK_NVML(call) {nvmlReturn_t check = call; nvml::check_nvml_ret(check);}
#define NVML_MAX_STRLEN 128


/**
 * @brief NVIDIA Management Library (NVML) interface.
 * @except Each function can throw a 'runtime_error' exception.
 */
namespace nvml
{
  namespace
  {
    inline void check_nvml_ret(nvmlReturn_t check)
    {
      if (check != NVML_SUCCESS)
      {
        if (check == NVML_ERROR_UNINITIALIZED)
          throw std::runtime_error("NVML was not successfully initialized");  
        else if (check == NVML_ERROR_INVALID_ARGUMENT)
          throw std::runtime_error("invalid argument");
        else if (check == NVML_ERROR_NOT_SUPPORTED)
          throw std::runtime_error("device does not support requested feature");
        else if (check == NVML_ERROR_NO_PERMISSION)
          throw std::runtime_error("NVML does not have permission to talk to the driver");
        // else if (check == NVML_ERROR_ALREADY_INITIALIZED) // deprecated
        //   throw std::runtime_error("already initialized")
        else if (check == NVML_ERROR_NOT_FOUND)
          throw std::runtime_error("process not found");
        else if (check == NVML_ERROR_INSUFFICIENT_SIZE)
          throw std::runtime_error("internal string buffer too small");
        else if (check == NVML_ERROR_INSUFFICIENT_POWER)
          throw std::runtime_error("device has improperly attached external power cable");
        else if (check == NVML_ERROR_DRIVER_NOT_LOADED)
          throw std::runtime_error("NVIDIA driver is not running");
        else if (check == NVML_ERROR_TIMEOUT)
          throw std::runtime_error("provided timeout has passed");
        else if (check == NVML_ERROR_IRQ_ISSUE)
          throw std::runtime_error("NVIDIA kernel detected an interrupt issue with the attached GPUs");
        else if (check == NVML_ERROR_LIBRARY_NOT_FOUND)
          throw std::runtime_error("NVML shared library could not be loaded");
        else if (check == NVML_ERROR_FUNCTION_NOT_FOUND)
          throw std::runtime_error("local NVML version does not support requested function");
        else if (check == NVML_ERROR_CORRUPTED_INFOROM)
          throw std::runtime_error("infoROM is corrupted");
        else if (check == NVML_ERROR_GPU_IS_LOST)
          throw std::runtime_error("GPU is inaccessible");
        else if (check == NVML_ERROR_RESET_REQUIRED)
          throw std::runtime_error("GPU needs to be reset before it can be used again");
        else if (check == NVML_ERROR_OPERATING_SYSTEM)
          throw std::runtime_error("GPU control device was blocked by the OS");
        else if (check == NVML_ERROR_LIB_RM_VERSION_MISMATCH)
          throw std::runtime_error("driver/library version mismatch");
        else if (check == NVML_ERROR_IN_USE)
          throw std::runtime_error("GPU currently in use");
        else if (check == NVML_ERROR_MEMORY)
          throw std::runtime_error("insufficient memory");
        else if (check == NVML_ERROR_MEMORY)
          throw std::runtime_error("no data");
        else if (check == NVML_ERROR_VGPU_ECC_NOT_SUPPORTED)
          throw std::runtime_error("operation is not available because ECC is enabled");
        else if (check == NVML_ERROR_UNKNOWN)
          throw std::runtime_error("unknown NVML error");
        else
          throw std::runtime_error(nvmlErrorString(check));
      }
    }
  }
  
  
  
  /**
   * @brief Initialize NVML.
   * @details Call once before calling other NVML methods.
   */
  inline void init()
  {
    CHECK_NVML( nvmlInit() );
  }
  
  /**
   * @brief Shut down NVML.
   * @details Call when you are finished.
   */
  inline void shutdown()
  {
    CHECK_NVML( nvmlShutdown() );
  }
}



namespace nvml
{
  /**
   * @brief NVML queries against the system, independent of any GPU devices.
   * @details Only query methods are available.
   */
  namespace system
  {
    /**
     * @brief System CUDA driver version.
     * @return The version is encoded as (major*1000) + (minor*10).
     */
    inline int get_cuda_driver_version()
    {
      int ret;
      CHECK_NVML( nvmlSystemGetCudaDriverVersion(&ret) );
      return ret;
    }
    
    /**
     * @brief System graphics driver version.
     */
    inline std::string get_driver_version()
    {
      std::string ret;
      ret.resize(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
      CHECK_NVML( nvmlSystemGetDriverVersion(&ret[0], ret.max_size()) );
      return ret;
    }
    
    /**
     * @brief Version of the NVML library.
     */
    inline std::string get_nvml_version()
    {
      std::string ret;
      ret.resize(NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE);
      CHECK_NVML( nvmlSystemGetNVMLVersion(&ret[0], ret.max_size()) );
      return ret;
    }
    
    /**
     * @brief Process name.
     * @param[in] pid Process ID.
     * @return Process name.
     */
    inline std::string get_process_name(unsigned int pid)
    {
      std::string ret;
      ret.resize(NVML_MAX_STRLEN);
      CHECK_NVML( nvmlSystemGetProcessName(pid, &ret[0], ret.max_size()) );
      return ret;
    }
  }
}



namespace nvml
{
  /**
   * @brief NVML queries against GPU devices.
   * @details Only query methods are available.
  */
  namespace device
  {
    inline std::string get_board_part_number(nvmlDevice_t device)
    {
      std::string ret;
      ret.resize(NVML_MAX_STRLEN);
      CHECK_NVML( nvmlDeviceGetBoardPartNumber(device, &ret[0], ret.max_size()) );
      return ret;
    }
    
    inline std::string get_brand(nvmlDevice_t device)
    {
      nvmlBrandType_t type;
      CHECK_NVML( nvmlDeviceGetBrand(device, &type) );
      if (type == NVML_BRAND_UNKNOWN)
        return "unknown";
      else if (type == NVML_BRAND_QUADRO)
        return "quadro";
      else if (type == NVML_BRAND_TESLA)
        return "tesla";
      else if (type == NVML_BRAND_NVS)
        return "nvs";
      else if (type == NVML_BRAND_GRID)
        return "grid";
      else if (type == NVML_BRAND_GEFORCE)
        return "geforce";
    #ifdef NVML_BRAND_TITAN
      else if (type == NVML_BRAND_TITAN)
        return "titan";
    #endif
      else
        return "missing from list; contact nvsmi devs";
    }
    
    inline std::string get_compute_mode(nvmlDevice_t device)
    {
      nvmlComputeMode_t mode;
      CHECK_NVML( nvmlDeviceGetComputeMode(device, &mode) );
      if (mode == NVML_COMPUTEMODE_DEFAULT)
        return "Default";
      else if (mode == NVML_COMPUTEMODE_EXCLUSIVE_THREAD)
        return "E. Thread";
      else if (mode == NVML_COMPUTEMODE_PROHIBITED)
        return "Prohibited";
      else if (mode == NVML_COMPUTEMODE_EXCLUSIVE_PROCESS)
        return "E. Process";
      else
        return "";
    }
    
    inline int get_count()
    {
      unsigned int num_gpus;
      CHECK_NVML( nvmlDeviceGetCount(&num_gpus) );
      return (int) num_gpus;
    }
    
    inline void get_cuda_compute_capability(nvmlDevice_t device, int *major, int *minor)
    {
      CHECK_NVML( nvmlDeviceGetCudaComputeCapability(device, major, minor) );
    }
    
    inline int get_curr_pcie_link_generation(nvmlDevice_t device)
    {
      unsigned int currLinkGen;
      CHECK_NVML( nvmlDeviceGetCurrPcieLinkGeneration(device, &currLinkGen) );
      return (int) currLinkGen;
    }
    
    inline int get_curr_pcie_link_width(nvmlDevice_t device)
    {
      unsigned int currLinkWidth;
      CHECK_NVML( nvmlDeviceGetCurrPcieLinkWidth(device, &currLinkWidth) );
      return (int) currLinkWidth;
    }
    
    inline int get_display_active(nvmlDevice_t device)
    {
      nvmlEnableState_t disp;
      CHECK_NVML( nvmlDeviceGetDisplayActive(device, &disp) );
      return (int) disp;
    }
    
    inline int get_fan_speed(nvmlDevice_t device)
    {
      unsigned int speed;
      nvmlReturn_t check = nvmlDeviceGetFanSpeed(device, &speed);
      if (check == NVML_ERROR_NOT_SUPPORTED)
        return INT_MIN;
      else
        nvml::check_nvml_ret(check);
      
      return (int) speed;
    }
    
    inline nvmlDevice_t get_handle_by_index(int index)
    {
      nvmlDevice_t device;
      CHECK_NVML( nvmlDeviceGetHandleByIndex(index, &device) );
      return device;
    }
    
    inline int get_index(nvmlDevice_t device)
    {
      unsigned int index;
      CHECK_NVML( nvmlDeviceGetIndex(device, &index) );
      return (int) index;
    }
    
    inline void get_memory_info(nvmlDevice_t device, double *memory_used, double *memory_total)
    {
      nvmlMemory_t memory;
      CHECK_NVML( nvmlDeviceGetMemoryInfo(device, &memory) );
      *memory_used = (double) memory.used;
      *memory_total = (double) memory.total;
    }
    
    inline std::string get_name(nvmlDevice_t device)
    {
      std::string ret;
      ret.resize(NVML_MAX_STRLEN);
      CHECK_NVML( nvmlDeviceGetName(device, &ret[0], ret.max_size()) );
      return ret;
    }
    
    inline int get_performance_state(nvmlDevice_t device)
    {
      nvmlPstates_t pState;
      CHECK_NVML( nvmlDeviceGetPerformanceState(device, &pState) );
      return (int) pState;
    }
    
    inline int get_persistence_mode(nvmlDevice_t device)
    {
      nvmlEnableState_t mode;
      CHECK_NVML( nvmlDeviceGetPersistenceMode(device, &mode) );
      return (int) mode;
    }
    
    inline int get_power_max(nvmlDevice_t device)
    {
      unsigned int power_min, power_max;
      CHECK_NVML( nvmlDeviceGetPowerManagementLimitConstraints(device, &power_min, &power_max) );
      return (int) power_max;
    }
    
    inline int get_power_usage(nvmlDevice_t device)
    {
      unsigned int power;
      CHECK_NVML( nvmlDeviceGetPowerUsage(device, &power) );
      return (int) power;
    }
    
    inline std::string get_serial(nvmlDevice_t device)
    {
      std::string ret;
      ret.resize(NVML_MAX_STRLEN);
      CHECK_NVML( nvmlDeviceGetSerial(device, &ret[0], ret.max_size()) );
      return ret;
    }
    
    inline int get_temperature(nvmlDevice_t device)
    {
      nvmlTemperatureSensors_t sensor = NVML_TEMPERATURE_GPU;
      unsigned int temp;
      CHECK_NVML( nvmlDeviceGetTemperature(device, sensor, &temp) );
      return (int) temp;
    }
    
    inline int get_utilization(nvmlDevice_t device)
    {
      nvmlUtilization_t utilization;
      CHECK_NVML( nvmlDeviceGetUtilizationRates(device, &utilization) );
      return (int) utilization.gpu;
    }
    
    inline std::string get_uuid(nvmlDevice_t device)
    {
      std::string ret;
      ret.resize(NVML_MAX_STRLEN);
      CHECK_NVML( nvmlDeviceGetUUID(device, &ret[0], ret.max_size()) );
      return ret;
    }
  }
}



#undef NVML_MAX_STRLEN
#undef CHECK_NVML


#endif
