#ifndef FML_NVML_H
#define FML_NVML_H


#include <climits>
#include <cmath>
#include <string>

// https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html#nvml-api-reference
#include <nvml.h>

#define CHECK_NVML(call) {nvmlReturn_t check = call; nvml::check_nvml_ret(check);}


namespace nvml
{
  namespace
  {
    void check_nvml_ret(nvmlReturn_t check)
    {
      if (check != NVML_SUCCESS)
      {
        // TODO
      }
    }
  }
  
  
  
  void init()
  {
    CHECK_NVML( nvmlInit() );
  }
  
  void shutdown()
  {
    CHECK_NVML( nvmlShutdown() );
  }
}



namespace nvml
{
  namespace system
  {
    #define MAX_STRLEN 80
    
    
    int get_cuda_driver_version()
    {
      int ret;
      CHECK_NVML( nvmlSystemGetCudaDriverVersion(&ret) );
      return ret;
    }
    
    std::string get_driver_version()
    {
      std::string ret;
      ret.resize(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
      CHECK_NVML( nvmlSystemGetDriverVersion(&ret[0], ret.max_size()) );
      return ret;
    }
    
    std::string get_nvml_version()
    {
      std::string ret;
      ret.resize(NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE);
      CHECK_NVML( nvmlSystemGetNVMLVersion(&ret[0], ret.max_size()) );
      return ret;
    }
    
    std::string get_process_name(unsigned int pid)
    {
      std::string ret;
      ret.resize(MAX_STRLEN);
      CHECK_NVML( nvmlSystemGetProcessName(pid, &ret[0], ret.max_size()) );
      return ret;
    }
    
    
    #undef MAX_STRLEN
  }
}



namespace nvml
{
  namespace device
  {
    #define MAX_STRLEN 80
    
    
    
    std::string get_board_part_number(nvmlDevice_t device)
    {
      std::string ret;
      ret.resize(MAX_STRLEN);
      CHECK_NVML( nvmlDeviceGetBoardPartNumber(device, &ret[0], ret.max_size()) );
      return ret;
    }
    
    std::string get_brand(nvmlDevice_t device)
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
    
    std::string get_compute_mode(nvmlDevice_t device)
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
    
    int get_count()
    {
      unsigned int num_gpus;
      CHECK_NVML( nvmlDeviceGetCount(&num_gpus) );
      return (int) num_gpus;
    }
    
    void get_cuda_compute_capability(nvmlDevice_t device, int *major, int *minor)
    {
      CHECK_NVML( nvmlDeviceGetCudaComputeCapability(device, major, minor) );
    }
    
    int get_curr_pcie_link_generation(nvmlDevice_t device)
    {
      unsigned int currLinkGen;
      CHECK_NVML( nvmlDeviceGetCurrPcieLinkGeneration(device, &currLinkGen) );
      return (int) currLinkGen;
    }
    
    int get_curr_pcie_link_width(nvmlDevice_t device)
    {
      unsigned int currLinkWidth;
      CHECK_NVML( nvmlDeviceGetCurrPcieLinkWidth(device, &currLinkWidth) );
      return (int) currLinkWidth;
    }
    
    int get_display_active(nvmlDevice_t device)
    {
      nvmlEnableState_t disp;
      CHECK_NVML( nvmlDeviceGetDisplayActive(device, &disp) );
      return (int) disp;
    }
    
    int get_fan_speed(nvmlDevice_t device)
    {
      unsigned int speed;
      nvmlReturn_t check = nvmlDeviceGetFanSpeed(device, &speed);
      if (check == NVML_ERROR_NOT_SUPPORTED)
        return INT_MIN;
      else
        nvml::check_nvml_ret(check);
      
      return (int) speed;
    }
    
    nvmlDevice_t get_handle_by_index(int index)
    {
      nvmlDevice_t device;
      CHECK_NVML( nvmlDeviceGetHandleByIndex(index, &device) );
      return device;
    }
    
    int get_index(nvmlDevice_t device)
    {
      unsigned int index;
      CHECK_NVML( nvmlDeviceGetIndex(device, &index) );
      return (int) index;
    }
    
    void get_memory_info(nvmlDevice_t device, double *memory_used, double *memory_total)
    {
      nvmlMemory_t memory;
      CHECK_NVML( nvmlDeviceGetMemoryInfo(device, &memory) );
      *memory_used = (double) memory.used;
      *memory_total = (double) memory.total;
    }
    
    std::string get_name(nvmlDevice_t device)
    {
      std::string ret;
      ret.resize(MAX_STRLEN);
      CHECK_NVML( nvmlDeviceGetName(device, &ret[0], ret.max_size()) );
      return ret;
    }
    
    int get_performance_state(nvmlDevice_t device)
    {
      nvmlPstates_t pState;
      CHECK_NVML( nvmlDeviceGetPerformanceState(device, &pState) );
      return (int) pState;
    }
    
    int get_persistence_mode(nvmlDevice_t device)
    {
      nvmlEnableState_t mode;
      CHECK_NVML( nvmlDeviceGetPersistenceMode(device, &mode) );
      return (int) mode;
    }
    
    int get_power_max(nvmlDevice_t device)
    {
      unsigned int power_min, power_max;
      CHECK_NVML( nvmlDeviceGetPowerManagementLimitConstraints(device, &power_min, &power_max) );
      return (int) power_max;
    }
    
    int get_power_usage(nvmlDevice_t device)
    {
      unsigned int power;
      CHECK_NVML( nvmlDeviceGetPowerUsage(device, &power) );
      return (int) power;
    }
    
    std::string get_serial(nvmlDevice_t device)
    {
      std::string ret;
      ret.resize(MAX_STRLEN);
      CHECK_NVML( nvmlDeviceGetSerial(device, &ret[0], ret.max_size()) );
      return ret;
    }
    
    int get_temperature(nvmlDevice_t device)
    {
      nvmlTemperatureSensors_t sensor = NVML_TEMPERATURE_GPU;
      unsigned int temp;
      CHECK_NVML( nvmlDeviceGetTemperature(device, sensor, &temp) );
      return (int) temp;
    }
    
    int get_utilization(nvmlDevice_t device)
    {
      nvmlUtilization_t utilization;
      CHECK_NVML( nvmlDeviceGetUtilizationRates(device, &utilization) );
      return (int) utilization.gpu;
    }
    
    std::string get_uuid(nvmlDevice_t device)
    {
      std::string ret;
      ret.resize(MAX_STRLEN);
      CHECK_NVML( nvmlDeviceGetUUID(device, &ret[0], ret.max_size()) );
      return ret;
    }
    
    
    #undef MAX_STRLEN
  }
}



#undef CHECK_NVML


#endif
