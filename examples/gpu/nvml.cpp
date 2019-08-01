#include <gpu/nvml.hh>
#include <iostream>


int main()
{
  nvml::init();
  
  std::string ver = nvml::system::get_driver_version();
  std::cout << "Cuda driver: " << ver << std::endl;
  
  int ngpus = nvml::device::get_count();
  
  for (int gpu=0; gpu<ngpus; gpu++)
  {
    std::cout << "gpu " << gpu << " of " << ngpus << std::endl;
    
    nvmlDevice_t device = nvml::device::get_handle_by_index(gpu);
    
    int disp = nvml::device::get_display_active(device);
    std::cout << "    Display: " << disp << std::endl;
    
    double used, total;
    nvml::device::get_memory_info(device, &used, &total);
    std::cout << "    Mem used (MB): " << used/1024/1024 << std::endl;
    std::cout << "    Mem total (MB): " << total/1024/1024 << std::endl;
  }
  
  nvml::shutdown();
  
  return 0;
}
