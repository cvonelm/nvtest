/*
 *
 * Copyright (C) 2021 Maxime Schmitt <maxime.schmitt91@gmail.com>
 *
 * This file is part of Nvtop.
 *
 * Nvtop is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Nvtop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with nvtop.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <cerrno>
#include <cstdbool>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

extern "C"
{
#include <dlfcn.h>
}

#define NVML_SUCCESS 0
#define NVML_ERROR_INSUFFICIENT_SIZE 7

typedef struct nvmlDevice *nvmlDevice_t;
typedef int nvmlReturn_t; // store the enum as int

// Init and shutdown

static nvmlReturn_t (*nvmlInit)(void);

static nvmlReturn_t (*nvmlShutdown)(void);

// Static information and helper functions

static nvmlReturn_t (*nvmlDeviceGetCount)(unsigned int *deviceCount);

static nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(unsigned int index, nvmlDevice_t *device);

static const char *(*nvmlErrorString)(nvmlReturn_t);

static nvmlReturn_t (*nvmlDeviceGetName)(nvmlDevice_t device, char *name, unsigned int length);

typedef struct {
  char busIdLegacy[16];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pciDeviceId;
  // Added in NVML 2.285 API
  unsigned int pciSubSystemId;
  char busId[32];
} nvmlPciInfo_t;

static nvmlReturn_t (*nvmlDeviceGetPciInfo)(nvmlDevice_t device, nvmlPciInfo_t *pciInfo);

static nvmlReturn_t (*nvmlDeviceGetMaxPcieLinkGeneration)(nvmlDevice_t device, unsigned int *maxLinkGen);

static nvmlReturn_t (*nvmlDeviceGetMaxPcieLinkWidth)(nvmlDevice_t device, unsigned int *maxLinkWidth);

typedef enum {
  NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0,
  NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1,
} nvmlTemperatureThresholds_t;

static nvmlReturn_t (*nvmlDeviceGetTemperatureThreshold)(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType,
                                                         unsigned int *temp);

// Dynamic information extraction

typedef enum {
  NVML_CLOCK_GRAPHICS = 0,
  NVML_CLOCK_SM = 1,
  NVML_CLOCK_MEM = 2,
  NVML_CLOCK_VIDEO = 3,
} nvmlClockType_t;

static nvmlReturn_t (*nvmlDeviceGetClockInfo)(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);

static nvmlReturn_t (*nvmlDeviceGetMaxClockInfo)(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);

typedef struct {
  unsigned int gpu;
  unsigned int memory;
} nvmlUtilization_t;

static nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(nvmlDevice_t device, nvmlUtilization_t *utilization);

typedef struct {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
} nvmlMemory_t;

static nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t device, nvmlMemory_t *memory);

static nvmlReturn_t (*nvmlDeviceGetCurrPcieLinkGeneration)(nvmlDevice_t device, unsigned int *currLinkGen);

static nvmlReturn_t (*nvmlDeviceGetCurrPcieLinkWidth)(nvmlDevice_t device, unsigned int *currLinkWidth);

typedef enum {
  NVML_PCIE_UTIL_TX_BYTES = 0,
  NVML_PCIE_UTIL_RX_BYTES = 1,
} nvmlPcieUtilCounter_t;

static nvmlReturn_t (*nvmlDeviceGetPcieThroughput)(nvmlDevice_t device, nvmlPcieUtilCounter_t counter,
                                                   unsigned int *value);

static nvmlReturn_t (*nvmlDeviceGetFanSpeed)(nvmlDevice_t device, unsigned int *speed);

typedef enum {
  NVML_TEMPERATURE_GPU = 0,
} nvmlTemperatureSensors_t;

static nvmlReturn_t (*nvmlDeviceGetTemperature)(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType,
                                                unsigned int *temp);

static nvmlReturn_t (*nvmlDeviceGetPowerUsage)(nvmlDevice_t device, unsigned int *power);

static nvmlReturn_t (*nvmlDeviceGetEnforcedPowerLimit)(nvmlDevice_t device, unsigned int *limit);

static nvmlReturn_t (*nvmlDeviceGetEncoderUtilization)(nvmlDevice_t device, unsigned int *utilization,
                                                       unsigned int *samplingPeriodUs);

static nvmlReturn_t (*nvmlDeviceGetDecoderUtilization)(nvmlDevice_t device, unsigned int *utilization,
                                                       unsigned int *samplingPeriodUs);

// Processes running on GPU

typedef struct {
  unsigned int pid;
  unsigned long long usedGpuMemory;
  // unsigned int gpuInstanceId;      // not supported by older NVIDIA drivers
  // unsigned int computeInstanceId;  // not supported by older NVIDIA drivers
} nvmlProcessInfo_t;

static nvmlReturn_t (*nvmlDeviceGetGraphicsRunningProcesses)(nvmlDevice_t device, unsigned int *infoCount,
                                                             nvmlProcessInfo_t *infos);

static nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses)(nvmlDevice_t device, unsigned int *infoCount,
                                                            nvmlProcessInfo_t *infos);

static void *libnvidia_ml_handle;

static nvmlReturn_t last_nvml_return_status = NVML_SUCCESS;
static char didnt_call_gpuinfo_init[] = "The NVIDIA extraction has not been initialized, please call "
                                        "gpuinfo_nvidia_init\n";
static const char *local_error_string = didnt_call_gpuinfo_init;

// Processes GPU Utilization

typedef struct {
  unsigned int pid;
  unsigned long long timeStamp;
  unsigned int smUtil;
  unsigned int memUtil;
  unsigned int encUtil;
  unsigned int decUtil;
} nvmlProcessUtilizationSample_t;

nvmlReturn_t (*nvmlDeviceGetProcessUtilization)(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization,
                                                unsigned int *processSamplesCount,
                                                unsigned long long lastSeenTimeStamp);

static bool gpuinfo_nvidia_init(void);
static void gpuinfo_nvidia_shutdown(void);

/*
 *
 * This function loads the libnvidia-ml.so shared object, initializes the
 * required function pointers and calls the nvidia library initialization
 * function. Returns true if everything has been initialized successfully. If
 * false is returned, the cause of the error can be retrieved by calling the
 * function gpuinfo_nvidia_last_error_string.
 *
 */
static bool gpuinfo_nvidia_init(void) {

  libnvidia_ml_handle = dlopen("libnvidia-ml.so", RTLD_LAZY);
  if (!libnvidia_ml_handle)
    libnvidia_ml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  if (!libnvidia_ml_handle) {
    local_error_string = dlerror();
    return false;
  }

  // Default to last version
  nvmlInit = dlsym(libnvidia_ml_handle, "nvmlInit_v2");
  if (!nvmlInit)
    nvmlInit = dlsym(libnvidia_ml_handle, "nvmlInit");
  if (!nvmlInit)
    goto init_error_clean_exit;

  nvmlShutdown = dlsym(libnvidia_ml_handle, "nvmlShutdown");
  if (!nvmlShutdown)
    goto init_error_clean_exit;

  // Default to last version if available
  nvmlDeviceGetCount = dlsym(libnvidia_ml_handle, "nvmlDeviceGetCount_v2");
  if (!nvmlDeviceGetCount)
    nvmlDeviceGetCount = dlsym(libnvidia_ml_handle, "nvmlDeviceGetCount");
  if (!nvmlDeviceGetCount)
    goto init_error_clean_exit;

  nvmlDeviceGetHandleByIndex = dlsym(libnvidia_ml_handle, "nvmlDeviceGetHandleByIndex_v2");
  if (!nvmlDeviceGetHandleByIndex)
    nvmlDeviceGetHandleByIndex = dlsym(libnvidia_ml_handle, "nvmlDeviceGetHandleByIndex");
  if (!nvmlDeviceGetHandleByIndex)
    goto init_error_clean_exit;

  nvmlErrorString = dlsym(libnvidia_ml_handle, "nvmlErrorString");
  if (!nvmlErrorString)
    goto init_error_clean_exit;

  nvmlDeviceGetName = dlsym(libnvidia_ml_handle, "nvmlDeviceGetName");
  if (!nvmlDeviceGetName)
    goto init_error_clean_exit;

  nvmlDeviceGetPciInfo = dlsym(libnvidia_ml_handle, "nvmlDeviceGetPciInfo_v3");
  if (!nvmlDeviceGetPciInfo)
    nvmlDeviceGetPciInfo = dlsym(libnvidia_ml_handle, "nvmlDeviceGetPciInfo_v2");
  if (!nvmlDeviceGetPciInfo)
    nvmlDeviceGetPciInfo = dlsym(libnvidia_ml_handle, "nvmlDeviceGetPciInfo");
  if (!nvmlDeviceGetPciInfo)
    goto init_error_clean_exit;

  nvmlDeviceGetMaxPcieLinkGeneration = dlsym(libnvidia_ml_handle, "nvmlDeviceGetMaxPcieLinkGeneration");
  if (!nvmlDeviceGetMaxPcieLinkGeneration)
    goto init_error_clean_exit;

  nvmlDeviceGetMaxPcieLinkWidth = dlsym(libnvidia_ml_handle, "nvmlDeviceGetMaxPcieLinkWidth");
  if (!nvmlDeviceGetMaxPcieLinkWidth)
    goto init_error_clean_exit;

  nvmlDeviceGetTemperatureThreshold = dlsym(libnvidia_ml_handle, "nvmlDeviceGetTemperatureThreshold");
  if (!nvmlDeviceGetTemperatureThreshold)
    goto init_error_clean_exit;

  nvmlDeviceGetClockInfo = dlsym(libnvidia_ml_handle, "nvmlDeviceGetClockInfo");
  if (!nvmlDeviceGetClockInfo)
    goto init_error_clean_exit;

  nvmlDeviceGetMaxClockInfo = dlsym(libnvidia_ml_handle, "nvmlDeviceGetMaxClockInfo");
  if (!nvmlDeviceGetMaxClockInfo)
    goto init_error_clean_exit;

  nvmlDeviceGetUtilizationRates = dlsym(libnvidia_ml_handle, "nvmlDeviceGetUtilizationRates");
  if (!nvmlDeviceGetUtilizationRates)
    goto init_error_clean_exit;

  nvmlDeviceGetMemoryInfo = dlsym(libnvidia_ml_handle, "nvmlDeviceGetMemoryInfo");
  if (!nvmlDeviceGetMemoryInfo)
    goto init_error_clean_exit;

  nvmlDeviceGetCurrPcieLinkGeneration = dlsym(libnvidia_ml_handle, "nvmlDeviceGetCurrPcieLinkGeneration");
  if (!nvmlDeviceGetCurrPcieLinkGeneration)
    goto init_error_clean_exit;

  nvmlDeviceGetCurrPcieLinkWidth = dlsym(libnvidia_ml_handle, "nvmlDeviceGetCurrPcieLinkWidth");
  if (!nvmlDeviceGetCurrPcieLinkWidth)
    goto init_error_clean_exit;

  nvmlDeviceGetPcieThroughput = dlsym(libnvidia_ml_handle, "nvmlDeviceGetPcieThroughput");
  if (!nvmlDeviceGetPcieThroughput)
    goto init_error_clean_exit;

  nvmlDeviceGetFanSpeed = dlsym(libnvidia_ml_handle, "nvmlDeviceGetFanSpeed");
  if (!nvmlDeviceGetFanSpeed)
    goto init_error_clean_exit;

  nvmlDeviceGetTemperature = dlsym(libnvidia_ml_handle, "nvmlDeviceGetTemperature");
  if (!nvmlDeviceGetTemperature)
    goto init_error_clean_exit;

  nvmlDeviceGetPowerUsage = dlsym(libnvidia_ml_handle, "nvmlDeviceGetPowerUsage");
  if (!nvmlDeviceGetPowerUsage)
    goto init_error_clean_exit;

  nvmlDeviceGetEnforcedPowerLimit = dlsym(libnvidia_ml_handle, "nvmlDeviceGetEnforcedPowerLimit");
  if (!nvmlDeviceGetEnforcedPowerLimit)
    goto init_error_clean_exit;

  nvmlDeviceGetEncoderUtilization = dlsym(libnvidia_ml_handle, "nvmlDeviceGetEncoderUtilization");
  if (!nvmlDeviceGetEncoderUtilization)
    goto init_error_clean_exit;

  nvmlDeviceGetDecoderUtilization = dlsym(libnvidia_ml_handle, "nvmlDeviceGetDecoderUtilization");
  if (!nvmlDeviceGetDecoderUtilization)
    goto init_error_clean_exit;

  nvmlDeviceGetGraphicsRunningProcesses = dlsym(libnvidia_ml_handle, "nvmlDeviceGetGraphicsRunningProcesses");
  if (!nvmlDeviceGetGraphicsRunningProcesses)
    goto init_error_clean_exit;

  nvmlDeviceGetComputeRunningProcesses = dlsym(libnvidia_ml_handle, "nvmlDeviceGetComputeRunningProcesses");
  if (!nvmlDeviceGetComputeRunningProcesses)
    goto init_error_clean_exit;

  // This one might not be available
  nvmlDeviceGetProcessUtilization = dlsym(libnvidia_ml_handle, "nvmlDeviceGetProcessUtilization");

  last_nvml_return_status = nvmlInit();
  std::cout << last_nvml_return_status;
  if (last_nvml_return_status != NVML_SUCCESS) {
    return false;
  }
  local_error_string = NULL;

  return true;

init_error_clean_exit:
  dlclose(libnvidia_ml_handle);
  libnvidia_ml_handle = NULL;
  return false;
}

static void gpuinfo_nvidia_shutdown(void) {
  if (libnvidia_ml_handle) {
    nvmlShutdown();
    dlclose(libnvidia_ml_handle);
    libnvidia_ml_handle = NULL;
    local_error_string = didnt_call_gpuinfo_init;
  }

}


int main()
{
    if(!gpuinfo_nvidia_init())
    {
        std::cout << last_nvml_return_status;
    }

    unsigned int count;
    nvmlReturn_t ret;
    if((ret = nvmlDeviceGetCount(&count)) != NVML_SUCCESS)
    {
        std::cout << ret << std::endl;
    }
    gpuinfo_nvidia_shutdown();
    return 0;
}
