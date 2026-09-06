#include "serenityboard/system_metrics.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "serenityboard/summary_writer.hpp"

namespace sb {

SystemMetricsCollector::SystemMetricsCollector(SummaryWriter &writer, double interval_seconds, int gpu_index)
    : writer_(writer), interval_(std::max(interval_seconds, 5.0)), gpu_index_(gpu_index) {}

SystemMetricsCollector::~SystemMetricsCollector() { stop(); }

void SystemMetricsCollector::start() {
  if (running_.exchange(true)) return;
  thread_ = std::thread([this] { loop(); });
}

void SystemMetricsCollector::stop() {
  running_ = false;
  if (thread_.joinable()) thread_.join();
}

void SystemMetricsCollector::loop() {
  long long step = 0;
  while (running_) {
    try {
      poll_once(step);
    } catch (...) {
    }
    ++step;
    // sleep in small slices so stop() returns promptly
    const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(interval_);
    while (running_ && std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void SystemMetricsCollector::poll_once(long long step) {
  // GPU via nvidia-smi (pynvml has no C++ analogue here; NVML linkage is optional work).
  {
    std::string cmd = "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total "
                      "--format=csv,noheader,nounits -i " + std::to_string(gpu_index_) + " 2>/dev/null";
    if (FILE *p = popen(cmd.c_str(), "r")) {
      char buf[256];
      std::string line;
      if (std::fgets(buf, sizeof buf, p)) line = buf;
      const int rc = pclose(p);
      if (rc == 0 && !line.empty()) {
        std::stringstream ss(line);
        std::string util, temp, used, total;
        if (std::getline(ss, util, ',') && std::getline(ss, temp, ',') && std::getline(ss, used, ',') &&
            std::getline(ss, total, ',')) {
          try {
            writer_.add_scalar("gpu/utilization", std::stod(util), step);
            writer_.add_scalar("gpu/temperature", std::stod(temp), step);
            writer_.add_scalar("gpu/memory_used", std::stod(used) / 1024.0, step);
            writer_.add_scalar("gpu/memory_total", std::stod(total) / 1024.0, step);
          } catch (...) {
          }
        }
      }
    }
  }
  // System via /proc (the Python fallback path).
  {
    std::ifstream loadavg("/proc/loadavg");
    double load1 = 0;
    if (loadavg >> load1) {
      const long ncpu = std::max(1L, sysconf(_SC_NPROCESSORS_ONLN));
      writer_.add_scalar("system/cpu_percent", std::min(load1 / static_cast<double>(ncpu) * 100.0, 100.0), step);
    }
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    double total_kb = 0, avail_kb = 0;
    while (meminfo >> key) {
      double value = 0;
      std::string unit;
      meminfo >> value >> unit;
      if (key == "MemTotal:") total_kb = value;
      else if (key == "MemAvailable:") avail_kb = value;
    }
    if (total_kb > 0) {
      writer_.add_scalar("system/ram_used_gb", (total_kb - avail_kb) / (1024.0 * 1024.0), step);
      writer_.add_scalar("system/ram_total_gb", total_kb / (1024.0 * 1024.0), step);
    }
  }
}

}  // namespace sb
