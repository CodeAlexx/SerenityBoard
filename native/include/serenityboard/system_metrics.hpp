// GPU/system metrics thread (serenityboard/writer/system_metrics.py):
// nvidia-smi query, /proc loadavg + meminfo; own step counter from 0.
#pragma once

#include <atomic>
#include <thread>

namespace sb {

class SummaryWriter;

class SystemMetricsCollector {
public:
  SystemMetricsCollector(SummaryWriter &writer, double interval_seconds = 10.0, int gpu_index = 0);
  ~SystemMetricsCollector();
  void start();
  void stop();
  /// One poll: logs gpu/* and system/* scalars at `step`. Public for tests.
  void poll_once(long long step);
  double interval() const { return interval_; }

private:
  void loop();
  SummaryWriter &writer_;
  double interval_;
  int gpu_index_;
  std::atomic<bool> running_{false};
  std::thread thread_;
};

}  // namespace sb
