// serenityboardd serve --logdir DIR [--port 6006] [--host 0.0.0.0] [--frontend DIR]
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "serenityboard/app.hpp"

namespace {
sb::App *g_app = nullptr;
void on_signal(int) {
  if (g_app) g_app->stop();
}
std::string default_frontend() {
  // native/build/serenityboardd -> ../../serenityboard/frontend
  std::error_code ec;
  const auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  for (auto dir = exe.parent_path(); !dir.empty() && dir != dir.parent_path(); dir = dir.parent_path()) {
    const auto candidate = dir / "serenityboard" / "frontend";
    if (std::filesystem::is_regular_file(candidate / "index.html", ec)) return candidate.string();
  }
  return "";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 2 || std::string(argv[1]) != "serve") {
    std::fprintf(stderr, "usage: serenityboardd serve --logdir DIR [--port 6006] [--host 0.0.0.0] [--frontend DIR]\n");
    return 1;
  }
  sb::AppOptions options;
  options.frontend_dir = default_frontend();
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--logdir") options.logdir = value();
    else if (a == "--port") options.port = std::stoi(value());
    else if (a == "--host") options.host = value();
    else if (a == "--frontend") options.frontend_dir = value();
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (options.logdir.empty()) {
    std::fprintf(stderr, "--logdir is required\n");
    return 2;
  }
  sb::App app(options);
  g_app = &app;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const int port = app.start();
  if (port == 0) {
    std::fprintf(stderr, "cannot bind %s:%d\n", options.host.c_str(), options.port);
    return 1;
  }
  std::printf("serenityboardd: serving %s on http://%s:%d (frontend: %s)\n", options.logdir.c_str(),
              options.host.c_str(), port, options.frontend_dir.empty() ? "none" : options.frontend_dir.c_str());
  std::fflush(stdout);
  app.run_forever();
  return 0;
}
