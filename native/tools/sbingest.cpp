// sbingest — put Diffusion Compiler outputs into a SerenityBoard run.
//   sbingest --logdir DIR --run NAME [--resume-step N] [--hparams JSON] SOURCE...
// SOURCE is one of:
//   steps:FILE    stdout log with H3_STEP / FLUX2_NATIVE_STEP / KREA2_NATIVE_STEP / KREA2_STEP lines ('-' = stdin)
//   report:FILE   --report JSON (difflux2sample, difkrea2sample) or a telemetry document (difbench, difregress, difprobe)
//   jsonl:FILE    DIF_TRACE_FILE runtime traces or training-report.jsonl
//   losses:FILE   losses.diftensor (F32 vector) -> loss/train
//   FILE          extension-based guess: .jsonl -> jsonl, .json -> report, .diftensor -> losses, else steps
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "serenityboard/compiler_ingest.hpp"

int main(int argc, char **argv) {
  std::string logdir, run;
  std::optional<long long> resume;
  std::optional<nlohmann::json> hparams;
  std::vector<std::string> sources;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s requires a value\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (a == "--logdir") logdir = value();
    else if (a == "--run") run = value();
    else if (a == "--resume-step") resume = std::stoll(value());
    else if (a == "--hparams") hparams = nlohmann::json::parse(value());
    else sources.push_back(a);
  }
  if (logdir.empty() || run.empty() || sources.empty()) {
    std::fprintf(stderr, "usage: sbingest --logdir DIR --run NAME [--resume-step N] [--hparams JSON] (steps:|report:|jsonl:|losses:)FILE...\n");
    return 2;
  }
  try {
    sb::WriterOptions options;
    options.run_name = run;
    options.hparams = hparams;
    options.resume_step = resume;
    options.system_metrics = false;
    sb::SummaryWriter writer(logdir, options);
    sb::ingest::Counts total;
    for (const auto &source : sources) {
      std::string kind, path;
      const auto colon = source.find(':');
      if (colon != std::string::npos && (source.compare(0, colon, "steps") == 0 || source.compare(0, colon, "report") == 0 ||
                                         source.compare(0, colon, "jsonl") == 0 || source.compare(0, colon, "losses") == 0)) {
        kind = source.substr(0, colon);
        path = source.substr(colon + 1);
      } else {
        path = source;
        if (path.size() > 6 && path.compare(path.size() - 6, 6, ".jsonl") == 0) kind = "jsonl";
        else if (path.size() > 5 && path.compare(path.size() - 5, 5, ".json") == 0) kind = "report";
        else if (path.size() > 10 && path.compare(path.size() - 10, 10, ".diftensor") == 0) kind = "losses";
        else kind = "steps";
      }
      sb::ingest::Counts c;
      if (kind == "steps") {
        if (path == "-") c = sb::ingest::ingest_step_lines(writer, std::cin);
        else {
          std::ifstream in(path);
          if (!in) throw std::runtime_error("cannot open " + path);
          c = sb::ingest::ingest_step_lines(writer, in);
        }
      } else if (kind == "report") {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open " + path);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        c = sb::ingest::ingest_report_json(writer, text, path);
      } else if (kind == "jsonl") {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open " + path);
        c = sb::ingest::ingest_jsonl(writer, in, path);
      } else {
        c = sb::ingest::ingest_losses_diftensor(writer, path);
      }
      std::printf("ingested %s (%s): scalars=%lld traces=%lld texts=%lld evals=%lld\n", path.c_str(), kind.c_str(), c.scalars, c.traces, c.texts, c.evals);
      total.scalars += c.scalars; total.traces += c.traces; total.texts += c.texts; total.evals += c.evals;
    }
    writer.flush();
    writer.close();
    std::printf("run %s in %s: scalars=%lld traces=%lld texts=%lld evals=%lld\n", run.c_str(), logdir.c_str(), total.scalars, total.traces, total.texts, total.evals);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "sbingest: %s\n", e.what());
    return 1;
  }
}
