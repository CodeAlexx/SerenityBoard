// Diffusion Compiler -> SerenityBoard ingest: turns what the compiler's tools
// already emit (stdout step lines, --report JSON, difbench documents,
// runtime-trace JSON lines, training-report JSON lines, losses.diftensor)
// into board.db rows through the native SummaryWriter. The dashboard is a
// consumer of the runtime's telemetry, never a second telemetry origin.
#pragma once

#include <istream>
#include <string>

#include "serenityboard/summary_writer.hpp"

namespace sb {
using Json = nlohmann::json;
}

namespace sb::ingest {

struct Counts {
  long long scalars{0}, traces{0}, texts{0}, evals{0};
};

/// `H3_STEP index=..`, `FLUX2_NATIVE_STEP step=n/N ..`, `KREA2_NATIVE_STEP step=n/N ..`,
/// `KREA2_STEP n/max loss= ms= ..`, `H3_PROFILE key=value ..`, `*_PASS` receipts.
Counts ingest_step_lines(SummaryWriter &w, std::istream &in);
/// difflux2sample / difkrea2sample hand-rolled reports (no schema head) and
/// schema-headed telemetry documents (difbench `benchmark`, `regression-report`,
/// `device-probe`, `runtime-trace`).
Counts ingest_report_json(SummaryWriter &w, const std::string &json_text, const std::string &source_name);
/// One JSON document per line: runtime-trace sinks (DIF_TRACE_FILE) and
/// training-report.jsonl (TrainingStepReport).
Counts ingest_jsonl(SummaryWriter &w, std::istream &in, const std::string &source_name);
/// DIFTNS01 F32 vector of per-step losses -> scalars `loss/train`.
Counts ingest_losses_diftensor(SummaryWriter &w, const std::string &path, const std::string &tag = "loss/train");

/// Parse `key=value key2=value2` after a leading token; values may be quoted.
std::map<std::string, std::string> parse_kv_line(const std::string &line);

}  // namespace sb::ingest
