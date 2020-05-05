using PkgBenchmark
#results = benchmarkpkg("McCormick")
#show(results)

# specify tag and uncommit to benchmark versus prior tagged version
tag = "v0.4.0"
results = judge("McCormick", tag)
show(results)

export_markdown("results.md", results)
