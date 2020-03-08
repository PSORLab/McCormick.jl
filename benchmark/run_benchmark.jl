using PkgBenchmark
results = benchmarkpkg("McCormick")
show(results)

#=
# specify tag and uncommit to benchmark versus prior tagged version
tag =
results = judge("EAGO", tag)
show(results)
=#

export_markdown("results.md", results)
