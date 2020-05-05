using PkgBenchmark
#results = benchmarkpkg("McCormick")
#show(results)

# specify tag and uncommit to benchmark versus prior tagged version
tag = "9b5ba7d756e3feac2a9ab5d42af464f1139b8900" # v0.4.1
results = judge("McCormick", tag)
show(results)

export_markdown("results.md", results)
