import iris

"""
Core iris processing pipeline and matcher instances for reuse across modules.
"""

iris_pipeline = iris.IRISPipeline()
matcher = iris.HammingDistanceMatcher()
