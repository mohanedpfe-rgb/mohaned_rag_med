"""Install production compatibility contracts before pytest imports test modules."""

from rag_project.runtime import install as install_runtime

install_runtime()
