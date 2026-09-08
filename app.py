from rag_project.runtime_hardening import install as install_runtime_hardening
from rag_project.runtime_hardening_extra import install as install_runtime_hardening_extra
from rag_project.runtime_recovery import install as install_runtime_recovery
from rag_project.runtime_quality_gate import install as install_runtime_quality_gate
from rag_project.runtime_final_gate import install as install_runtime_final_gate

install_runtime_hardening()
install_runtime_hardening_extra()
install_runtime_recovery()
install_runtime_quality_gate()
install_runtime_final_gate()

from rag_project.app.dev_ui import main

if __name__ == "__main__":
    main()
