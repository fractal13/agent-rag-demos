VENV := .virtual_environment

all: help

help:
	@echo
	@echo "Targets:"
	@echo "install                     - Install environment necessary to support this project."
	@echo "install-deb                 - Install OS packages necessary to support this project. Assumes apt/dpkg package management system."
	@echo "install-pip                 - Install Python pakcages necessary to suport this project."
	@echo

$(VENV):
	python3 -m venv $(VENV)

install: install-deb install-pip

install-deb:
	@echo python3.12-venv is necessary for venv.
	@echo ffmpeg is necessary to read audio files for ASR
	for package in python3.12-venv ffmpeg; do \
		dpkg -l | egrep '^ii *'$${package}' ' 2>&1 > /dev/null || sudo apt install $${package}; \
	done

install-pip: $(VENV)
	. $(VENV)/bin/activate; pip3 install --upgrade -r requirements.txt

code-agent-rag-demo:
	. $(VENV)/bin/activate; src/code_agent_rag_demo.py

code-agent-rag-specific-demo:
	. $(VENV)/bin/activate; src/code_agent_rag_specific_demo.py

code-agent-rag-knowledge-base-demo:
	. $(VENV)/bin/activate; src/code_agent_rag_knowledge_base_demo.py

code-agent-rag-kb-specific-demo:
	. $(VENV)/bin/activate; src/code_agent_rag_kb_specific_demo.py
