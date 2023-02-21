.DEFAULT_GOAL := help

##@ Utility
help: ## Display this help. (Default)
# based on "https://gist.github.com/prwhite/8168133?permalink_comment_id=4260260#gistcomment-4260260"
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

##@ Utility
help_sort: ## Display alphabetized version of help.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

conda-lock: ## Regenerate conda lockfile from environment file.
	conda-lock -f environment.yml --lockfile conda-lock.yml

install-conda: ## Install conda environment from lockfile.
	conda-lock install -n api conda-lock.yml

install-kernel: ## Instal jupyter kernel for conda environment.
	/opt/conda/envs/api/bin/python -m ipykernel install --prefix=/opt/conda/ --name=api