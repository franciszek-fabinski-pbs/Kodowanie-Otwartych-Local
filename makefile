LOGFILE= logs/$(shell date +"%d.%m.%Y-%H_%M_%S").log

run: link
	@echo "Running the classification model..."
	@uv run main.py > $(LOGFILE)

test: link
	@echo "Running program with stdout output..."
	@uv run main.py

link:
	@echo "Linking local models to the project..."
	@ln -sf ~/Modele/* models/

log:
	@echo "Showing latest log file..."
	@ls -1 logs | sort | tail -n1 | xargs -I{} less logs/{}

push:
	@echo "pushing project to server..."
	@rsync -avz --exclude='.venv'  ../kodowanie-otwartych-local pbs@tyfus.pbs.local:~/
