LOGFILE= logs/$(shell date +"%d.%m.%Y-%H_%M_%S").log

run: link
	@echo "Running the local LLM..."
	@uv run main.py > $(LOGFILE)

link:
	@echo "Linking local models to the project..."
	@ln -sf ~/Modele/* models/
