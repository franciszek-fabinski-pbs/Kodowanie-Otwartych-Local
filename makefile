run: link
	@echo "Running the local LLM..."
	@uv run main.py

link:
	@echo "Linking local models to the project..."
	@ln -sf ~/Modele/* models/
