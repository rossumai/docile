install:
	echo "Installing binary dependencies"
	brew install poppler cairo pango gdk-pixbuf libffi

	echo "Installing poetry"
	poetry install --with test --with dev

	echo "Installing pre-commit"
	pre-commit install


test:
	poetry run pytest
