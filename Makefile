install:
	echo "Installing binary dependencies"
	brew install poetry poppler cairo pango gdk-pixbuf libffi

	echo "Installing poetry dependencies"
	poetry install --with test --with dev

	echo "Installing pre-commit"
	pre-commit install

test:
	poetry run pytest

jupyter:
	poetry run jupyter notebook
